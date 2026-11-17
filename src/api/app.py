"""API inferensi LK-09; metadata dan artefak model diambil dari MLflow via HTTP."""
from contextlib import asynccontextmanager
import logging
import os
import subprocess
import sys

from fastapi import FastAPI, HTTPException
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
import pandas as pd
from pydantic import BaseModel, Field, FiniteFloat

LOGGER = logging.getLogger("uvicorn.error")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "pm25-xgboost")
MODEL_URI = f"models:/{MODEL_NAME}@production"


def initialize_model():
    """Registry Compose baru diisi sekali menggunakan pipeline dan snapshot proyek."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, "production")
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        LOGGER.info("Registry baru: menyiapkan model dari snapshot DVC proyek.")
        run_name = "lk09-compose-bootstrap"
        commands = [
            ["src/models/train.py", "--run_name", run_name, "--n_estimators", "100",
             "--learning_rate", "0.08", "--max_depth", "3", "--subsample", "0.9",
             "--colsample_bytree", "0.9"],
            ["src/models/evaluate_model.py", "--run_name", run_name],
            ["src/models/registry_ops.py", "register", "--run_name", run_name],
            ["src/models/registry_ops.py", "promote", "--best", "--stage", "Production"],
        ]
        for script, *args in commands:
            subprocess.run([sys.executable, script, "--tracking_uri", TRACKING_URI, *args], check=True)
        version = client.get_model_version_by_alias(MODEL_NAME, "production")
    # Pin versi agar identitas pada health sama dengan artefak yang dimuat.
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version.version}")
    LOGGER.info("Model berhasil dimuat dari %s: %s versi %s, run_id=%s",
                TRACKING_URI, MODEL_NAME, version.version, version.run_id)
    return model, version


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.version = initialize_model()
    schema = app.state.model.metadata.get_input_schema()
    app.state.columns = schema.input_names()
    app.state.types = {field.name: field.type.name for field in schema.inputs}
    yield


app = FastAPI(title="API Prediksi PM2.5 Jakarta (LK-09)", lifespan=lifespan)


class PredictionRequest(BaseModel):
    instances: list[dict[str, FiniteFloat]] = Field(min_length=1)


@app.get("/health")
def health():
    return {"status": "ok", "tracking_uri": TRACKING_URI, "model_uri": MODEL_URI,
            "model_version": app.state.version.version, "run_id": app.state.version.run_id}


@app.post("/predict")
def predict(payload: PredictionRequest):
    required = set(app.state.columns)
    for i, row in enumerate(payload.instances):
        if set(row) != required:
            raise HTTPException(status_code=422, detail={"row": i,
                                "missing": sorted(required - set(row)),
                                "extra": sorted(set(row) - required)})
    frame = pd.DataFrame(payload.instances, columns=app.state.columns).astype(float)
    for column, kind in app.state.types.items():
        if kind in ("integer", "long"):
            values = frame[column].tolist()
            if any(not value.is_integer() for value in values):
                raise HTTPException(status_code=422, detail=f"{column} harus bilangan bulat")
            try:
                frame[column] = pd.Series([int(value) for value in values],
                                          dtype="int64" if kind == "long" else "int32")
            except (OverflowError, ValueError):
                raise HTTPException(status_code=422, detail=f"{column} di luar rentang integer")
    predictions = app.state.model.predict(frame)
    return {"predictions": [float(value) for value in predictions],
            "unit": "ug/m3", "horizon_hours": 3,
            "model_version": app.state.version.version}
