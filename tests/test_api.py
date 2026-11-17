"""Kontrak HTTP inferensi: input lengkap, fitur kurang, dan nilai non-finite."""
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
import pytest

from src.api import app as serving


class Model:
    metadata = SimpleNamespace(get_input_schema=lambda: SimpleNamespace(
        input_names=lambda: ["pm2_5"],
        inputs=[SimpleNamespace(name="pm2_5", type=SimpleNamespace(name="double"))]))

    def predict(self, frame):
        if "hour" in frame:
            assert str(frame["hour"].dtype) == "int64"
        return np.array([12.5] * len(frame))


@pytest.fixture
def client(monkeypatch):
    version = SimpleNamespace(version="1", run_id="test-run")
    monkeypatch.setattr(serving, "initialize_model", lambda: (Model(), version))
    with TestClient(serving.app) as client:
        yield client


def test_predict_returns_model_output(client):
    response = client.post("/predict", json={"instances": [{"pm2_5": 35.0}]})
    assert response.status_code == 200
    assert response.json() == {"predictions": [12.5], "unit": "ug/m3",
                               "horizon_hours": 3, "model_version": "1"}


def test_missing_feature_rejected(client):
    response = client.post("/predict", json={"instances": [{}]})
    assert response.status_code == 422
    assert response.json()["detail"]["missing"] == ["pm2_5"]


def test_empty_batch_and_non_finite_rejected(client):
    assert client.post("/predict", json={"instances": []}).status_code == 422
    assert client.post("/predict", json={"instances": [{"pm2_5": "NaN"}]}).status_code == 422


def test_integer_signature_is_preserved(client):
    client.app.state.columns = ["pm2_5", "hour"]
    client.app.state.types = {"pm2_5": "double", "hour": "long"}
    assert client.post("/predict", json={"instances": [{"pm2_5": 35, "hour": 8.0}]}).status_code == 200
    assert client.post("/predict", json={"instances": [{"pm2_5": 35, "hour": 8.5}]}).status_code == 422
