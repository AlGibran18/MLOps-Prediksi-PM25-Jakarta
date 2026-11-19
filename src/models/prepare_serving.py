"""Pin model Production dan siapkan input standar MLflow Models."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import mlflow
from mlflow import MlflowClient
import yaml

from src.features.schema import PROTOCOL
from src.models.evaluate_model import check_thresholds
from src.models.train import load_data, verify_data_version


def validate_production(version, run, raw_md5, features_md5, thresholds):
    if version.current_stage != "Production" or run.info.status != "FINISHED":
        raise ValueError("Alias production harus menunjuk model Production dari run FINISHED")
    expected = {"data_version_md5": features_md5, "raw_data_version_md5": raw_md5,
                "evaluation_protocol": PROTOCOL}
    if any(run.data.tags.get(key) != value for key, value in expected.items()):
        raise ValueError("Lineage model Production tidak cocok dengan dataset/protokol aktif")
    if not math.isfinite(run.data.metrics.get("val_rmse", float("inf"))):
        raise ValueError("Model Production tidak memiliki validation RMSE yang valid")
    violations = check_thresholds(run.data.metrics, thresholds)
    if violations:
        raise ValueError("Model Production gagal gerbang kualitas: " + "; ".join(violations))


def prepare(tracking_uri, output):
    raw_md5 = verify_data_version("data/raw/air_quality_raw.csv")
    features_md5 = verify_data_version("data/processed/features.csv")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    version = client.get_model_version_by_alias("pm25-xgboost", "production")
    run = client.get_run(version.run_id)
    thresholds = yaml.safe_load(Path("configs/thresholds.yaml").read_text())
    validate_production(version, run, raw_md5, features_md5, thresholds)
    # Resolve alias sekali, lalu gunakan versi eksplisit saat mengunduh artefak.
    model_uri = f"models:/pm25-xgboost/{version.version}"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        downloaded = mlflow.artifacts.download_artifacts(artifact_uri=model_uri,
                                                        dst_path=temporary)
        destination = output / "model"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(downloaded, destination)
    X, _ = load_data("data/processed/features.csv")
    request = {"dataframe_split": json.loads(X.tail(1).to_json(orient="split", index=False))}
    (output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    reference = {"model_uri": model_uri, "model_name": "pm25-xgboost",
                 "model_version": str(version.version), "run_id": version.run_id,
                 "stage": version.current_stage, "raw_md5": raw_md5,
                 "features_md5": features_md5, "protocol": PROTOCOL}
    (output / "model_reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    print(json.dumps(reference, indent=2))
    print(f"Snapshot dan request MLflow siap pada {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    parser.add_argument("--output", default="build/serving")
    args = parser.parse_args()
    prepare(args.tracking_uri, args.output)


if __name__ == "__main__":
    main()
