"""Verifikasi langsung dua layanan, inferensi, lineage, dan named volume."""
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlflow import MlflowClient
from src.features.schema import PROTOCOL
from src.models.train import load_data, verify_data_version

BASE = "http://localhost:18000"


def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(BASE+path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.load(response)


def ready():
    for _ in range(180):
        try:
            return request("/health")
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    raise RuntimeError("API belum siap setelah 180 detik")


def main():
    before = ready()
    client = MlflowClient(tracking_uri="http://localhost:5000")
    version = client.get_model_version_by_alias("pm25-xgboost", "production")
    assert before["run_id"] == version.run_id
    run = client.get_run(version.run_id)
    assert run.data.tags["data_version_md5"] == verify_data_version("data/processed/features.csv")
    assert run.data.tags["raw_data_version_md5"] == verify_data_version("data/raw/air_quality_raw.csv")
    assert run.data.tags["evaluation_protocol"] == PROTOCOL
    subprocess.run(["docker", "compose", "exec", "-T", "api-service", "python", "-c",
                    "import urllib.request; print(urllib.request.urlopen('http://mlflow-server:5000/health').read().decode())"],
                   check=True)
    X, _ = load_data("data/processed/features.csv")
    prediction = request("/predict", {"instances": X.tail(1).to_dict(orient="records")})
    assert len(prediction["predictions"]) == 1 and prediction["horizon_hours"] == 3
    try:
        request("/predict", {"instances": [{}]})
    except urllib.error.HTTPError as error:
        assert error.code == 422
    else:
        raise AssertionError("Permintaan tanpa fitur tidak ditolak")
    subprocess.run(["docker", "compose", "down"], check=True)
    subprocess.run(["docker", "compose", "up", "-d", "--wait", "--wait-timeout", "240"], check=True)
    after = ready()
    assert before["run_id"] == after["run_id"]
    assert before["model_version"] == after["model_version"]
    logs = subprocess.check_output(["docker", "compose", "logs", "--no-log-prefix", "api-service"], text=True)
    assert "Registry baru/lineage berubah:" not in logs
    print(json.dumps({"before": before, "after": after, "prediction": prediction,
                      "lineage": "passed", "persistence": "passed"}, indent=2))


if __name__ == "__main__":
    main()
