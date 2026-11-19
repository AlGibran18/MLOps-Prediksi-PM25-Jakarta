"""Uji /invocations lokal atau seluruh replika terhadap snapshot model yang sama."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

import mlflow.pyfunc
import numpy as np
import pandas as pd


def invoke(base, payload):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(90):
        try:
            with opener.open(base + "/ping", timeout=5) as response:
                if response.status == 200:
                    break
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(2)
    else:
        raise RuntimeError(f"Serving belum siap: {base}")
    request = urllib.request.Request(base + "/invocations", json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
    with opener.open(request, timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Uji server lokal; tanpa opsi ini uji semua replika Compose")
    parser.add_argument("--expected", type=int, default=3)
    args = parser.parse_args()
    payload = json.loads(Path("build/serving/request.json").read_text())
    split = payload["dataframe_split"]
    model = mlflow.pyfunc.load_model("build/serving/model")
    expected = model.predict(pd.DataFrame(split["data"], columns=split["columns"]))
    if args.url:
        endpoints = [("local", args.url)]
    else:
        ids = subprocess.check_output(["docker", "compose", "--profile", "serving", "ps", "-q", "model-serving"], text=True).split()
        if len(ids) != args.expected:
            raise AssertionError(f"Diharapkan {args.expected} replika, ditemukan {len(ids)}")
        containers = json.loads(subprocess.check_output(["docker", "inspect", *ids], text=True))
        if len({item["Image"] for item in containers}) != 1:
            raise AssertionError("Replika tidak menggunakan image yang sama")
        endpoints = []
        for item in containers:
            if item["State"].get("Health", {}).get("Status") != "healthy":
                raise AssertionError(f"Replika belum healthy: {item['Name']}")
            binding = item["NetworkSettings"]["Ports"]["8080/tcp"][0]
            endpoints.append((item["Name"], "http://127.0.0.1:" + binding["HostPort"]))
    for name, url in endpoints:
        result = invoke(url, payload)
        actual = np.asarray(result["predictions"], dtype=float)
        if actual.shape != np.asarray(expected).shape or not np.isfinite(actual).all():
            raise AssertionError(f"Respons prediksi tidak valid: {result}")
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        print(json.dumps({"service": name, "url": url, "response": result, "snapshot_match": True}))


if __name__ == "__main__":
    main()
