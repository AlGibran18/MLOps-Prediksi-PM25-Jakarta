"""Verifikasi tiga scrape target, metrik nyata, serta query panel melalui Grafana."""
import argparse
import base64
import json
import math
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prometheus_client.parser import text_string_to_metric_families
from src.observability.load import discover_replicas


def request(url, payload=None, auth=False):
    headers = {"Content-Type": "application/json"}
    if auth:
        credentials = os.getenv("GRAFANA_ADMIN_USER", "admin") + ":" + os.getenv("GRAFANA_ADMIN_PASSWORD", "admin")
        headers["Authorization"] = "Basic " + base64.b64encode(credentials.encode()).decode()
    req = urllib.request.Request(url, None if payload is None else json.dumps(payload).encode(), headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=20) as response:
        body = response.read().decode()
        return json.loads(body) if "json" in response.headers.get("Content-Type", "") else body


def query(expression):
    return request("http://127.0.0.1:9090/api/v1/query?" + urllib.parse.urlencode({"query": expression}))["data"]["result"]


def eventually(check, description):
    deadline = time.monotonic() + 90
    last = None
    while time.monotonic() < deadline:
        try:
            if check():
                return
        except (OSError, KeyError, ValueError) as error:
            last = error
        time.sleep(3)
    raise AssertionError(f"Timeout: {description}; {last}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=int, default=3)
    args = parser.parse_args()
    replicas = discover_replicas(args.expected)
    for name, url in replicas:
        metrics = request(url + "/metrics")
        samples = [sample for family in text_string_to_metric_families(metrics) for sample in family.samples]
        for metric in ("model_inference_requests_total", "model_inference_duration_seconds_count",
                       "model_prediction_pm25_ug_m3_count", "model_serving_process_memory_bytes",
                       "model_serving_process_cpu_seconds_total", "model_serving_info"):
            assert sum(sample.value for sample in samples if sample.name == metric) > 0, (name, metric)
        try:
            request(url + "/invocations", {"dataframe_split": {"columns": [], "data": [[]]}})
        except urllib.error.HTTPError as error:
            assert 400 <= error.code < 500
        else:
            raise AssertionError("Input invalid tidak ditolak")

    def targets_ready():
        targets = request("http://127.0.0.1:9090/api/v1/targets")["data"]["activeTargets"]
        targets = [target for target in targets if target["labels"].get("job") == "model-serving"]
        return len(targets) == args.expected and all(target["health"] == "up" for target in targets)
    eventually(targets_ready, "semua replika UP di Prometheus")

    for metric in ("model_inference_requests_total", "model_prediction_pm25_ug_m3_count",
                   "model_serving_process_memory_bytes", "model_serving_process_cpu_seconds_total"):
        eventually(lambda: len(query(f'sum by (replica) ({metric}{{job="model-serving"}}) > 0')) == args.expected,
                   metric + " pada setiap replika")
    eventually(lambda: len(query('sum by (replica) (rate(model_inference_requests_total{job="model-serving"}[1m])) > 0')) == args.expected,
               "throughput aktif setiap replika")

    health = request("http://127.0.0.1:3000/api/datasources/uid/prometheus/health", auth=True)
    assert health["status"] == "OK", health
    dashboard = request("http://127.0.0.1:3000/api/dashboards/uid/pm25-serving", auth=True)["dashboard"]
    assert len(dashboard["panels"]) == 10
    for panel in dashboard["panels"]:
        for target in panel["targets"]:
            def panel_has_data():
                now = int(time.time() * 1000)
                payload = {"from": str(now - 120000), "to": str(now), "queries": [{
                    "refId": "A", "datasource": {"type": "prometheus", "uid": "prometheus"},
                    "expr": target["expr"], "range": True, "instant": False,
                    "intervalMs": 5000, "maxDataPoints": 120, "format": "time_series"}]}
                result = request("http://127.0.0.1:3000/api/ds/query", payload, auth=True)["results"]["A"]
                assert not result.get("error"), result
                values = [value for frame in result.get("frames", [])
                          for column in frame.get("data", {}).get("values", [])[1:]
                          for value in column]
                return any(isinstance(value, (int, float)) and math.isfinite(value) for value in values)
            eventually(panel_has_data, "data Grafana: " + panel["title"])
    print(json.dumps({"prometheus_targets_up": args.expected, "metrics": "passed",
                      "grafana_datasource": "healthy", "grafana_panels_with_data": len(dashboard["panels"])}, indent=2))


if __name__ == "__main__":
    main()
