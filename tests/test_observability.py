import json

from fastapi.testclient import TestClient
import httpx
from prometheus_client.parser import text_string_to_metric_families
import pytest

from src.observability.gateway import create_app


def sample(client, name):
    for family in text_string_to_metric_families(client.get("/metrics").text):
        for value in family.samples:
            if value.name == name:
                yield value


def total(client, name):
    return sum(value.value for value in sample(client, name))


def test_proxy_preserves_input_output_and_counts_batch_predictions(tmp_path):
    reference = tmp_path / "model_reference.json"
    reference.write_text(json.dumps({"model_name": "pm25-xgboost", "model_version": "7", "run_id": "source-run"}))
    payload = {"dataframe_split": {"columns": ["hour"], "data": [[1], [2]]}}
    def backend(request):
        assert request.url.path == "/invocations"
        assert json.loads(request.content) == payload
        return httpx.Response(200, json={"predictions": [20.0, 40.0]})
    with TestClient(create_app(httpx.MockTransport(backend), reference_path=reference)) as client:
        result = client.post("/invocations", json=payload)
        assert result.json() == {"predictions": [20.0, 40.0]}
        assert total(client, "model_inference_requests_total") == 1
        assert total(client, "model_inference_duration_seconds_count") == 1
        assert total(client, "model_prediction_pm25_ug_m3_count") == 2
        assert total(client, "model_prediction_pm25_ug_m3_sum") == 60
        identity = list(sample(client, "model_serving_info"))[0]
        assert identity.labels["model_version"] == "7"
        assert identity.labels["run_id"] == "source-run"
        assert total(client, "model_serving_process_memory_bytes") > 0


@pytest.mark.parametrize("status", [400, 422, 500])
def test_upstream_errors_are_preserved_and_not_counted_as_predictions(status):
    transport = httpx.MockTransport(lambda _: httpx.Response(status, json={"error": "invalid"}))
    with TestClient(create_app(transport)) as client:
        assert client.post("/invocations", content=b"invalid").status_code == status
        assert total(client, "model_inference_requests_total") == 1
        assert total(client, "model_prediction_pm25_ug_m3_count") == 0
        assert any(value.labels["status"] == str(status) and value.value == 1
                   for value in sample(client, "model_inference_requests_total"))


def test_connection_failure_is_observed_as_502():
    def unavailable(request):
        raise httpx.ConnectError("offline", request=request)
    with TestClient(create_app(httpx.MockTransport(unavailable))) as client:
        assert client.post("/invocations", json={}).status_code == 502
        assert total(client, "model_inference_requests_total") == 1
        assert total(client, "model_inference_duration_seconds_count") == 1
        assert total(client, "model_prediction_pm25_ug_m3_count") == 0


def test_health_and_scrapes_do_not_inflate_inference_throughput():
    transport = httpx.MockTransport(lambda _: httpx.Response(200, text="OK"))
    with TestClient(create_app(transport)) as client:
        assert client.get("/ping").status_code == 200
        assert client.get("/metrics").status_code == 200
        assert total(client, "model_inference_requests_total") == 0
        assert total(client, "model_inference_duration_seconds_count") == 0
