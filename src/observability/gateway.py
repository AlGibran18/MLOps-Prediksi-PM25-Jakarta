"""Proxy tipis: inferensi tetap dilakukan MLflow; /metrics berada di tiap replika."""
from contextlib import asynccontextmanager
import json
import math
import os
from pathlib import Path
import socket
import threading
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
import httpx
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from prometheus_client.exposition import CONTENT_TYPE_LATEST
import psutil


class ProcessFamilyCollector:
    """CPU kumulatif dan jumlah RSS supervisor, gateway, serta proses MLflow turunannya."""
    def __init__(self, root_pid, replica):
        self.root_pid, self.replica = root_pid, replica
        self.previous = {}
        self.cpu_total = 0.0
        self.lock = threading.Lock()

    def collect(self):
        with self.lock:
            memory, current = 0, {}
            try:
                root = psutil.Process(self.root_pid)
                processes = [root, *root.children(recursive=True)]
            except psutil.Error:
                processes = []
            for process in processes:
                try:
                    key = (process.pid, process.create_time())
                    times = process.cpu_times()
                    used = times.user + times.system
                    self.cpu_total += max(0, used - self.previous.get(key, 0))
                    current[key] = used
                    memory += process.memory_info().rss
                except psutil.Error:
                    continue
            self.previous = current
            cpu = CounterMetricFamily("model_serving_process_cpu_seconds",
                                      "CPU kumulatif keluarga proses serving, dalam detik.", labels=["replica"])
            cpu.add_metric([self.replica], self.cpu_total)
            rss = GaugeMetricFamily("model_serving_process_memory_bytes",
                                    "Jumlah RSS keluarga proses; bukan penggunaan memori cgroup/container.", labels=["replica"])
            rss.add_metric([self.replica], memory)
        yield cpu
        yield rss


def create_app(transport=None, upstream="http://127.0.0.1:8000", reference_path=None):
    replica = socket.gethostname()
    registry = CollectorRegistry()
    registry.register(ProcessFamilyCollector(int(os.getenv("MODEL_SERVING_ROOT_PID", os.getpid())), replica))
    requests = Counter("model_inference_requests_total", "Jumlah request inferensi menurut status HTTP.",
                       ["replica", "status"], registry=registry)
    latency = Histogram("model_inference_duration_seconds", "Latensi HTTP inferensi termasuk proxy dan serialisasi.",
                        ["replica"], buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30), registry=registry)
    predictions = Histogram("model_prediction_pm25_ug_m3", "Distribusi nilai prediksi PM2.5; bukan metrik akurasi.",
                            ["replica"], buckets=(0, 5, 15, 25, 35, 55, 75, 100, 150, 250, 500), registry=registry)
    latency.labels(replica)
    predictions.labels(replica)
    for status in ("200", "400", "422", "502"):
        requests.labels(replica, status)
    path = Path(reference_path or Path(__file__).with_name("model_reference.json"))
    if path.exists():
        reference = json.loads(path.read_text())
        info = Gauge("model_serving_info", "Identitas snapshot yang dibungkus image.",
                     ["replica", "model_name", "model_version", "run_id"], registry=registry)
        info.labels(replica, reference["model_name"], str(reference["model_version"]), reference["run_id"]).set(1)

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(base_url=upstream, transport=transport, trust_env=False,
                                     timeout=httpx.Timeout(60, connect=5)) as client:
            app.state.upstream = client
            yield

    app = FastAPI(title="MLflow Models Observability", lifespan=lifespan)

    @app.get("/ping")
    async def ping():
        try:
            result = await app.state.upstream.get("/ping")
            return Response(content=result.content, status_code=result.status_code)
        except httpx.RequestError:
            return JSONResponse({"error": "MLflow belum siap"}, status_code=503)

    @app.get("/metrics")
    async def metrics():
        return Response(content=generate_latest(registry), headers={"Content-Type": CONTENT_TYPE_LATEST})

    @app.post("/invocations")
    async def invocations(request: Request):
        started = time.perf_counter()
        status = 502
        try:
            result = await app.state.upstream.post("/invocations", content=await request.body(),
                headers={"Content-Type": request.headers.get("content-type", "application/json")})
            status = result.status_code
            if 200 <= status < 300:
                try:
                    for value in result.json()["predictions"]:
                        if math.isfinite(float(value)):
                            predictions.labels(replica).observe(float(value))
                except (ValueError, TypeError, KeyError):
                    pass
            return Response(content=result.content, status_code=status,
                            headers={"Content-Type": result.headers.get("content-type", "application/json")})
        except httpx.RequestError:
            return JSONResponse({"error": "Layanan inferensi MLflow tidak dapat dijangkau"}, status_code=502)
        finally:
            requests.labels(replica, str(status)).inc()
            latency.labels(replica).observe(time.perf_counter() - started)

    return app


app = create_app()
