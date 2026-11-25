"""Supervisor server MLflow internal dan gateway metrik dalam satu replika."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading


def main():
    stopped = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.set())
    env = dict(os.environ, MODEL_SERVING_ROOT_PID=str(os.getpid()))
    backend = subprocess.Popen([
        sys.executable, "-m", "mlflow", "models", "serve", "--model-uri", "/opt/ml/model",
        "--env-manager", "local", "--host", "127.0.0.1", "--port", "8000", "--workers", "1"], env=env)
    gateway = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "gateway:app", "--app-dir", str(Path(__file__).parent),
        "--host", "0.0.0.0", "--port", "8080", "--workers", "1"], env=env)
    children = [backend, gateway]
    exit_code = 0
    try:
        while not stopped.wait(.2):
            if any(child.poll() is not None for child in children):
                exit_code = 1
                break
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
