"""Kirim variasi fitur demo ke setiap replika untuk mengaktifkan grafik monitoring."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request


def discover_replicas(expected):
    ids = subprocess.check_output(["docker", "compose", "--profile", "monitoring", "ps", "-q", "model-serving"], text=True).split()
    if len(ids) != expected:
        raise RuntimeError(f"Diharapkan {expected} replika, ditemukan {len(ids)}")
    containers = json.loads(subprocess.check_output(["docker", "inspect", *ids], text=True))
    replicas = []
    for container in containers:
        if container["State"].get("Health", {}).get("Status") != "healthy":
            raise RuntimeError(f"Replika belum healthy: {container['Name']}")
        port = container["NetworkSettings"]["Ports"]["8080/tcp"][0]["HostPort"]
        replicas.append((container["Name"], "http://127.0.0.1:" + port))
    return sorted(replicas)


def send(url, payload):
    request = urllib.request.Request(url + "/invocations", json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=10) as response:
            result = json.load(response)
            return response.status == 200 and bool(result.get("predictions"))
    except (OSError, ValueError, urllib.error.URLError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=180, help="Durasi pengiriman dalam detik")
    parser.add_argument("--rps", type=float, default=6, help="Target total request/detik, bukan per replika")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--replicas", type=int, default=3)
    args = parser.parse_args()
    if min(args.duration, args.rps, args.workers, args.replicas) <= 0:
        parser.error("Semua parameter harus lebih dari nol")
    replicas = discover_replicas(args.replicas)
    columns = json.loads(Path("build/serving/request.json").read_text())["dataframe_split"]["columns"]
    with Path("data/processed/features.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("Dataset fitur kosong")
    counts = {name: {"success": 0, "failed": 0} for name, _ in replicas}
    pending, sent = {}, 0
    started = time.monotonic()

    def collect(finished):
        for future in finished:
            name = pending.pop(future)
            counts[name]["success" if future.result() else "failed"] += 1

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while time.monotonic() - started < args.duration:
            if len(pending) >= args.workers * 2:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                collect(finished)
                if time.monotonic() - started >= args.duration:
                    break
            name, url = replicas[sent % len(replicas)]
            row = rows[sent % len(rows)]
            values = [int(float(row[column])) if column == "hour" else float(row[column]) for column in columns]
            payload = {"dataframe_split": {"columns": columns, "data": [values]}}
            pending[pool.submit(send, url, payload)] = name
            sent += 1
            time.sleep(max(0, started + sent / args.rps - time.monotonic()))
        collect(wait(pending)[0])
    print(json.dumps({"elapsed_seconds": round(time.monotonic() - started, 2),
                      "routing": "round-robin eksplisit ke port tiap replika", "replicas": counts}, indent=2))
    if any(value["failed"] or not value["success"] for value in counts.values()):
        raise SystemExit("Ada replika gagal atau belum menerima request")


if __name__ == "__main__":
    main()
