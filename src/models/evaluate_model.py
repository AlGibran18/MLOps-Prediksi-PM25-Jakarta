"""
evaluate_model.py - Gerbang validasi model (LK-08, Tahap 3).

Membandingkan metrik run terbaru di MLflow dengan ambang batas pada
configs/thresholds.yaml. Keluar dengan kode 1 jika ada metrik yang melanggar,
sehingga job berikutnya (pendaftaran ke Registry) tidak berjalan.
"""
import argparse
import os
import sys

import mlflow
import yaml
from mlflow import MlflowClient

LIMITS = [("rmse", "rmse_max", "max"), ("mae", "mae_max", "max"), ("r2", "r2_min", "min")]


def check_thresholds(metrics, thresholds):
    """Kembalikan daftar pelanggaran (list kosong = lolos validasi)."""
    violations = []
    for metric, key, kind in LIMITS:
        limit = thresholds.get(key)
        if limit is None:
            continue
        value = metrics.get(metric)
        if value is None or value != value:  # hilang atau NaN
            violations.append(f"{metric}: nilai tidak tersedia (hilang/NaN)")
        elif kind == "max" and value > limit:
            violations.append(f"{metric}={value:.4f} melebihi batas maksimum {limit}")
        elif kind == "min" and value < limit:
            violations.append(f"{metric}={value:.4f} di bawah batas minimum {limit}")
    return violations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracking_uri", default="sqlite:///mlflow.db")
    p.add_argument("--experiment_name", default="PM25-Jakarta-XGBoost")
    p.add_argument("--run_name", default=None, help="kosong = run terbaru")
    p.add_argument("--thresholds", default="configs/thresholds.yaml")
    args = p.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name(args.experiment_name)
    if exp is None:
        sys.exit(f"[GAGAL] Experiment '{args.experiment_name}' tidak ditemukan.")

    filt = f"attributes.run_name = '{args.run_name}'" if args.run_name else ""
    runs = client.search_runs([exp.experiment_id], filter_string=filt,
                              order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        sys.exit("[GAGAL] Tidak ada run yang dievaluasi.")
    run = runs[0]

    with open(args.thresholds) as f:
        thresholds = yaml.safe_load(f) or {}
    metrics = run.data.metrics
    violations = check_thresholds(metrics, thresholds)

    lines = [f"### Evaluasi model: `{run.info.run_name}`", "",
             "| Metrik | Nilai | Ambang |", "|---|---|---|"]
    for metric, key, kind in LIMITS:
        lim = thresholds.get(key)
        sign = "<=" if kind == "max" else ">="
        lines.append(f"| {metric} | {metrics.get(metric, float('nan')):.4f} | "
                     f"{sign} {lim if lim is not None else '-'} |")
    lines += ["", "**Hasil: LOLOS**" if not violations else "**Hasil: GAGAL**"]
    lines += [f"- {v}" for v in violations]
    report = "\n".join(lines)
    print(report)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(report + "\n")

    if violations:
        sys.exit(1)


if __name__ == "__main__":
    main()
