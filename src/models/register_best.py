"""
register_best.py - Pilih run terbaik (RMSE terendah), daftarkan ke Model Registry,
dan tulis dokumentasi Model Metadata ke docs/model_metadata.md (LK-06 No. 5 & luaran 3).
"""
import argparse
from datetime import datetime

import mlflow
from mlflow import MlflowClient


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracking_uri", default="sqlite:///mlflow.db")
    p.add_argument("--experiment_name", default="PM25-Jakarta-XGBoost")
    p.add_argument("--model_name", default="pm25-xgboost")
    p.add_argument("--output", default="docs/model_metadata.md")
    args = p.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name(args.experiment_name)
    if exp is None:
        raise SystemExit(f"Experiment '{args.experiment_name}' tidak ditemukan. Jalankan train.py dulu.")

    runs = client.search_runs([exp.experiment_id], order_by=["metrics.rmse ASC"], max_results=1)
    if not runs:
        raise SystemExit("Belum ada run.")
    best = runs[0]

    mv = mlflow.register_model(f"runs:/{best.info.run_id}/model", args.model_name)
    client.set_registered_model_alias(args.model_name, "champion", mv.version)

    lines = [
        "# Model Metadata - PM2.5 Jakarta (LK-06)",
        "",
        f"- Tanggal dicatat : {datetime.now():%Y-%m-%d %H:%M}",
        f"- Experiment      : {args.experiment_name}",
        f"- Run terbaik     : `{best.info.run_name}` (`{best.info.run_id}`)",
        f"- Model Registry  : `{args.model_name}` versi {mv.version} (alias `champion`)",
        f"- Versi data (DVC): `{best.data.tags.get('data_version_md5', 'unknown')}`",
        "- Kriteria pemilihan: RMSE test terendah",
        "",
        "## Parameter terbaik",
        "| Parameter | Nilai |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in sorted(best.data.params.items())]
    lines += ["", "## Metrik", "| Metrik | Nilai |", "|---|---|"]
    lines += [f"| {k} | {v:.4f} |" for k, v in sorted(best.data.metrics.items())]

    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[OK] Terdaftar: {args.model_name} v{mv.version}. Metadata: {args.output}")


if __name__ == "__main__":
    main()