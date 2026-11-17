import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

import mlflow
from mlflow import MlflowClient
from src.models.train import get_dvc_md5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracking_uri", default=__import__("os").getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    p.add_argument("--experiment_name", default="PM25-Jakarta-XGBoost-Corrected")
    p.add_argument("--model_name", default="pm25-xgboost")
    p.add_argument("--output", default="docs/model_metadata.md")
    args = p.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name(args.experiment_name)
    if exp is None:
        raise SystemExit(f"Experiment '{args.experiment_name}' tidak ditemukan. Jalankan train.py dulu.")

    runs = client.search_runs([exp.experiment_id], filter_string=f"attributes.status = 'FINISHED' and tags.data_version_md5 = '{get_dvc_md5('data/processed/features.csv.dvc')}' and tags.evaluation_protocol = 'purged_temporal_v1'", order_by=["metrics.val_rmse ASC"], max_results=1)
    if not runs:
        raise SystemExit("Belum ada run.")
    best = runs[0]
    if "val_rmse" not in best.data.metrics:
        raise SystemExit("Run tidak mempunyai validation RMSE")

    mv = mlflow.register_model(f"runs:/{best.info.run_id}/model", args.model_name)
    for key in ['data_version_md5','raw_data_version_md5','evaluation_protocol','data_source']:
        client.set_model_version_tag(args.model_name, mv.version, key, best.data.tags.get(key, 'unknown'))
    client.set_registered_model_alias(args.model_name, "champion", mv.version)

    lines = [
        "# Model Metadata - PM2.5 Jakarta (LK-06 diperbaiki)",
        "",
        f"- Tanggal dicatat : {datetime.now(ZoneInfo('Asia/Jakarta')):%Y-%m-%d %H:%M} WIB",
        f"- Experiment      : {args.experiment_name}",
        f"- Run terbaik     : `{best.info.run_name}` (`{best.info.run_id}`)",
        f"- Model Registry  : `{args.model_name}` versi {mv.version} (alias `champion`)",
        f"- Versi data (DVC): `{best.data.tags.get('data_version_md5', 'unknown')}`",
        f"- Versi raw (DVC): `{best.data.tags.get('raw_data_version_md5', 'unknown')}`",
        f"- Protokol evaluasi: `{best.data.tags.get('evaluation_protocol', 'unknown')}`",
        f"- Sumber dataset: `{best.data.tags.get('data_source', 'unknown')}` (hasil demo tidak membuktikan akurasi data aktual)",
        "- Kriteria pemilihan: validation RMSE terendah; test bukan dasar pemilihan",
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
