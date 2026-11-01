import argparse
import warnings
from datetime import datetime

import mlflow
import yaml
from mlflow import MlflowClient

warnings.filterwarnings("ignore", category=FutureWarning)

STAGE_ALIAS = {"Staging": "staging", "Production": "production"}


def get_versions(client, name):
    vs = client.search_model_versions(f"name='{name}'")
    return sorted(vs, key=lambda v: int(v.version))


def aliases_of(client, name, version):
    """Alias tidak selalu terisi pada hasil search; ambil dari registered model."""
    rm = client.get_registered_model(name)
    return sorted(a for a, v in rm.aliases.items() if str(v) == str(version))


def run_info(client, mv):
    run = client.get_run(mv.run_id)
    return run


def cmd_list(client, args):
    print(f"Model: {args.model_name}")
    print(f"{'Ver':<4}{'Stage':<12}{'Alias':<24}{'Run':<22}{'RMSE':>8}{'MAE':>8}{'R2':>8}")
    for mv in get_versions(client, args.model_name):
        run = run_info(client, mv)
        m = run.data.metrics
        al = aliases_of(client, args.model_name, mv.version)
        aliases = ",".join(al) if al else "-"
        print(f"{mv.version:<4}{mv.current_stage:<12}{aliases:<24}{run.info.run_name:<22}"
              f"{m.get('rmse', float('nan')):>8.4f}{m.get('mae', float('nan')):>8.4f}"
              f"{m.get('r2', float('nan')):>8.4f}")


def cmd_register(client, args):
    exp = client.get_experiment_by_name(args.experiment_name)
    if exp is None:
        raise SystemExit(f"Experiment '{args.experiment_name}' tidak ditemukan.")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"attributes.run_name = '{args.run_name}'",
        order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        raise SystemExit(f"Run '{args.run_name}' tidak ditemukan.")
    run = runs[0]
    mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", args.model_name)
    client.set_model_version_tag(args.model_name, mv.version, "source_run_name", run.info.run_name)
    client.set_model_version_tag(args.model_name, mv.version, "data_version_md5",
                                 run.data.tags.get("data_version_md5", "unknown"))
    client.update_model_version(args.model_name, mv.version,
                                description=args.description or f"Dari run {run.info.run_name}")
    print(f"[OK] {args.model_name} versi {mv.version} terdaftar dari run '{run.info.run_name}'.")


def cmd_promote(client, args):
    versions = get_versions(client, args.model_name)
    if args.best:
        def rmse(mv):
            return run_info(client, mv).data.metrics.get("rmse", float("inf"))
        target = min(versions, key=rmse)
        version = target.version
        print(f"[INFO] Versi dengan RMSE terendah: v{version}")
    else:
        version = str(args.version)

    stage = args.stage
    # Hanya satu versi aktif per stage Production/Staging: versi lama diarsipkan.
    client.transition_model_version_stage(
        args.model_name, version, stage,
        archive_existing_versions=(stage in ("Staging", "Production")))
    if stage in STAGE_ALIAS:
        client.set_registered_model_alias(args.model_name, STAGE_ALIAS[stage], version)
    if stage == "Production":
        client.set_registered_model_alias(args.model_name, "champion", version)
        rm = client.get_registered_model(args.model_name)
        if str(rm.aliases.get("staging")) == str(version):
            client.delete_registered_model_alias(args.model_name, "staging")
    print(f"[OK] {args.model_name} v{version} -> {stage}")


def cmd_export(client, args):
    prod = [v for v in get_versions(client, args.model_name) if v.current_stage == "Production"]
    if not prod:
        raise SystemExit("Belum ada versi berstatus Production.")
    mv = prod[-1]
    run = run_info(client, mv)
    meta = {
        "registered_model_name": args.model_name,
        "version": int(mv.version),
        "stage": mv.current_stage,
        "aliases": aliases_of(client, args.model_name, mv.version),
        "model_uri_stage": f"models:/{args.model_name}/Production",
        "model_uri_alias": f"models:/{args.model_name}@production",
        "run_id": mv.run_id,
        "run_name": run.info.run_name,
        "data_version_md5": run.data.tags.get("data_version_md5", "unknown"),
        "params": dict(run.data.params),
        "metrics": {k: round(v, 4) for k, v in run.data.metrics.items()},
        "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(args.output, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)
    print(f"[OK] Metadata v{mv.version} (Production) ditulis ke {args.output}")


def main():
    p = argparse.ArgumentParser(description="Operasi MLflow Model Registry")
    p.add_argument("--tracking_uri", default="sqlite:///mlflow.db")
    p.add_argument("--experiment_name", default="PM25-Jakarta-XGBoost")
    p.add_argument("--model_name", default="pm25-xgboost")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")
    r = sub.add_parser("register")
    r.add_argument("--run_name", required=True)
    r.add_argument("--description", default=None)
    pr = sub.add_parser("promote")
    g = pr.add_mutually_exclusive_group(required=True)
    g.add_argument("--version", type=int)
    g.add_argument("--best", action="store_true", help="pilih versi RMSE terendah")
    pr.add_argument("--stage", required=True, choices=["None", "Staging", "Production", "Archived"])
    e = sub.add_parser("export")
    e.add_argument("--output", default="models/registry_reference.yaml")

    args = p.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    {"list": cmd_list, "register": cmd_register, "promote": cmd_promote, "export": cmd_export}[args.cmd](client, args)


if __name__ == "__main__":
    main()
