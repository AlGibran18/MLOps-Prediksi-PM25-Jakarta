"""XGBoost + MLflow dengan lineage checksum dan evaluasi temporal terpisah."""
import argparse
import hashlib
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.features.schema import EXPERIMENT, FEATURE_COLUMNS, HORIZON, PROTOCOL, TARGET_COL


def get_dvc_md5(dvc_file):
    try:
        outs = yaml.safe_load(Path(dvc_file).read_text())["outs"]
        return outs[0]["md5"]
    except (OSError, KeyError, TypeError):
        return "unknown"


def verify_data_version(path):
    path = Path(path)
    expected = get_dvc_md5(str(path)+".dvc")
    actual = hashlib.md5(path.read_bytes()).hexdigest()
    if expected != actual:
        raise ValueError(f"Lineage tidak cocok: {path}; pointer={expected}, aktual={actual}. Jalankan dvc add.")
    return actual


def load_frame(path):
    df = pd.read_csv(path)
    required = {"timestamp", "target_timestamp", "source", TARGET_COL, *FEATURE_COLUMNS}
    if required - set(df):
        raise ValueError(f"Kolom fitur kurang: {sorted(required - set(df))}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    df["target_timestamp"] = pd.to_datetime(df["target_timestamp"], utc=True, errors="raise")
    df = df.sort_values("timestamp").reset_index(drop=True)
    if df["timestamp"].duplicated().any():
        raise ValueError("Timestamp fitur duplikat")
    if not (df["target_timestamp"] == df["timestamp"]+pd.Timedelta(hours=HORIZON)).all():
        raise ValueError("Waktu label harus tepat +3 jam")
    if not (df["timestamp"].diff().dropna() >= pd.Timedelta(hours=1)).all():
        raise ValueError("Timestamp fitur minimal berjarak 1 jam")
    if df["source"].nunique() != 1 or df["source"].isna().any():
        raise ValueError("Provenance sumber tidak konsisten")
    values = df[FEATURE_COLUMNS+[TARGET_COL]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Fitur/label tidak finite; perbaiki preprocessing, bukan imputasi seluruh dataset")
    return df


def load_data(path):
    df = load_frame(path)
    return df[FEATURE_COLUMNS].copy(), df[TARGET_COL].copy()


def chronological_split(X, y, test_size=0.2, gap=HORIZON):
    if not 0 < test_size < 1 or gap < HORIZON:
        raise ValueError("test_size harus (0,1), gap minimal horizon 3 jam")
    n_test = max(2, int(len(X)*test_size))
    end_train = len(X)-n_test-gap
    if end_train < 2:
        raise ValueError("Data terlalu sedikit untuk split temporal dengan purge")
    return X.iloc[:end_train], X.iloc[-n_test:], y.iloc[:end_train], y.iloc[-n_test:]


def temporal_indices(n, test_size=.2, validation_size=.2, gap=HORIZON):
    if not 0 < test_size < 1 or not 0 < validation_size < 1 or gap < HORIZON:
        raise ValueError("Fraksi split tidak valid atau gap kurang dari horizon")
    n_test, n_val = max(2, int(n*test_size)), max(2, int(n*validation_size))
    test_start = n-n_test
    val_end, val_start = test_start-gap, test_start-gap-n_val
    train_end = val_start-gap
    if train_end < 2:
        raise ValueError("Data terlalu sedikit untuk train/validation/test dengan dua purge")
    return slice(0, train_end), slice(val_start, val_end), slice(test_start, n)


def metrics(y, prediction):
    return {"rmse": float(np.sqrt(mean_squared_error(y, prediction))),
            "mae": float(mean_absolute_error(y, prediction)),
            "r2": float(r2_score(y, prediction))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="data/processed/features.csv")
    p.add_argument("--raw_path", default="data/raw/air_quality_raw.csv")
    p.add_argument("--tracking_uri", default=None)
    p.add_argument("--experiment_name", default=EXPERIMENT)
    p.add_argument("--run_name", default=None)
    p.add_argument("--test_size", type=float, default=.2)
    p.add_argument("--validation_size", type=float, default=.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--n_estimators", type=int, default=150)
    p.add_argument("--learning_rate", type=float, default=.1)
    p.add_argument("--max_depth", type=int, default=3)
    p.add_argument("--subsample", type=float, default=1.)
    p.add_argument("--colsample_bytree", type=float, default=1.)
    args = p.parse_args()
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    elif not __import__('os').getenv('MLFLOW_TRACKING_URI'):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    data_md5 = verify_data_version(args.data_path)
    raw_md5 = verify_data_version(args.raw_path)
    df = load_frame(args.data_path)
    X, y = df[FEATURE_COLUMNS], df[TARGET_COL]
    train, val, test = temporal_indices(len(df), args.test_size, args.validation_size)
    # Buktikan label train telah tersedia sebelum origin validation/test.
    assert df.iloc[train]["target_timestamp"].max() < df.iloc[val]["timestamp"].min()
    assert df.iloc[val]["target_timestamp"].max() < df.iloc[test]["timestamp"].min()
    params = {name: getattr(args, name) for name in ["n_estimators", "learning_rate", "max_depth",
              "subsample", "colsample_bytree", "random_state"]}
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params({**params, "model_type": "XGBRegressor", "n_features": X.shape[1],
            "n_train_rows": len(X.iloc[train]), "n_validation_rows": len(X.iloc[val]),
            "n_test_rows": len(X.iloc[test]), "purge_hours": HORIZON,
            "selection_metric": "val_rmse", "test_size": args.test_size,
            "validation_size": args.validation_size})
        mlflow.set_tags({"data_version_md5": data_md5, "raw_data_version_md5": raw_md5,
                         "evaluation_protocol": PROTOCOL, "data_source": df['source'].iloc[0],
                         "task": "PM2.5 forecasting +3h"})
        bounds = {"train_label_end": str(df.iloc[train]['target_timestamp'].max()),
                  "validation_start": str(df.iloc[val]['timestamp'].min()),
                  "validation_label_end": str(df.iloc[val]['target_timestamp'].max()),
                  "test_start": str(df.iloc[test]['timestamp'].min())}
        mlflow.log_params(bounds)
        model = XGBRegressor(objective="reg:squarederror", n_jobs=2, **params)
        model.fit(X.iloc[train], y.iloc[train])
        validation = metrics(y.iloc[val], model.predict(X.iloc[val]))
        testing = metrics(y.iloc[test], model.predict(X.iloc[test]))
        mlflow.log_metrics({**testing, **{f"val_{k}": v for k,v in validation.items()},
                           "train_rmse": metrics(y.iloc[train], model.predict(X.iloc[train]))['rmse']})
        # Persistence baseline: ramalan +3h sama dengan PM2.5 saat ini.
        mlflow.log_metrics({f"baseline_{k}": v for k,v in metrics(y.iloc[test], X.iloc[test]['pm2_5']).items()})
        mlflow.log_dict({"data_md5":data_md5,"raw_md5":raw_md5,"source":df['source'].iloc[0],
                         "protocol":PROTOCOL,**bounds}, "data_lineage.json")
        mlflow.xgboost.log_model(model, name="model", signature=infer_signature(X.iloc[train], model.predict(X.iloc[train])),
            input_example=X.iloc[train].head(3),
            pip_requirements=["mlflow-skinny==3.17.1", "xgboost-cpu==3.4.1", "numpy==2.5.5", "pandas==3.0.6", "scikit-learn==1.9.1"])
        print(f"[OK] run_id={run.info.run_id}; val_rmse={validation['rmse']:.4f}; "
              f"test_rmse={testing['rmse']:.4f}; test_mae={testing['mae']:.4f}; data={data_md5}")


if __name__ == "__main__":
    main()
