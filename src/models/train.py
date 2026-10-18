"""
train.py - Pelatihan model prediksi PM2.5 (+3 jam) dengan XGBoost + MLflow (LK-06).

Data   : data/processed/features.csv (versi data dilacak DVC pada LK-05)
Target : target_pm2_5_lead_3
Contoh :
    python src/models/train.py --n_estimators 200 --learning_rate 0.05 --max_depth 4
"""
import argparse
import os
import re

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

TARGET_COL = "target_pm2_5_lead_3"
TIME_COL_CANDIDATES = ["datetime", "timestamp", "date", "time", "dt", "datetime_wib"]


def parse_args():
    p = argparse.ArgumentParser(description="Train XGBoost PM2.5 + MLflow tracking")
    p.add_argument("--data_path", default="data/processed/features.csv")
    p.add_argument("--tracking_uri", default="sqlite:///mlflow.db")
    p.add_argument("--experiment_name", default="PM25-Jakarta-XGBoost")
    p.add_argument("--run_name", default=None)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    # Hyperparameter XGBoost
    p.add_argument("--n_estimators", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=0.1)
    p.add_argument("--max_depth", type=int, default=3)
    p.add_argument("--subsample", type=float, default=1.0)
    p.add_argument("--colsample_bytree", type=float, default=1.0)
    return p.parse_args()


def load_data(path):
    """Baca fitur, urutkan berdasarkan waktu (jika ada), pisahkan X dan y."""
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Kolom target '{TARGET_COL}' tidak ada. Kolom: {list(df.columns)}")

    time_col = next((c for c in TIME_COL_CANDIDATES if c in df.columns), None)
    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.sort_values(time_col).reset_index(drop=True)

    df = df.dropna(subset=[TARGET_COL])
    X = df.drop(columns=[TARGET_COL]).select_dtypes(include=[np.number])
    X = X.fillna(X.median(numeric_only=True))
    y = df[TARGET_COL]
    return X, y


def chronological_split(X, y, test_size):
    """Split berurutan waktu (tanpa shuffle) agar tidak ada kebocoran data."""
    n_test = max(2, int(len(X) * test_size))
    n_train = len(X) - n_test
    if n_train < 2:
        raise ValueError(f"Data terlalu sedikit ({len(X)} baris) untuk dilatih.")
    return X.iloc[:n_train], X.iloc[n_train:], y.iloc[:n_train], y.iloc[n_train:]


def get_dvc_md5(dvc_file):
    """Ambil hash versi data dari file .dvc agar run MLflow terhubung ke versi data LK-05."""
    try:
        with open(dvc_file) as f:
            m = re.search(r"md5:\s*([0-9a-f]+)", f.read())
        return m.group(1) if m else "unknown"
    except OSError:
        return "unknown"


def main():
    args = parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    X, y = load_data(args.data_path)
    X_train, X_test, y_train, y_test = chronological_split(X, y, args.test_size)

    params = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "random_state": args.random_state,
    }

    with mlflow.start_run(run_name=args.run_name):
        # --- Log parameter ---
        mlflow.log_params(params)
        mlflow.log_param("model_type", "XGBRegressor")
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_train_rows", len(X_train))
        mlflow.log_param("n_test_rows", len(X_test))
        mlflow.set_tag("data_version_md5", get_dvc_md5(args.data_path + ".dvc"))
        mlflow.set_tag("task", "PM2.5 forecasting +3h")

        # --- Training ---
        model = XGBRegressor(objective="reg:squarederror", **params)
        model.fit(X_train, y_train)

        # --- Evaluasi & log metrik ---
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
        mae = float(mean_absolute_error(y_test, pred_test))
        r2 = float(r2_score(y_test, pred_test)) if len(y_test) > 1 else float("nan")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("train_rmse", float(np.sqrt(mean_squared_error(y_train, pred_train))))

        # --- Log model (artefak) ---
        signature = infer_signature(X_train, pred_train)
        mlflow.xgboost.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
        )

        print(f"[OK] run_id={mlflow.active_run().info.run_id} | "
              f"RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f} | params={params}")


if __name__ == "__main__":
    main()