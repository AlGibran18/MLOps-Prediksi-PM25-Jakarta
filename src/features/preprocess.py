"""Fitur per jam, hanya memakai informasi saat ini/masa lalu."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.schema import FEATURE_COLUMNS, HORIZON, LAGS, POLLUTANTS, TARGET_COL


def build_features(raw):
    required = {"timestamp", "source", *POLLUTANTS}
    if required - set(raw):
        raise ValueError(f"Kolom raw kurang: {sorted(required - set(raw))}")
    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    if (df["timestamp"] != df["timestamp"].dt.floor("h")).any():
        raise ValueError("Timestamp raw harus tepat pada grid per jam")
    if df["timestamp"].duplicated().any():
        raise ValueError("Timestamp raw duplikat")
    if df["source"].nunique() != 1 or df["source"].isna().any():
        raise ValueError("Satu dataset harus memiliki satu sumber yang diketahui")
    for c in POLLUTANTS:
        df[c] = pd.to_numeric(df[c], errors="raise")
    values = df[POLLUTANTS].to_numpy(dtype=float)
    if np.isinf(values).any() or (values < 0).any():
        raise ValueError("Polutan negatif/infinite tidak diterima")
    source = df["source"].iloc[0]
    df = df.sort_values("timestamp").set_index("timestamp").asfreq("h")
    # Label tetap observasi asli: jangan mengimputasi ground truth.
    target = df["pm2_5"].shift(-HORIZON)
    # Hanya carry-forward maksimum 2 jam; tidak interpolasi dari masa depan.
    df[POLLUTANTS] = df[POLLUTANTS].ffill(limit=2)
    df["source"] = source
    df["timestamp_wib"] = df.index.tz_convert("Asia/Jakarta")
    df["hour"] = df["timestamp_wib"].dt.hour
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    for col in POLLUTANTS:
        for lag in LAGS:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    df[TARGET_COL] = target
    df["target_timestamp"] = df.index + pd.Timedelta(hours=HORIZON)
    return df.dropna(subset=FEATURE_COLUMNS+[TARGET_COL]).reset_index()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/air_quality_raw.csv")
    p.add_argument("--output", default="data/processed/features.csv")
    args = p.parse_args()
    result = build_features(pd.read_csv(args.input))
    if result.empty:
        raise ValueError("Tidak ada window/target lengkap; kumpulkan data lebih panjang")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    print(f"[OK] {len(result)} baris, {len(FEATURE_COLUMNS)} fitur -> {path}")


if __name__ == "__main__":
    main()
