"""Ingestion data aktual; simulasi hanya lewat --simulate dan diberi provenance."""
import argparse
from pathlib import Path
import os

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import requests

from src.features.schema import POLLUTANTS

LATITUDE, LONGITUDE = -6.2088, 106.8456
RAW_FILE = Path("data/raw/air_quality_raw.csv")


def fetch_air_quality_api(api_key):
    if not api_key or api_key == "dummy_key":
        raise ValueError("OWM_API_KEY belum diisi. Gunakan --simulate hanya untuk demonstrasi.")
    try:
        response = requests.get("https://api.openweathermap.org/data/2.5/air_pollution",
                                params={"lat": LATITUDE, "lon": LONGITUDE, "appid": api_key}, timeout=20)
    except requests.RequestException:
        raise RuntimeError("Koneksi OpenWeatherMap gagal; data lama dipertahankan.") from None
    # Jangan cetak URL respons: URL memuat API key.
    if response.status_code != 200:
        raise RuntimeError(f"OpenWeatherMap gagal: HTTP {response.status_code}; data lama dipertahankan.")
    records = response.json().get("list", [])
    if not records or any(c not in records[0].get("components", {}) for c in POLLUTANTS):
        raise ValueError("Respons API tidak memuat seluruh polutan wajib.")
    record = records[0]
    return pd.DataFrame([{"timestamp": pd.to_datetime(record["dt"], unit="s", utc=True),
                          "observed_at": pd.to_datetime(record["dt"], unit="s", utc=True),
                          **{c: float(record["components"][c]) for c in POLLUTANTS},
                          "source": "openweathermap"}])


def generate_simulation(start, hours=168, seed=42):
    """Deret deterministik untuk demonstrasi, bukan observasi kualitas udara Jakarta."""
    if hours < 1:
        raise ValueError("hours harus positif")
    start = pd.Timestamp(start)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    dates = pd.date_range(start.floor("h"), periods=hours, freq="h")
    hour = dates.tz_convert("Asia/Jakarta").hour.to_numpy()
    day = dates.asi8 / (24 * 3600 * 1e9)
    rng = np.random.default_rng(seed)
    pm25 = np.clip(32 + 9 * np.sin(2*np.pi*(hour-6)/24) + 4*np.cos(2*np.pi*day/7)
                   + rng.normal(0, 2, hours), 1, None)
    return pd.DataFrame({"timestamp": dates, "pm2_5": np.round(pm25, 2),
        "pm10": np.round(np.clip(pm25*1.3 + rng.normal(0, 2, hours), 1, None), 2),
        "no2": np.round(np.clip(12+pm25*.25+rng.normal(0, 1, hours), .1, None), 2),
        "so2": np.round(np.clip(4+pm25*.08+rng.normal(0, .4, hours), .1, None), 2),
        "co": np.round(np.clip(260+pm25*4+rng.normal(0, 10, hours), 1, None), 2),
        "source": "simulation"})


def append_records(new, path=RAW_FILE):
    path = Path(path)
    new = new.copy()
    new["timestamp"] = pd.to_datetime(new["timestamp"], utc=True).dt.ceil("h")
    values = new[POLLUTANTS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Nilai polutan harus finite dan tidak negatif")
    if path.exists():
        old = pd.read_csv(path)
        if "source" not in old or set(old["source"].unique()) != set(new["source"].unique()):
            raise ValueError("Sumber lama/baru berbeda atau tidak diketahui; gunakan file terpisah.")
        old["timestamp"] = pd.to_datetime(old["timestamp"], utc=True).dt.ceil("h")
        new = pd.concat([old, new], ignore_index=True)
    result = new.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    result.to_csv(temporary, index=False)
    temporary.replace(path)
    print(f"[OK] {len(result)} baris -> {path}; source={','.join(result['source'].unique())}")
    return result


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=str(RAW_FILE))
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--start", default="2026-09-01T00:00:00Z")
    p.add_argument("--hours", type=int, default=168)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.simulate:
        data = generate_simulation(args.start, args.hours, args.seed)
    else:
        data = fetch_air_quality_api(os.getenv("OWM_API_KEY", ""))
    append_records(data, args.output)


if __name__ == "__main__":
    main()
