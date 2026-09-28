import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

LATITUDE = -6.2088
LONGITUDE = 106.8456
API_KEY = os.getenv("OWM_API_KEY", "")
API_URL = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LATITUDE}&lon={LONGITUDE}&appid={API_KEY}"

RAW_DATA_DIR = os.path.join("data", "raw")

def fetch_air_quality_api():
    if not API_KEY or API_KEY == "dummy_key":
        print("[WARNING] API Key tidak valid. Menggunakan Fallback...")
        return None
    try:
        response = requests.get(API_URL, timeout=10)
        if response.status_code == 200:
            data = response.json()
            components = data.get("list", [])[0].get("components", {})
            dt = data.get("list", [])[0].get("dt")
            
            record = {
                "timestamp": datetime.fromtimestamp(dt, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "pm2_5": float(components.get("pm2_5", 0.0)),
                "pm10": float(components.get("pm10", 0.0)),
                "no2": float(components.get("no2", 0.0)),
                "so2": float(components.get("so2", 0.0)),
                "co": float(components.get("co", 0.0))
            }
            print("[INFO] Berhasil mengambil data API.")
            return pd.DataFrame([record])
        return None
    except Exception as e:
        print(f"[ERROR] {e}")
        return None

def generate_fallback_data(num_samples=24):
    print("[INFO] Membangkitkan 24 jam data simulasi (Fallback)...")
    now = datetime.now(timezone.utc)
    dates = [now - pd.Timedelta(hours=i) for i in range(num_samples - 1, -1, -1)]
    
    np.random.seed(42)
    return pd.DataFrame({
        "timestamp": [d.strftime("%Y-%m-%d %H:%M:%S") for d in dates],
        "pm2_5": np.round(np.clip(np.random.normal(35.0, 12.0, num_samples), 5.0, 150.0), 2),
        "pm10": np.round(np.clip(np.random.normal(45.0, 15.0, num_samples), 10.0, 200.0), 2),
        "no2": np.round(np.clip(np.random.normal(20.0, 5.0, num_samples), 2.0, 50.0), 2),
        "so2": np.round(np.clip(np.random.normal(10.0, 3.0, num_samples), 1.0, 30.0), 2),
        "co": np.round(np.clip(np.random.normal(400.0, 80.0, num_samples), 100.0, 1000.0), 2)
    })

def run_ingestion():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    df = fetch_air_quality_api()
    if df is None or df.empty:
        df = generate_fallback_data()
    
    # Simpan dengan timestamp (Syarat LK-04)
    time_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    df.to_csv(os.path.join(RAW_DATA_DIR, f"air_quality_raw_{time_str}.csv"), index=False)
    
    # Update file utama
    main_filepath = os.path.join(RAW_DATA_DIR, "air_quality_raw.csv")
    if os.path.exists(main_filepath):
        df_main = pd.read_csv(main_filepath)
        df_combined = pd.concat([df_main, df], ignore_index=True).drop_duplicates(subset=["timestamp"], keep="last")
        df_combined.to_csv(main_filepath, index=False)
    else:
        df.to_csv(main_filepath, index=False)
    print("[SUCCESS] Data Ingestion Selesai.")

if __name__ == "__main__":
    run_ingestion()