import os
import pandas as pd
import numpy as np

RAW_FILE = os.path.join("data", "raw", "air_quality_raw.csv")
PROCESSED_FILE = os.path.join("data", "processed", "features.csv")

def preprocess():
    df = pd.read_csv(RAW_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    numeric_cols = ['pm2_5', 'pm10', 'no2', 'so2', 'co']
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
    
    # Cyclical Encoding
    df['hour'] = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    
    # Lag 1-6 dan Target Lead 3
    for col in numeric_cols:
        for lag in range(1, 7):
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    df['target_pm2_5_lead_3'] = df['pm2_5'].shift(-3)
    
    df_clean = df.dropna().reset_index(drop=True)
    os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
    df_clean.to_csv(PROCESSED_FILE, index=False)
    print(f"[SUCCESS] Fitur diekstrak ke {PROCESSED_FILE}. Total baris: {len(df_clean)}")

if __name__ == "__main__":
    preprocess()