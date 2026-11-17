"""Kontrak yang sama untuk preprocessing, training, pengujian, dan inferensi."""
POLLUTANTS = ["pm2_5", "pm10", "no2", "so2", "co"]
LAGS = range(1, 7)
HORIZON = 3
TARGET_COL = "target_pm2_5_lead_3"
FEATURE_COLUMNS = POLLUTANTS + ["hour", "hour_sin", "hour_cos"] + [
    f"{column}_lag_{lag}" for column in POLLUTANTS for lag in LAGS
]
EXPERIMENT = "PM25-Jakarta-XGBoost-Corrected"
PROTOCOL = "purged_temporal_v1"
