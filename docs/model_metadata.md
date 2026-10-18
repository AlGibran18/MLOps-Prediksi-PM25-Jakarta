# Model Metadata - PM2.5 Jakarta (LK-06)

- Tanggal dicatat : 2026-10-18 12:36
- Experiment      : PM25-Jakarta-XGBoost
- Run terbaik     : `run1-baseline` (`d74c6f3108584eb18af2d5dbace27f48`)
- Model Registry  : `pm25-xgboost` versi 1 (alias `champion`)
- Versi data (DVC): `304ab4abc3549e00937f8db11ebcb030`
- Kriteria pemilihan: RMSE test terendah

## Parameter terbaik
| Parameter | Nilai |
|---|---|
| colsample_bytree | 1.0 |
| learning_rate | 0.1 |
| max_depth | 3 |
| model_type | XGBRegressor |
| n_estimators | 50 |
| n_features | 38 |
| n_test_rows | 3 |
| n_train_rows | 14 |
| random_state | 42 |
| subsample | 1.0 |

## Metrik
| Metrik | Nilai |
|---|---|
| mae | 6.4740 |
| r2 | 0.2357 |
| rmse | 6.7733 |
| train_rmse | 0.6464 |
