# Metadata model aktif — LK-06/LK-07 diperbaiki

- Dicatat: 2026-11-17T15:31:21.884602+07:00
- Experiment: `PM25-Jakarta-XGBoost-Corrected`
- Model registry: `pm25-xgboost`, versi `2`, stage `Production`
- Run: `run2-more-trees` (`ec27aa1d8e074548883cf2b6f2ba7b46`)
- Data fitur MD5: `65c3e5aaa6d16236bd260ce73ee8a298`
- Data raw MD5: `fadb38dde13da2dd352faa8f7733344e`
- Sumber: **simulation**; metrik ini bukan bukti akurasi pengamatan aktual Jakarta.
- Protokol: `purged_temporal_v1`, train/validation/test berurutan dan purge 3 jam pada kedua batas.
- Pemilihan: **validation RMSE** minimum, hanya pada dataset dan protokol yang sama. Test tidak dipakai memilih kandidat.

## Parameter
| Parameter | Nilai |
|---|---|
| colsample_bytree | 1.0 |
| learning_rate | 0.05 |
| max_depth | 3 |
| model_type | XGBRegressor |
| n_estimators | 200 |
| n_features | 38 |
| n_test_rows | 65 |
| n_train_rows | 191 |
| n_validation_rows | 65 |
| purge_hours | 3 |
| random_state | 42 |
| selection_metric | val_rmse |
| subsample | 1.0 |
| test_size | 0.2 |
| test_start | 2026-09-12 04:00:00+00:00 |
| train_label_end | 2026-09-09 07:00:00+00:00 |
| validation_label_end | 2026-09-12 03:00:00+00:00 |
| validation_size | 0.2 |
| validation_start | 2026-09-09 08:00:00+00:00 |

## Metrik
| Metrik | Nilai |
|---|---|
| baseline_mae | 4.560769 |
| baseline_r2 | 0.232637 |
| baseline_rmse | 5.509189 |
| mae | 1.916394 |
| r2 | 0.863108 |
| rmse | 2.326894 |
| train_rmse | 0.477364 |
| val_mae | 1.927669 |
| val_r2 | 0.859862 |
| val_rmse | 2.442974 |

Catatan hasil historis yang tidak konsisten disimpan di [legacy](legacy/README.md).
