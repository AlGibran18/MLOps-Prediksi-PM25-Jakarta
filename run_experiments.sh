#!/usr/bin/env bash
# Menjalankan 4 variasi hyperparameter (minimal 3 sesuai tugas LK-06).
set -e

python src/models/train.py --run_name "run1-baseline"   --n_estimators 50  --learning_rate 0.10 --max_depth 3
python src/models/train.py --run_name "run2-more-trees" --n_estimators 200 --learning_rate 0.05 --max_depth 3
python src/models/train.py --run_name "run3-deeper"     --n_estimators 200 --learning_rate 0.05 --max_depth 5
python src/models/train.py --run_name "run4-regularized" --n_estimators 300 --learning_rate 0.03 --max_depth 4 --subsample 0.8 --colsample_bytree 0.8

echo "Selesai. Jalankan: mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000"