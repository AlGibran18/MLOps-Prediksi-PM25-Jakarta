#!/usr/bin/env bash
# Jalankan dari root repo setelah layanan LK9 healthy dan dataset dipulihkan.
set -euo pipefail
python -m src.models.prepare_serving
mlflow models build-docker \
  --model-uri build/serving/model \
  --name "${MODEL_SERVING_IMAGE:-pm25-jakarta-model:production}" \
  --env-manager local
