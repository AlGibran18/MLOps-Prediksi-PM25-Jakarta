#!/usr/bin/env bash
# Jalankan dari root repo setelah layanan LK9 healthy dan dataset dipulihkan.
set -euo pipefail
python -m src.models.prepare_serving
mlflow models build-docker \
  --model-uri build/serving/model \
  --name "${MODEL_SERVING_BASE_IMAGE:-pm25-jakarta-model:mlflow-base}" \
  --env-manager local
# Artefak tetap berasal dari image resmi hasil MLflow Models; tambahkan instrumentasi.
# Konteks khusus diperlukan karena build/ diabaikan oleh .dockerignore utama.
instrumentation_context=$(mktemp -d)
trap 'rm -rf "$instrumentation_context"' EXIT
cp docker/Dockerfile.observability "$instrumentation_context/Dockerfile"
cp requirements-observability.txt "$instrumentation_context/"
mkdir -p "$instrumentation_context/src" "$instrumentation_context/build/serving"
cp -R src/observability "$instrumentation_context/src/"
cp build/serving/model_reference.json "$instrumentation_context/build/serving/"
docker build \
  --build-arg "MODEL_BASE_IMAGE=${MODEL_SERVING_BASE_IMAGE:-pm25-jakarta-model:mlflow-base}" \
  --tag "${MODEL_SERVING_IMAGE:-pm25-jakarta-model:production}" \
  "$instrumentation_context"
