FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

FROM base AS mlflow-server
COPY requirements-mlflow.txt ./
# Pasang komponen server; dependensi training hanya diperlukan pada image API.
RUN pip install --no-compile --no-cache-dir -r requirements-mlflow.txt \
    && pip install --no-compile --no-cache-dir --no-deps mlflow==3.17.1
EXPOSE 5000

FROM base AS api-service
COPY . .
RUN pip install --no-compile --no-cache-dir -r requirements-serving.txt \
    && python docker/restore_features.py
EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
