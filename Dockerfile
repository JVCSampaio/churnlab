# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    CHURNLAB_ROOT=/app \
    MLFLOW_ALLOW_FILE_STORE=true \
    MLFLOW_TRACKING_URI=file:///app/mlruns

# XGBoost's wheel needs libgomp at runtime on the slim image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY data ./data
RUN python -m churnlab.pipeline --no-track

EXPOSE 8000
CMD ["uvicorn", "churnlab.api:app", "--host", "0.0.0.0", "--port", "8000"]
