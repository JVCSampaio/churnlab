# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MLFLOW_TRACKING_URI=file:///app/mlruns

# Install dependencies first (better layer caching).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source + data.
COPY src/ ./src/
COPY data/ ./data/

# Train the model at build time so the image ships ready to serve.
RUN python -m churnlab.pipeline

# Expose the service.
EXPOSE 8000
CMD ["uvicorn", "churnlab.api:app", "--host", "0.0.0.0", "--port", "8000"]
