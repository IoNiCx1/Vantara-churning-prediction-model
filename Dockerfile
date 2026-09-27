FROM python:3.11-slim

WORKDIR /app

# System deps needed by lightgbm/xgboost at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install the CPU-only torch build first, in its own layer: the default
# PyPI torch wheel bundles full CUDA support (~800MB) which is both
# unnecessary (this API never uses a GPU) and much more likely to hit a
# network timeout on a slow connection. The CPU wheel is ~200MB.
# Installing it separately also means a later requirements.txt change
# doesn't force torch to re-download, since Docker caches this layer.
RUN pip install --no-cache-dir --timeout 600 --retries 10 \
    torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --timeout 600 --retries 10 -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]