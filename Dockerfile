# Inference image: ONNX Runtime + LightGBM + numpy. No PyTorch (that stays in training).
# Build after `inbox train`, which writes artifacts/models.
FROM python:3.13-slim AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
COPY --from=build /install /usr/local
WORKDIR /app
COPY artifacts/models ./models
ENV INBOX_MODELS=/app/models PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["inbox", "serve", "--host", "0.0.0.0", "--port", "8000"]
