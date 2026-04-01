# PuttPro server — supports both GPU (CUDA) and CPU-only builds.
#
# Build for GPU (default):
#   docker build -t puttpro .
#
# Build CPU-only (cheaper cloud instances, no GPU):
#   docker build --build-arg BASE=cpu -t puttpro-cpu .
#
# Run locally with GPU:
#   docker run --gpus all -p 5000:5000 -p 5001:5001 puttpro
#
# Run in cloud (provide real cert via env vars — see docker-compose.yml):
#   docker run --gpus all -p 5000:5000 \
#     -e PUTTPRO_CERT_FILE=/certs/cert.pem \
#     -e PUTTPRO_KEY_FILE=/certs/key.pem \
#     -v /path/to/certs:/certs \
#     puttpro

ARG BASE=gpu

# ── GPU base ─────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS base-gpu
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ── CPU base ─────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS base-cpu
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ── Final stage ──────────────────────────────────────────────────────────────
FROM base-${BASE} AS final
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

# Pre-download YOLOv8n weights so they are baked into the image.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

EXPOSE 5000 5001

CMD ["python", "app.py"]
