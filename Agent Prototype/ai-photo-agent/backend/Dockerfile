# ─── Stage 1: Base image with CUDA support ────────────────────────────────
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    curl git && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.11 /usr/bin/python
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models
RUN python -c "import clip; clip.load('ViT-B/32', device='cpu')" || true
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true

COPY . .

# ─── Stage 2: API Server ──────────────────────────────────────────────────
FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--loop", "uvloop"]

# ─── Stage 3: Celery Worker ───────────────────────────────────────────────
FROM base AS worker
CMD ["celery", "-A", "app.worker.celery_app", "worker", "--loglevel=info", "--concurrency=4", "-Q", "photo_processing,default"]

# ─── Stage 4: Celery Beat ─────────────────────────────────────────────────
FROM base AS beat
CMD ["celery", "-A", "app.worker.celery_app", "beat", "--loglevel=info"]
