# syntax=docker/dockerfile:1.7
ARG PYTHON_IMAGE=python:3.11-slim
FROM ${PYTHON_IMAGE} AS cpu-base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install --upgrade pip \
    && python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements.txt
COPY service/app ./app
COPY service/ui ./ui
COPY dataset_adapters ./dataset_adapters
COPY scripts/gpu_smoke.py ./scripts/gpu_smoke.py
COPY common.py config.yaml ./
COPY tests/fixtures ./tests/fixtures
FROM cpu-base AS api
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
FROM cpu-base AS hybrid
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
FROM cpu-base AS ui
EXPOSE 7860
CMD ["python", "-m", "ui.app"]
ARG CUDA_IMAGE_TAG=12.1.1-runtime-ubuntu22.04
FROM nvidia/cuda:${CUDA_IMAGE_TAG} AS gpu
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ffmpeg build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install --upgrade pip \
    && python3 -m pip install torch==2.8.0 torchvision==0.23.0 \
    && python3 -m pip install -r requirements.txt \
    && python3 -m pip uninstall -y torchaudio
COPY service/app ./app
COPY service/ui ./ui
COPY dataset_adapters ./dataset_adapters
COPY scripts/gpu_smoke.py ./scripts/gpu_smoke.py
COPY common.py config.yaml ./
COPY tests/fixtures ./tests/fixtures
EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
FROM api AS default