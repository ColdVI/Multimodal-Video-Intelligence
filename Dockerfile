# syntax=docker/dockerfile:1.7
ARG PYTHON_IMAGE=python:3.11-slim-bookworm
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
FROM ${PYTHON_IMAGE} AS gpu
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
    && python -m pip install \
        torch==2.8.0 \
        torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/cu126 \
    && python -m pip install -r requirements.txt \
    && python -c "import sys, scipy, torch; \
assert sys.version_info[:2] == (3, 11), sys.version; \
assert scipy.__version__ == '1.16.3', scipy.__version__; \
assert torch.__version__.startswith('2.8.0'), torch.__version__; \
assert torch.version.cuda == '12.6', torch.version.cuda; \
print('Python:', sys.version); \
print('SciPy:', scipy.__version__); \
print('Torch:', torch.__version__); \
print('Torch CUDA:', torch.version.cuda)"
COPY service/app ./app
COPY service/ui ./ui
COPY dataset_adapters ./dataset_adapters
COPY scripts/gpu_smoke.py ./scripts/gpu_smoke.py
COPY common.py config.yaml ./
COPY tests/fixtures ./tests/fixtures
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# This target is built only through scripts/export_offline_bundle.py. The
# external named context is hash-verified before BuildKit is invoked, so model
# weights never need to be copied into the repository or its default context.
FROM gpu AS gpu-bundled
ENV MODEL_BUNDLE_ROOT=/opt/mvi-model-bundle \
    QWEN_REPO_PATH=/opt/mvi-model-bundle/source \
    QWEN_MODEL_PATH=/opt/mvi-model-bundle/model \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
COPY --from=mvi_model_bundle / /opt/mvi-model-bundle/

FROM api AS default
