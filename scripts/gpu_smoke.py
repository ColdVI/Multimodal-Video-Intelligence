from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_ROOT = REPO_ROOT / "service" if (REPO_ROOT / "service" / "app").is_dir() else REPO_ROOT
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.config import settings  # noqa: E402


def _git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=False, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip().splitlines()[0] if result.returncode == 0 and result.stdout.strip() else None


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(dataset: Path, data_root: Path, output: Path, limit: int = 10) -> tuple[dict[str, Any], int]:
    command = (
        "python scripts/gpu_smoke.py --dataset "
        f"{dataset} --data-root {data_root} --output {output} --windows {limit}"
    )
    base: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "model_id": settings.qwen_model_id,
        "model_revision": settings.qwen_model_revision,
        "source_commit": settings.qwen_source_commit,
        "embedding_dimension": 2048,
        "embed_batch_size": settings.embed_batch_size,
    }
    try:
        import torch
    except ImportError:
        payload = {
            **base, "status": "not_run", "result": "not_run", "reason": "torch is not installed",
            "required_command": command,
            "expected_environment": "Linux host with NVIDIA GPU/runtime and verified Qwen bundle",
        }
        _write(output, payload)
        return payload, 4
    if not torch.cuda.is_available():
        payload = {
            **base, "status": "not_run", "result": "not_run", "reason": "CUDA GPU is unavailable",
            "required_command": command,
            "expected_environment": "Linux host with NVIDIA GPU/runtime and verified Qwen bundle",
            "gpu_name": None, "driver_version": _driver_version(),
            "cuda_runtime": getattr(torch.version, "cuda", None), "torch_version": torch.__version__,
            "windows_embedded": 0,
        }
        _write(output, payload)
        return payload, 4

    from app.embedding.qwen import embed_videos
    from app.ingestion.generic_loader import batched, iter_window_records, release_frames

    torch.cuda.reset_peak_memory_stats()
    iterator = iter_window_records(dataset, data_root=data_root)
    embedded = 0
    started = time.perf_counter()
    for batch in batched(iterator, settings.embed_batch_size):
        if embedded >= limit:
            break
        selected = batch[: limit - embedded]
        vectors = embed_videos([record.frames for record in selected])
        if vectors.shape != (len(selected), 2048) or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
            raise RuntimeError(f"invalid Qwen batch result: shape={vectors.shape}; dtype={vectors.dtype}")
        embedded += len(selected)
        release_frames(selected)
    elapsed = time.perf_counter() - started
    if embedded != limit:
        raise RuntimeError(f"dataset yielded {embedded} windows; smoke requires exactly {limit}")
    payload = {
        **base,
        "status": "pass", "result": "pass",
        "gpu_name": torch.cuda.get_device_name(0),
        "driver_version": _driver_version(),
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "torch_version": torch.__version__,
        "windows_embedded": embedded,
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3),
        "elapsed_s": round(elapsed, 3),
        "windows_per_s": round(embedded / elapsed, 3),
    }
    _write(output, payload)
    return payload, 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed ten real video windows with pinned Qwen on CUDA")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=settings.data_root)
    parser.add_argument("--output", type=Path, default=settings.artifacts_root / "faz11" / "gpu_smoke.json")
    parser.add_argument("--windows", type=int, default=10)
    args = parser.parse_args()
    if args.windows < 1:
        parser.error("--windows must be positive")
    payload, code = run(args.dataset, args.data_root, args.output, args.windows)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
