from __future__ import annotations

import statistics
import argparse
import json
import sys
import threading
import time
from typing import Any

import numpy as np

from app.config import settings


_load_lock = threading.Lock()
_embedder: Any | None = None
_load_details: dict[str, Any] = {}


def _cpu_supports_bf16(torch_module: Any) -> bool:
    probe = getattr(getattr(torch_module, "cpu", None), "_is_avx512_bf16_supported", None)
    try:
        return bool(probe()) if callable(probe) else False
    except Exception:
        return False


def get_embedder() -> Any:
    """Qwen text tower'ini tek kez, process icinde lock altinda CPU'ya yukler."""
    global _embedder, _load_details
    if _embedder is not None:
        return _embedder
    with _load_lock:
        if _embedder is not None:
            return _embedder
        started = time.perf_counter()
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("hybrid_text requires torch; no synthetic fallback is allowed") from exc
        repo_path = settings.qwen_repo_path
        if not repo_path.exists():
            raise RuntimeError(
                f"Qwen source is not provisioned at {repo_path}; no synthetic fallback is allowed"
            )
        model_path = settings.qwen_model_path
        if not model_path.exists():
            raise RuntimeError(
                f"Qwen model is not provisioned at {model_path}; no synthetic fallback is allowed"
            )
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

        use_bf16 = _cpu_supports_bf16(torch)
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        _embedder = Qwen3VLEmbedder(
            model_name_or_path=str(model_path),
            fps=1.0,
            max_frames=16,
            max_length=16384,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        )
        _load_details = {
            "model_load_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "dtype": "bfloat16" if use_bf16 else "float32",
            "device": "cpu",
            "model_id": settings.qwen_model_id,
            "model_revision": settings.qwen_model_revision,
        }
        return _embedder


def embed_text(text: str) -> np.ndarray:
    if not isinstance(text, str) or not text:
        raise ValueError("text must be a non-empty string")
    result = get_embedder().process([{"text": text}])
    vector = result.detach().cpu().float().numpy()[0].astype(np.float32, copy=False)
    if vector.shape != (2048,) or not np.isfinite(vector).all():
        raise RuntimeError(f"Qwen returned an invalid text embedding: {vector.shape}")
    return vector


def benchmark(text: str, warm_runs: int | None = None) -> dict[str, Any]:
    """Cold model load ile cold/warm query gecikmesini ayri olcer.

    Dogru cold load kaniti icin restart sonrasi, baska embed cagrisindan once calistirilir.
    Fallback karari yalniz warm p50 uzerinden verilir.
    """
    global _load_details
    load_started = time.perf_counter()
    already_loaded = _embedder is not None
    get_embedder()
    observed_load_ms = (time.perf_counter() - load_started) * 1000.0
    cold_started = time.perf_counter()
    first = embed_text(text)
    cold_query_ms = (time.perf_counter() - cold_started) * 1000.0
    count = warm_runs or settings.qwen_text_warm_runs
    warm_ms: list[float] = []
    for _ in range(count):
        started = time.perf_counter()
        current = embed_text(text)
        warm_ms.append((time.perf_counter() - started) * 1000.0)
        if not np.allclose(first, current, atol=1e-6):
            raise RuntimeError("Qwen warm-query determinism check failed")
    warm_p50_ms = statistics.median(warm_ms)
    limit_ms = settings.qwen_text_warm_limit_s * 1000.0
    return {
        **_load_details,
        "model_load_ms": 0.0 if already_loaded else round(observed_load_ms, 3),
        "cold_query_ms": round(cold_query_ms, 3),
        "warm_query_ms": [round(value, 3) for value in warm_ms],
        "warm_p50_ms": round(warm_p50_ms, 3),
        "warm_limit_ms": round(limit_ms, 3),
        "fallback_recommended": warm_p50_ms > limit_ms,
        "fallback_basis": "warm_p50",
    }


def load_details() -> dict[str, Any]:
    return dict(_load_details)


def runtime_details() -> dict[str, Any]:
    if _load_details:
        return load_details()
    try:
        import torch

        dtype = "bfloat16" if _cpu_supports_bf16(torch) else "float32"
    except ImportError:
        dtype = "unavailable (torch missing)"
    return {
        "dtype": dtype,
        "device": "cpu",
        "model_id": settings.qwen_model_id,
        "model_revision": settings.qwen_model_revision,
    }


def fallback_mode(metrics: dict[str, Any]) -> str:
    """Fallback karari cold sureye degil, yalniz warm p50'ye dayanir."""
    return "cached_only" if float(metrics["warm_p50_ms"]) > float(metrics["warm_limit_ms"]) else "hybrid_text"


__all__ = ["benchmark", "embed_text", "fallback_mode", "get_embedder", "load_details", "runtime_details"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure Qwen CPU cold/warm text latency")
    parser.add_argument("--text", default="dense traffic")
    parser.add_argument(
        "--out", type=str,
        default=str(settings.artifacts_root / "research" / "hybrid_text_benchmark.json"),
    )
    args = parser.parse_args()
    metrics = benchmark(args.text)
    metrics["selected_mode"] = fallback_mode(metrics)
    metrics["synthetic_fallback"] = False
    from pathlib import Path

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
