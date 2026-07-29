from __future__ import annotations

import os
from typing import Any

import numpy as np

from app.config import settings
from app.embedding.cache import cached_count, cached_embedding
from app.embedding.synthetic import synthetic_embedding
from app.mrl import truncate_and_normalize


def embed_item(dataset_id: str, segment_id: str, source: str | list[str] | None, dim: int = 2048) -> np.ndarray:
    if settings.embedding_mode == "synthetic":
        base = synthetic_embedding(segment_id, settings.embedding_dim)
    elif settings.embedding_mode == "cached":
        base = cached_embedding(dataset_id, segment_id)
    else:
        from app.embedding.qwen import process
        base = process({"video": source})
    return truncate_and_normalize(base, dim)


def embed_query(query: str, dataset_id: str, dim: int = 2048) -> np.ndarray:
    if settings.embedding_mode == "real":
        from app.embedding.qwen import process
        base = process({"text": query})
    else:
        # Cached mode contains item vectors only; real query vectors can be added as
        # {dataset}_queries.npz. Until then, deterministic query embedding is explicit.
        base = synthetic_embedding(f"query:{dataset_id}:{query}", settings.embedding_dim)
    return truncate_and_normalize(base, dim)


def mode_info(dataset_id: str | None = None) -> dict[str, Any]:
    mode = settings.embedding_mode
    result: dict[str, Any] = {"mode": mode}
    if mode == "cached" and dataset_id:
        result["dataset_id"] = dataset_id
        result["count"] = cached_count(dataset_id)
    elif mode == "real":
        result.update({"model": settings.qwen_model, "dtype": settings.torch_dtype, "gpu": os.getenv("GPU_NAME", "runtime-detected")})
    return result
