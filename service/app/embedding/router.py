from __future__ import annotations

import numpy as np

from app.config import settings
from app.embedding.cache import cache
from app.embedding.synthetic import synthetic_embedding
from app.mrl import truncate_and_normalize


def embed_query(text: str, dimension: int) -> np.ndarray:
    if settings.embedding_mode == "synthetic":
        base = synthetic_embedding(f"query:{text}")
    elif settings.embedding_mode == "cached":
        base = cache.query(text, settings.qwen_model_revision)
    elif settings.embedding_mode == "hybrid_text":
        base = cache.query_or_none(text, settings.qwen_model_revision)
        if base is None:
            from app.embedding.text_cpu import embed_text

            base = embed_text(text)
            cache.put_query(text, settings.qwen_model_revision, base)
    elif settings.embedding_mode == "real":
        from app.embedding.qwen import embed_text

        base = embed_text(text)
    else:  # settings.validate() normalde buraya izin vermez.
        raise RuntimeError(f"unsupported embedding mode: {settings.embedding_mode}")
    return truncate_and_normalize(base, dimension)


def embed_item(key: str, dimension: int, *, dataset_id: str, media: str | list[str] | None = None) -> np.ndarray:
    if settings.embedding_mode == "synthetic":
        base = synthetic_embedding(key)
    elif settings.embedding_mode in {"cached", "hybrid_text"}:
        base = cache.item(dataset_id, key)
    else:
        if media is None:
            raise ValueError("real embedding mode requires a video path or frame list")
        from app.embedding.qwen import embed_video

        base = embed_video(media)
    return truncate_and_normalize(base, dimension)


def mode_details(dataset_id: str | None = None) -> dict[str, object]:
    if dataset_id:
        from app.db import postgres

        info = postgres.dataset_info(dataset_id)
        if info is not None and info.get("vector_provenance") == "synthetic":
            return {
                "mode": settings.embedding_mode,
                "vector_provenance": "synthetic",
                "level": "danger",
                "message": (
                    f"{dataset_id}: SENTETİK EMBEDDING — sonuç sıralamaları anlamsızdır. "
                    "Yalnızca sistem/gecikme doğrulaması."
                ),
            }
    if settings.embedding_mode == "synthetic":
        return {
            "mode": "synthetic",
            "vector_provenance": "synthetic",
            "level": "danger",
            "message": "SENTETİK EMBEDDING — sonuç sıralamaları anlamsızdır. Yalnızca sistem/gecikme doğrulaması.",
        }
    if settings.embedding_mode == "cached":
        count = cache.count(dataset_id) if dataset_id else 0
        return {
            "mode": "cached",
            "vector_provenance": "real",
            "synthetic_fallback": False,
            "level": "success",
            "message": f"GERÇEK EMBEDDING (Qwen3-VL-2B, cached) — {dataset_id or 'dataset'}: {count} vektör",
        }
    if settings.embedding_mode == "hybrid_text":
        from app.embedding.text_cpu import runtime_details

        runtime = runtime_details()
        return {
            "mode": "hybrid_text",
            "vector_provenance": "real",
            "level": "success",
            "message": (
                f"GERCEK EMBEDDING ({runtime['model_id']}@{runtime['model_revision']}, "
                f"{runtime['dtype']}, CPU) - {dataset_id or 'dataset'}: "
                f"{cache.count(dataset_id) if dataset_id else 0} item, "
                f"{cache.query_cache_count()} query cache; cache miss sorgular CPU'da canli hesaplanir"
            ),
            **runtime,
            "cached_items": cache.count(dataset_id) if dataset_id else 0,
            "cached_queries": cache.query_cache_count(),
            "synthetic_fallback": False,
        }
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable"
        dtype = "float16" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8 else "bfloat16"
    except ImportError:
        gpu, dtype = "torch unavailable", "unknown"
    return {
        "mode": "real",
        "vector_provenance": "real",
        "synthetic_fallback": False,
        "level": "success",
        "message": f"GERÇEK EMBEDDING (Qwen3-VL-2B, real) — {dtype}, {gpu}",
    }
