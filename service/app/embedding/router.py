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
        base = cache.query(text)
    else:
        from app.embedding.qwen import embed_text

        base = embed_text(text)
    return truncate_and_normalize(base, dimension)


def embed_item(key: str, dimension: int, *, dataset_id: str, media: str | list[str] | None = None) -> np.ndarray:
    if settings.embedding_mode == "synthetic":
        base = synthetic_embedding(key)
    elif settings.embedding_mode == "cached":
        base = cache.item(dataset_id, key)
    else:
        if media is None:
            raise ValueError("real embedding mode requires a video path or frame list")
        from app.embedding.qwen import embed_video

        base = embed_video(media)
    return truncate_and_normalize(base, dimension)


def mode_details(dataset_id: str | None = None) -> dict[str, object]:
    if settings.embedding_mode == "synthetic":
        return {
            "mode": "synthetic",
            "level": "danger",
            "message": "SENTETİK EMBEDDING — sonuç sıralamaları anlamsızdır. Yalnızca sistem/gecikme doğrulaması.",
        }
    if settings.embedding_mode == "cached":
        count = cache.count(dataset_id) if dataset_id else 0
        return {
            "mode": "cached",
            "level": "success",
            "message": f"GERÇEK EMBEDDING (Qwen3-VL-2B, cached) — {dataset_id or 'dataset'}: {count} vektör",
        }
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable"
        dtype = "float16" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8 else "bfloat16"
    except ImportError:
        gpu, dtype = "torch unavailable", "unknown"
    return {
        "mode": "real",
        "level": "success",
        "message": f"GERÇEK EMBEDDING (Qwen3-VL-2B, real) — {dtype}, {gpu}",
    }

