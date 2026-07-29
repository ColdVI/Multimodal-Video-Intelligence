"""Lazy adapter for the verified Qwen3-VL-Embedding repository API."""
from __future__ import annotations

import sys
from functools import lru_cache

import numpy as np

from app.config import settings


@lru_cache(maxsize=1)
def get_embedder():
    if str(settings.qwen_repo) not in sys.path:
        sys.path.insert(0, str(settings.qwen_repo))
    import torch
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[settings.torch_dtype]
    return Qwen3VLEmbedder(model_name_or_path=settings.qwen_model, torch_dtype=dtype)


def process(payload: dict) -> np.ndarray:
    result = get_embedder().process([payload])
    if hasattr(result, "detach"):
        result = result.detach().cpu().float().numpy()
    vector = np.asarray(result[0], dtype=np.float32)
    return vector / np.linalg.norm(vector)
