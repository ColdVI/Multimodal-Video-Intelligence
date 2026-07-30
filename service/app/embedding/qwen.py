from __future__ import annotations

import sys
from functools import lru_cache

import numpy as np

from app.config import settings


@lru_cache(maxsize=1)
def get_embedder():
    import torch

    repo_path = str(settings.qwen_repo_path)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

    if not torch.cuda.is_available():
        raise RuntimeError("EMBEDDING_MODE=real requires CUDA")
    capability = torch.cuda.get_device_capability()
    dtype = torch.float16 if capability[0] < 8 else torch.bfloat16
    return Qwen3VLEmbedder(
        model_name_or_path=str(settings.qwen_model_path),
        fps=1.0,
        max_frames=16,
        max_length=16384,
        torch_dtype=dtype,
    )


def embed_text(text: str) -> np.ndarray:
    result = get_embedder().process([{"text": text}])
    return result.detach().cpu().float().numpy()[0]


def embed_video(video: str | list[str]) -> np.ndarray:
    result = get_embedder().process([{"video": video}])
    return result.detach().cpu().float().numpy()[0]

