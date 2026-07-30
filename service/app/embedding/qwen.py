from __future__ import annotations

import sys
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from app.config import settings

if TYPE_CHECKING:
    from PIL import Image


@lru_cache(maxsize=1)
def get_embedder():
    import torch

    repo_path = str(settings.qwen_repo_path)
    if not settings.qwen_repo_path.is_dir():
        raise RuntimeError(f"Qwen source is not provisioned at {settings.qwen_repo_path}")
    if not settings.qwen_model_path.is_dir():
        raise RuntimeError(f"Qwen model is not provisioned at {settings.qwen_model_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("EMBEDDING_MODE=real requires CUDA")
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

    capability = torch.cuda.get_device_capability()
    dtype = torch.float16 if capability[0] < 8 else torch.bfloat16
    return Qwen3VLEmbedder(
        model_name_or_path=str(settings.qwen_model_path),
        fps=1.0,
        max_frames=16,
        max_length=16384,
        torch_dtype=dtype,
        attn_implementation=settings.attn_impl,
    )


def embed_text(text: str) -> np.ndarray:
    result = get_embedder().process([{"text": text}])
    return result.detach().cpu().float().numpy()[0]


def embed_videos(videos: list[str | list[str] | list["Image.Image"]]) -> np.ndarray:
    """Embed one media batch with a single Qwen ``process`` call."""
    if not videos or any(not video for video in videos):
        raise ValueError("videos must be a non-empty batch of non-empty media inputs")
    result = get_embedder().process([{"video": video} for video in videos])
    vectors = result.detach().cpu().float().numpy().astype(np.float32, copy=False)
    if vectors.shape != (len(videos), 2048):
        raise RuntimeError(f"Qwen returned invalid video embedding shape: {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise RuntimeError("Qwen returned non-finite video embeddings")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise RuntimeError("Qwen returned a zero-norm video embedding")
    return (vectors / norms).astype(np.float32, copy=False)


def embed_video(video: str | list[str] | list["Image.Image"]) -> np.ndarray:
    """Compatibility wrapper for existing single-item callers."""
    return embed_videos([video])[0]


__all__ = ["embed_text", "embed_video", "embed_videos", "get_embedder"]
