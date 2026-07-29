from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import settings


@lru_cache(maxsize=8)
def _dataset_cache(dataset_id: str) -> tuple[np.ndarray, dict[str, int]]:
    root = settings.artifacts_dir / "embeddings"
    vectors_path = root / f"{dataset_id}_2048.npy"
    ids_path = root / f"{dataset_id}_ids.parquet"
    if not vectors_path.exists() or not ids_path.exists():
        raise FileNotFoundError(f"cached embeddings missing for {dataset_id}: {vectors_path}, {ids_path}")
    vectors = np.load(vectors_path, mmap_mode="r")
    ids = pd.read_parquet(ids_path)
    id_column = "segment_id" if "segment_id" in ids else "id"
    if len(vectors) != len(ids) or vectors.ndim != 2 or vectors.shape[1] != 2048:
        raise ValueError(f"invalid cached embedding contract for {dataset_id}")
    return vectors, {str(value): idx for idx, value in enumerate(ids[id_column])}


def cached_embedding(dataset_id: str, segment_id: str) -> np.ndarray:
    vectors, lookup = _dataset_cache(dataset_id)
    try:
        result = np.asarray(vectors[lookup[segment_id]], dtype=np.float32)
    except KeyError as exc:
        raise KeyError(f"no cached embedding for {segment_id}") from exc
    if not np.isfinite(result).all() or float(np.linalg.norm(result)) == 0:
        raise ValueError(f"invalid cached embedding for {segment_id}")
    return result / np.linalg.norm(result)


def cached_count(dataset_id: str) -> int:
    return int(_dataset_cache(dataset_id)[0].shape[0])
