from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import settings


class CachedEmbeddingStore:
    def __init__(self, root: Path | None = None):
        self.root = root or settings.artifacts_root / "embeddings"

    @lru_cache(maxsize=8)
    def _dataset(self, dataset_id: str) -> tuple[np.ndarray, dict[str, int]]:
        vectors_path = self.root / f"{dataset_id}_2048.npy"
        ids_path = self.root / f"{dataset_id}_ids.parquet"
        vectors = np.load(vectors_path, mmap_mode="r")
        ids = pd.read_parquet(ids_path)
        id_col = "segment_id" if "segment_id" in ids.columns else "id"
        if len(ids) != len(vectors):
            raise ValueError(f"cached id/vector count mismatch for {dataset_id}")
        return vectors, {str(value): i for i, value in enumerate(ids[id_col])}

    def item(self, dataset_id: str, segment_id: str) -> np.ndarray:
        vectors, positions = self._dataset(dataset_id)
        if segment_id not in positions:
            raise KeyError(f"cached embedding missing: {segment_id}")
        return np.asarray(vectors[positions[segment_id]], dtype=np.float32)

    @lru_cache(maxsize=1)
    def _queries(self) -> dict[str, np.ndarray]:
        path = self.root / "query_embeddings.json"
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {key: np.asarray(value, dtype=np.float32) for key, value in payload.items()}

    def query(self, text: str) -> np.ndarray:
        try:
            return self._queries()[text]
        except KeyError as exc:
            raise KeyError(
                "cached mode requires this free-text query in artifacts/embeddings/query_embeddings.json"
            ) from exc

    def count(self, dataset_id: str) -> int:
        try:
            vectors, _ = self._dataset(dataset_id)
        except (FileNotFoundError, ValueError):
            return 0
        return len(vectors)


cache = CachedEmbeddingStore()

