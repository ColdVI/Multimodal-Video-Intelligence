from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import settings


class _ProcessFileLock:
    """Kucuk, platform-bagimsiz O_EXCL kilidi; runtime query cache yazarlari icin."""

    def __init__(self, path: Path, timeout_s: float = 30.0):
        self.path = path
        self.timeout_s = timeout_s
        self.fd: int | None = None

    def __enter__(self):
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, f"{os.getpid()}\n".encode("ascii"))
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"query cache lock timeout: {self.path}")
                time.sleep(0.05)

    def __exit__(self, exc_type, exc, tb):
        if self.fd is not None:
            os.close(self.fd)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _cache_key(model_revision: str, text: str) -> str:
    return hashlib.sha256(f"{model_revision}\0{text}".encode("utf-8")).hexdigest()


def _validated_vector(value: Any, source: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (2048,):
        raise ValueError(f"{source}: expected (2048,), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{source}: NaN/Inf embedding")
    return vector


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
        if vectors.ndim != 2 or vectors.shape[1] != 2048:
            raise ValueError(f"cached item shape mismatch for {dataset_id}: {vectors.shape}")
        if len(ids) != len(vectors):
            raise ValueError(f"cached id/vector count mismatch for {dataset_id}")
        return vectors, {str(value): i for i, value in enumerate(ids[id_col])}

    def item(self, dataset_id: str, segment_id: str) -> np.ndarray:
        vectors, positions = self._dataset(dataset_id)
        if segment_id not in positions:
            raise KeyError(f"cached embedding missing: {segment_id}")
        return _validated_vector(vectors[positions[segment_id]], f"{dataset_id}:{segment_id}")

    @lru_cache(maxsize=1)
    def _fixed_queries(self) -> tuple[str | None, dict[str, np.ndarray]]:
        """Yalniz sabit demo sorgulari; tum CapERA GT bu JSON'a konmaz."""
        path = self.root / "query_embeddings.json"
        if not path.exists():
            return None, {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "queries" in payload:
            revision = payload.get("model_revision")
            values = payload["queries"]
        else:
            revision = None
            values = payload
        return revision, {
            str(text): _validated_vector(vector, f"query_embeddings.json:{text}")
            for text, vector in values.items()
        }

    @lru_cache(maxsize=4)
    def _bulk_queries(self, dataset_id: str) -> tuple[np.ndarray, dict[str, int]]:
        vectors_path = self.root / f"{dataset_id}_queries_2048.npy"
        ids_path = self.root / f"{dataset_id}_query_ids.parquet"
        vectors = np.load(vectors_path, mmap_mode="r")
        ids = pd.read_parquet(ids_path)
        if "query_text" not in ids.columns:
            raise ValueError(f"{ids_path.name}: query_text column missing")
        if vectors.shape != (len(ids), 2048):
            raise ValueError(f"cached query id/vector shape mismatch for {dataset_id}: {vectors.shape}/{len(ids)}")
        positions: dict[str, int] = {}
        for position, value in enumerate(ids["query_text"]):
            positions.setdefault(str(value), position)
        return vectors, positions

    def _runtime_payload(self) -> dict[str, Any]:
        path = self.root / "query_cache.json"
        if not path.exists():
            return {"schema_version": 1, "entries": {}}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or not isinstance(payload.get("entries"), dict):
            raise ValueError("query_cache.json schema is invalid")
        return payload

    def query_or_none(self, text: str, model_revision: str, *, dataset_id: str = "capera") -> np.ndarray | None:
        fixed_revision, fixed = self._fixed_queries()
        if text in fixed and fixed_revision in (None, model_revision):
            return fixed[text].copy()
        try:
            vectors, positions = self._bulk_queries(dataset_id)
            if text in positions:
                return _validated_vector(vectors[positions[text]], f"{dataset_id}:query:{text}")
        except FileNotFoundError:
            pass
        entry = self._runtime_payload()["entries"].get(_cache_key(model_revision, text))
        if entry is None:
            return None
        if entry.get("model_revision") != model_revision or entry.get("text") != text:
            raise ValueError("query cache key/content mismatch")
        return _validated_vector(entry["embedding"], "query_cache.json")

    def query(self, text: str, model_revision: str, *, dataset_id: str = "capera") -> np.ndarray:
        vector = self.query_or_none(text, model_revision, dataset_id=dataset_id)
        if vector is None:
            raise KeyError(
                "cached mode has no real embedding for this text; add it to the fixed demo cache "
                "or use EMBEDDING_MODE=hybrid_text with a provisioned Qwen model"
            )
        return vector

    def put_query(self, text: str, model_revision: str, vector: np.ndarray) -> None:
        vector = _validated_vector(vector, "runtime query")
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / "query_cache.json"
        lock = target.with_suffix(target.suffix + ".lock")
        with _ProcessFileLock(lock):
            payload = self._runtime_payload()
            payload["entries"][_cache_key(model_revision, text)] = {
                "model_revision": model_revision,
                "text": text,
                "embedding": vector.tolist(),
            }
            fd, temporary_name = tempfile.mkstemp(prefix="query_cache.", suffix=".tmp", dir=self.root)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, target)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)

    def count(self, dataset_id: str) -> int:
        try:
            vectors, _ = self._dataset(dataset_id)
        except (FileNotFoundError, ValueError):
            return 0
        return len(vectors)

    def query_cache_count(self) -> int:
        fixed_count = len(self._fixed_queries()[1])
        runtime_count = len(self._runtime_payload()["entries"])
        return fixed_count + runtime_count


cache = CachedEmbeddingStore()


__all__ = ["CachedEmbeddingStore", "cache", "_cache_key"]
