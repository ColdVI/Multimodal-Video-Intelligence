from __future__ import annotations

import numpy as np


def truncate_and_normalize(vector: np.ndarray, dimension: int) -> np.ndarray:
    source = np.asarray(vector, dtype=np.float32).reshape(-1)
    if dimension <= 0 or dimension > source.size:
        raise ValueError(f"dimension must be in [1, {source.size}], got {dimension}")
    result = source[:dimension].copy()
    if not np.isfinite(result).all():
        raise ValueError("embedding contains NaN or Inf")
    norm = float(np.linalg.norm(result))
    if norm == 0.0:
        raise ValueError("embedding norm is zero")
    result /= norm
    if not np.isclose(np.linalg.norm(result), 1.0, atol=1e-5):
        raise AssertionError("MRL normalization contract failed")
    return result

