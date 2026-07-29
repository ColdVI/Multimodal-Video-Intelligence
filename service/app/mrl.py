import numpy as np

SUPPORTED_DIMS = (2048, 1024, 512, 256)


def truncate_and_normalize(vector: np.ndarray, dim: int) -> np.ndarray:
    if dim not in SUPPORTED_DIMS:
        raise ValueError(f"unsupported dimension: {dim}")
    value = np.asarray(vector, dtype=np.float32)
    if value.ndim != 1 or value.size < dim or not np.isfinite(value).all():
        raise ValueError("embedding must be a finite 1-D vector with enough dimensions")
    result = value[:dim].copy()
    norm = float(np.linalg.norm(result))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("embedding norm must be finite and non-zero")
    result /= norm
    if abs(float(np.linalg.norm(result)) - 1.0) > 1e-5:
        raise AssertionError("MRL result is not unit-normalized")
    return result
