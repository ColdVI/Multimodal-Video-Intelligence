"""Deterministic non-semantic embeddings used only for system validation."""
import hashlib

import numpy as np


def synthetic_embedding(key: str, dim: int = 2048) -> np.ndarray:
    # GPU olmadan DB/index/API/UI/gecikme yolunun tamamını doğrulamak içindir.
    # Bu vektörler semantik değildir ve kalite ölçümünde asla kullanılmaz.
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % (2**32)
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)
