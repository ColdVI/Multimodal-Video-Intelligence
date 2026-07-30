from __future__ import annotations

import hashlib

import numpy as np


# NEDEN: DB/index/API/UI/latency yolunun tamamı GPU olmadan uçtan uca
# test edilebilsin. Bu vektörler SEMANTİK DEĞİL — kalite ölçümünde
# ASLA kullanılmaz, sadece sistem doğrulaması içindir.
def synthetic_embedding(key: str, dim: int = 2048) -> np.ndarray:
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % (2**32)
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)

