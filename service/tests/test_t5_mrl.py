from __future__ import annotations

import numpy as np

from app.bench.quality import halfvec_quantization_experiment


def test_t5_2048_halfvec_is_reported_as_quantization_not_exact_equality():
    rng = np.random.default_rng(7)
    items = rng.normal(size=(5, 2048)).astype(np.float32)
    items /= np.linalg.norm(items, axis=1, keepdims=True)
    queries = items[:3].copy()
    result = halfvec_quantization_experiment(items, queries, np.asarray([0, 1, 2]))
    assert result["dimension"] == 2048
    assert result["storage"] == "halfvec"
    assert result["exact_equality_required"] is False
    assert result["mean_item_cosine_after_quantization"] > 0.999
