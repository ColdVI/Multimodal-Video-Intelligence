from __future__ import annotations

import numpy as np
import pytest

from app.embedding.synthetic import synthetic_embedding
from app.mrl import truncate_and_normalize


def test_synthetic_embedding_is_deterministic_normalized_and_distinct():
    first = synthetic_embedding("alpha", 2048)
    second = synthetic_embedding("alpha", 2048)
    other = synthetic_embedding("beta", 2048)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)
    assert first.dtype == np.float32
    assert np.linalg.norm(first) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("dimension", [2048, 1024, 512, 256])
def test_mrl_contract(dimension):
    result = truncate_and_normalize(synthetic_embedding("segment"), dimension)
    assert result.shape == (dimension,)
    assert np.isfinite(result).all()
    assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)


def test_mrl_rejects_invalid_vectors():
    with pytest.raises(ValueError, match="NaN"):
        truncate_and_normalize(np.array([np.nan, 1], dtype=np.float32), 2)

