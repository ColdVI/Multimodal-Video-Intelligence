from __future__ import annotations

import httpx
import numpy as np
import pytest

from app.bench.protocol import FLOAT32_EXACT_DIMENSIONS
from app.mrl import truncate_and_normalize
from app.search.common_exact import stable_top_k
from faz8_support import readiness


@pytest.mark.parametrize("dimension", [2048, 1024, 512, 256])
def test_t1_mrl_normalization_is_deterministic(dimension):
    source = np.arange(1, 2049, dtype=np.float32)
    first = truncate_and_normalize(source, dimension)
    second = truncate_and_normalize(source, dimension)
    np.testing.assert_array_equal(first, second)
    assert np.linalg.norm(first) == pytest.approx(1.0, abs=1e-5)


def test_t1_stable_tie_break_preserves_input_order():
    matrix = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    hits = stable_top_k(matrix, np.asarray([1, 0], dtype=np.float32), ["b", "a", "c"], 3)
    assert [row["segment_id"] for row in hits] == ["b", "a", "c"]


@pytest.mark.parametrize("dimension", FLOAT32_EXACT_DIMENSIONS)
def test_t1_cross_backend_exact_equality_only_for_float32_dimensions(dimension):
    readiness("system")
    payload = {
        "query": "dense traffic", "dataset_id": "auair", "strategy": "exact",
        "dimension": dimension, "top_k": 10, "repeats": 1,
    }
    ids = []
    for backend in ("clickhouse", "qdrant", "pgvector"):
        response = httpx.post(
            "http://localhost:8000/search",
            json={**payload, "backend": backend, "pattern": "C" if backend == "pgvector" else "A"},
            timeout=60,
        )
        response.raise_for_status()
        ids.append([row["segment_id"] for row in response.json()["results"]])
    assert ids[0] == ids[1] == ids[2]
