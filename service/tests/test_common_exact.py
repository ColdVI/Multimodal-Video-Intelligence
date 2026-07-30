from __future__ import annotations

import numpy as np

from app.search.common_exact import stable_top_k


def test_stable_exact_has_perfect_recall_against_itself():
    rng = np.random.default_rng(42)
    matrix = rng.standard_normal((200, 32)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    query = rng.standard_normal(32).astype(np.float32)
    query /= np.linalg.norm(query)
    ids = [f"s{i:03d}" for i in range(200)]
    expected = stable_top_k(matrix, query, ids, 10)
    actual = stable_top_k(matrix.copy(), query.copy(), ids.copy(), 10)
    assert {row["segment_id"] for row in actual} == {row["segment_id"] for row in expected}


def test_stable_sort_breaks_ties_by_input_order():
    matrix = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    rows = stable_top_k(matrix, np.array([1, 0], dtype=np.float32), ["a", "b", "c"], 2)
    assert [row["segment_id"] for row in rows] == ["a", "b"]

