from __future__ import annotations

import numpy as np
import pytest

from app.search.exact_rerank import evaluate_physical_read_gate, rerank_candidates_exact
from app.db import clickhouse

DATASET_ID = "advret_exact_rerank_test"
DIMENSION = 512


def test_gate_passes_when_rows_read_scales_with_candidate_count_not_partition():
    verdict = evaluate_physical_read_gate(rows_read=250, candidate_count=50, partition_size=1_000_000)
    assert verdict.status == "passed"
    assert verdict.rows_per_candidate == 5.0


def test_gate_fails_when_rows_read_is_most_of_the_partition_despite_few_candidates():
    verdict = evaluate_physical_read_gate(rows_read=950_000, candidate_count=50, partition_size=1_000_000)
    assert verdict.status == "failed"
    assert "full partition scan" in verdict.reason


def test_gate_fails_when_rows_per_candidate_exceeds_tolerance_even_without_partition_size():
    verdict = evaluate_physical_read_gate(rows_read=10_000, candidate_count=50, partition_size=None)
    assert verdict.status == "failed"
    assert verdict.rows_per_candidate == 200.0


def test_gate_reports_not_run_when_rows_read_was_never_measured():
    verdict = evaluate_physical_read_gate(rows_read=None, candidate_count=50, partition_size=1000)
    assert verdict.status == "not_run"


def test_gate_rejects_non_positive_candidate_count():
    with pytest.raises(ValueError):
        evaluate_physical_read_gate(rows_read=100, candidate_count=0, partition_size=1000)


def test_gate_custom_tolerance_factor_is_respected():
    verdict = evaluate_physical_read_gate(rows_read=1000, candidate_count=50, partition_size=None, tolerance_factor=50.0)
    assert verdict.status == "passed"  # 20/candidate, under the relaxed 50x tolerance


def test_unsupported_backend_falls_back_with_explicit_note_not_silently():
    rows, diagnostics = rerank_candidates_exact(
        DATASET_ID, DIMENSION, [0.0] * DIMENSION, ["seg1"], top_k=5, backend="milvus",
    )
    assert rows == []
    assert diagnostics["notes"] == ["exact_rerank_unsupported:milvus"]
    assert diagnostics["rows_read"] is None


def test_empty_candidate_ids_raises_rather_than_running_an_unbounded_query():
    with pytest.raises(ValueError, match="non-empty"):
        rerank_candidates_exact(DATASET_ID, DIMENSION, [0.0] * DIMENSION, [], top_k=5)


def _vector(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIMENSION).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


@pytest.fixture
def exact_rerank_corpus():
    if not clickhouse.health():
        pytest.skip("live ClickHouse is not reachable; exact rerank live test is NOT RUN")
    rows = [
        {
            "segment_id": f"exact_seg_{i:03d}", "dataset_id": DATASET_ID, "video_id": "exact_video",
            "t_start": float(i), "t_end": float(i + 1), "altitude_m": 10.0, "velocity_mps": 1.0,
            "gimbal_pitch": 0.0, "person_count": 0, "vehicle_count": 0, "is_night": 0,
            "embedding": _vector(i),
        }
        for i in range(30)
    ]
    clickhouse.replace_vectors(DATASET_ID, DIMENSION, rows)
    try:
        yield rows
    finally:
        clickhouse.replace_vectors(DATASET_ID, DIMENSION, [])


def test_rerank_candidates_exact_only_ranks_within_the_candidate_set_live(exact_rerank_corpus):
    candidate_ids = [row["segment_id"] for row in exact_rerank_corpus[:5]]
    rows, diagnostics = rerank_candidates_exact(
        DATASET_ID, DIMENSION, _vector(999), candidate_ids, top_k=3,
    )
    assert len(rows) == 3
    assert {row["segment_id"] for row in rows}.issubset(set(candidate_ids))
    assert diagnostics["candidate_count"] == 5
    assert diagnostics["rows_read"] is not None
    assert diagnostics["notes"] == []


def test_rerank_candidates_exact_matches_brute_force_ranking_live(exact_rerank_corpus):
    """Correctness cross-check against numpy_exact (the project's own designated
    correctness reference, per plan Sec.20) -- both must agree on the top-k order for
    the same candidate-restricted set."""
    from app.search import common_exact

    candidate_ids = [row["segment_id"] for row in exact_rerank_corpus[:10]]
    query_vector = _vector(999)
    rerank_rows, _ = rerank_candidates_exact(DATASET_ID, DIMENSION, query_vector, candidate_ids, top_k=5)
    reference_rows, _ = common_exact.search_vectors(
        DATASET_ID, DIMENSION, query_vector, top_k=5, strategy="exact", candidate_ids=candidate_ids,
    )
    assert [row["segment_id"] for row in rerank_rows] == [row["segment_id"] for row in reference_rows]
