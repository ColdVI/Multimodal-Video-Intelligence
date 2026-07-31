from __future__ import annotations

import numpy as np
import pytest

from app.db import clickhouse, postgres, qdrant
from app.search import engine

pytestmark = pytest.mark.filterwarnings("ignore")

DATASET_ID = "advret_phase_neg1_test"
DIMENSION = 512


def _vector(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(DIMENSION).astype(np.float32)
    return (values / np.linalg.norm(values)).tolist()


def _row(index: int) -> dict:
    return {
        "segment_id": f"advret_seg_{index:03d}", "dataset_id": DATASET_ID, "video_id": "advret_video",
        "t_start": float(index), "t_end": float(index + 1), "altitude_m": 10.0, "velocity_mps": 1.0,
        "gimbal_pitch": 0.0, "person_count": 1, "vehicle_count": 0, "is_night": 0,
        "embedding": _vector(index),
    }


@pytest.fixture
def clickhouse_test_corpus():
    if not clickhouse.health():
        pytest.skip("live ClickHouse is not reachable; Phase -1 diagnostics contract is NOT RUN")
    rows = [_row(index) for index in range(20)]
    clickhouse.replace_vectors(DATASET_ID, DIMENSION, rows)
    try:
        yield rows
    finally:
        clickhouse.replace_vectors(DATASET_ID, DIMENSION, [])


def test_clickhouse_candidate_count_reflects_candidate_restriction_not_full_corpus(clickhouse_test_corpus):
    """Regression test for the pre-fix bug where count_sql omitted candidate_clause: a
    stage-2-style call restricted to a 5-row candidate subset out of a 20-row corpus must
    report candidate_count==5, not 20."""
    candidate_ids = [row["segment_id"] for row in clickhouse_test_corpus[:5]]
    rows, diagnostics = clickhouse.search_vectors(
        DATASET_ID, DIMENSION, _vector(999), top_k=3, strategy="exact",
        candidate_ids=candidate_ids, diagnose=True,
    )
    assert diagnostics["candidate_count"] == 5
    assert diagnostics["filtered_corpus_count"] == 20
    assert diagnostics["candidate_input_count"] == 5
    assert len(rows) == 3


def test_clickhouse_well_filled_default_request_does_not_compute_count_or_explain(clickhouse_test_corpus):
    """Phase -1.3: on the hot path (not underfilled, diagnose/explain not requested),
    count()/EXPLAIN must not run -- candidate_count and plan_used_vector_index must be
    honestly None (not measured), not a stale/implicit value."""
    rows, diagnostics = clickhouse.search_vectors(
        DATASET_ID, DIMENSION, _vector(999), top_k=3, strategy="exact", candidate_ids=None,
    )
    assert len(rows) == 3  # well-filled: corpus has 20 rows, only 3 requested
    assert diagnostics["candidate_count"] is None
    assert diagnostics["filtered_corpus_count"] is None
    assert diagnostics["candidate_count_status"] == "not_requested"
    assert diagnostics["plan_used_vector_index"] is None
    assert diagnostics["explain_status"] == "not_requested"


def test_clickhouse_underfilled_request_computes_count_even_without_diagnose_flag(clickhouse_test_corpus):
    """When the corpus itself can't satisfy top_k, candidate_shortage/ann_filter_loss
    classification needs a real count -- so count() must run even with diagnose=False."""
    rows, diagnostics = clickhouse.search_vectors(
        DATASET_ID, DIMENSION, _vector(999), top_k=1000, strategy="exact", candidate_ids=None,
    )
    assert len(rows) == 20
    assert diagnostics["candidate_count"] == 20
    assert diagnostics["candidate_count_status"] == "computed"


def test_clickhouse_explain_flag_controls_plan_used_vector_index(clickhouse_test_corpus):
    rows, diagnostics = clickhouse.search_vectors(
        DATASET_ID, DIMENSION, _vector(999), top_k=3, strategy="exact", candidate_ids=None, explain=True,
    )
    assert diagnostics["explain_status"] == "computed"
    assert diagnostics["plan_used_vector_index"] in (True, False)


def test_clickhouse_empty_candidate_ids_short_circuits_without_query(clickhouse_test_corpus):
    rows, diagnostics = clickhouse.search_vectors(
        DATASET_ID, DIMENSION, _vector(999), top_k=3, strategy="exact", candidate_ids=[],
    )
    assert rows == []
    assert diagnostics["candidate_count"] == 0
    assert diagnostics["filtered_corpus_count"] == 0


@pytest.fixture
def postgres_test_corpus():
    if not postgres.health():
        pytest.skip("live Postgres is not reachable; Phase -1 diagnostics contract is NOT RUN")
    postgres.init_schema(include_vectors=True, dimensions=(DIMENSION,))
    rows = [(f"advret_pg_{index:03d}", DATASET_ID, _vector(index)) for index in range(20)]
    postgres.upsert_vectors(DIMENSION, rows)
    with postgres.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO datasets(dataset_id, has_telemetry, has_captions, vector_provenance) "
            "VALUES (%s, false, false, 'synthetic') ON CONFLICT (dataset_id) DO NOTHING",
            (DATASET_ID,),
        )
        cur.execute(
            "INSERT INTO videos(dataset_id, video_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (DATASET_ID, "advret_video"),
        )
        for index in range(20):
            cur.execute(
                "INSERT INTO segments(segment_id, dataset_id, video_id, t_start, t_end) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (segment_id) DO UPDATE SET t_start=EXCLUDED.t_start",
                (f"advret_pg_{index:03d}", DATASET_ID, "advret_video", float(index), float(index + 1)),
            )
        conn.commit()
    try:
        yield rows
    finally:
        table, _ = postgres.VECTOR_TABLES[DIMENSION]
        with postgres.connection() as conn, conn.cursor() as cur:
            cur.execute(f"DELETE FROM {table} WHERE dataset_id=%s", (DATASET_ID,))
            cur.execute("DELETE FROM segments WHERE dataset_id=%s", (DATASET_ID,))
            cur.execute("DELETE FROM videos WHERE dataset_id=%s", (DATASET_ID,))
            cur.execute("DELETE FROM datasets WHERE dataset_id=%s", (DATASET_ID,))
            conn.commit()


def test_postgres_candidate_count_reflects_candidate_restriction_not_full_corpus(postgres_test_corpus):
    candidate_ids = [row[0] for row in postgres_test_corpus[:5]]
    rows, diagnostics = postgres.search_vectors(
        DATASET_ID, DIMENSION, _vector(999), top_k=3, strategy="exact",
        candidate_ids=candidate_ids, diagnose=True,
    )
    assert diagnostics["candidate_count"] == 5
    assert diagnostics["filtered_corpus_count"] == 20
    assert len(rows) == 3


def test_postgres_well_filled_default_request_does_not_compute_count(postgres_test_corpus):
    rows, diagnostics = postgres.search_vectors(
        DATASET_ID, DIMENSION, _vector(999), top_k=3, strategy="exact", candidate_ids=None,
    )
    assert diagnostics["candidate_count"] is None
    assert diagnostics["candidate_count_status"] == "not_requested"


@pytest.fixture
def qdrant_test_corpus():
    if not qdrant.health():
        pytest.skip("live Qdrant is not reachable; Phase -1 diagnostics contract is NOT RUN")
    qdrant.init_schema(dimensions=(DIMENSION,))
    rows = [
        {
            "segment_id": f"advret_qd_{index:03d}", "dataset_id": DATASET_ID,
            "embedding": _vector(index),
        }
        for index in range(20)
    ]
    qdrant.replace_vectors(DATASET_ID, DIMENSION, rows)
    try:
        yield rows
    finally:
        from qdrant_client import models

        qdrant.client().delete(
            collection_name=qdrant.collection_name(DIMENSION),
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[models.FieldCondition(
                    key="dataset_id", match=models.MatchValue(value=DATASET_ID),
                )])
            ),
            wait=True,
        )


def test_qdrant_candidate_count_reflects_candidate_restriction_not_full_corpus(qdrant_test_corpus):
    candidate_ids = [row["segment_id"] for row in qdrant_test_corpus[:5]]
    rows, diagnostics = qdrant.search_vectors(
        DATASET_ID, DIMENSION, _vector(999), top_k=3, strategy="exact",
        candidate_ids=candidate_ids, diagnose=True,
    )
    assert diagnostics["candidate_count"] == 5
    assert diagnostics["filtered_corpus_count"] == 20


def test_qdrant_well_filled_default_request_does_not_compute_count(qdrant_test_corpus):
    rows, diagnostics = qdrant.search_vectors(
        DATASET_ID, DIMENSION, _vector(999), top_k=3, strategy="exact", candidate_ids=None,
    )
    assert diagnostics["candidate_count"] is None
    assert diagnostics["candidate_count_status"] == "not_requested"


def test_adaptive_underfilled_final_result_with_full_stage1_is_ann_filter_loss_not_shortage(monkeypatch):
    """Reproduces the exact scenario the plan describes: stage-1 (base_dim/top_n) returns
    a FULL set of candidates (no shortage there), but stage-2 (dimension/top_k rerank of
    just those candidates) comes back short. Before the fix, engine.py's setdefault left
    stage-2's own (mis-scoped) candidate_count in place; now it must be classified as
    ann_filter_loss using stage1_returned_candidate_count, never candidate_shortage."""
    from types import SimpleNamespace

    request = SimpleNamespace(
        query="q", dataset_id="mini", backend="clickhouse", strategy="ann", dimension=512,
        adaptive_mrl=SimpleNamespace(enabled=True, base_dim=256, top_n=50),
        metadata_filters={}, telemetry_filters={}, pattern="A", top_k=10, repeats=1,
    )
    monkeypatch.setattr(engine.postgres, "dataset_info", lambda dataset_id: {
        "dataset_id": dataset_id, "has_telemetry": True, "has_captions": False, "vector_provenance": "real",
    })
    monkeypatch.setattr(engine.postgres, "get_active_run_snapshot", lambda dataset_id: None)
    monkeypatch.setattr(engine.postgres, "filter_segment_ids", lambda *a, **k: [f"s{i:03d}" for i in range(200)])
    monkeypatch.setattr(engine.postgres, "hydrate", lambda ids: [{"segment_id": i} for i in ids])
    monkeypatch.setattr(
        engine, "embed_query_multi",
        lambda text, dims: {dim: np.ones(dim, dtype=np.float32) / np.sqrt(dim) for dim in dims},
    )

    def stage_aware_search(dataset_id, dimension, query_vector, top_k, strategy, candidate_ids):
        if dimension == 256:  # stage 1: full top_n, no shortage at all
            selected = list(candidate_ids)[:top_k]
            return (
                [{"segment_id": v, "score": 1.0} for v in selected],
                {"plan_used_vector_index": True, "indexed_vectors_count": 200, "notes": [],
                 "candidate_count": len(selected), "filtered_corpus_count": len(candidate_ids)},
            )
        # stage 2: candidate_ids is the 50-item rerank set, but the ANN call only
        # returns 4 of them -- an ANN loss confined to stage 2, not a real shortage.
        return (
            [{"segment_id": v, "score": 1.0} for v in candidate_ids[:4]],
            {"plan_used_vector_index": True, "indexed_vectors_count": 200, "notes": [],
             "candidate_count": len(candidate_ids), "filtered_corpus_count": len(candidate_ids)},
        )

    monkeypatch.setitem(engine.BACKENDS, "clickhouse", stage_aware_search)
    response = engine.search(request)
    diagnostics = response["diagnostics"]
    assert diagnostics["stage1_returned_candidate_count"] == 50
    assert diagnostics["stage2_returned_count"] == 4
    assert diagnostics["final_returned_count"] == 4
    assert diagnostics["underfilled"] is True
    assert diagnostics["ann_filter_loss"] is True
    assert diagnostics["candidate_shortage"] is False
    assert diagnostics["underfilled_reason"] == "ann_filter_loss"
