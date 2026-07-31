from __future__ import annotations

import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from app.db import clickhouse, postgres
from app.search import engine

DATASET_ID = "advret_relaxation_integration_test"
DIMENSION = 512


def _vector(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIMENSION).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


@pytest.fixture
def relaxation_corpus():
    if not (clickhouse.health() and postgres.health()):
        pytest.skip("live ClickHouse/Postgres not reachable; relaxation integration is NOT RUN")
    run_id = str(uuid.uuid4())
    with postgres.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO datasets(dataset_id, has_telemetry, has_captions, vector_provenance) "
            "VALUES (%s, false, false, 'synthetic') ON CONFLICT (dataset_id) DO NOTHING",
            (DATASET_ID,),
        )
        cur.execute(
            "INSERT INTO ingest_runs(run_id, dataset_id, dataset_version, status, vector_provenance, "
            "model_id, model_revision, source_commit, enabled_backends, enabled_dimensions, manifest_hash, started_at, finished_at) "
            "VALUES (%s, %s, 'v1', 'completed', 'synthetic', 'test-model', 'test-rev', 'test-commit', "
            "'[\"clickhouse\"]', '[512]', 'test-manifest-hash', now(), now())",
            (run_id, DATASET_ID),
        )
        cur.execute(
            "INSERT INTO dataset_active_runs(dataset_id, active_run_id, activated_at) VALUES (%s, %s, now())",
            (DATASET_ID, run_id),
        )
        conn.commit()
        videos = [(DATASET_ID, "relax_video", None, None, None, None)]
        # Only ONE segment has both bus_count>=1 AND person_count>=1 (the rules parser's
        # combined inference from "otobüs ve yaya"); 5 more have only bus_count>=1.
        # Pass 1 (both constraints) underfills at top_k=3; Pass 3 (drop the parser's
        # lower-confidence person_count soft constraint) should rescue it to 6 results.
        segments = [(f"relax_seg_{i:03d}", DATASET_ID, "relax_video", float(i), float(i + 1), None) for i in range(6)]
        metadata = [
            (f"relax_seg_{i:03d}", 1 if i == 0 else 0, 0, 1, None, None, None)
            for i in range(6)
        ]
        postgres.write_run_metadata_chunk(run_id, DATASET_ID, 0, videos, segments, metadata, [])

    ch_rows = [
        {
            "run_id": run_id, "chunk_index": 0, "segment_id": f"relax_seg_{i:03d}", "dataset_id": DATASET_ID,
            "video_id": "relax_video", "t_start": float(i), "t_end": float(i + 1),
            "event_category": None, "split": None, "latitude": None, "longitude": None,
            "altitude_m": None, "velocity_mps": None, "roll": None, "pitch": None, "yaw": None,
            "yaw_rate": None, "gimbal_pitch": None, "gimbal_heading": None, "compass_heading": None,
            "person_count": 1 if i == 0 else 0, "vehicle_count": 0, "bus_count": 1,
            "is_night": 0, "embedding": _vector(i),
        }
        for i in range(6)
    ]
    clickhouse.write_run_chunk(run_id, DATASET_ID, DIMENSION, 0, ch_rows)
    try:
        yield run_id
    finally:
        with postgres.connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM dataset_active_runs WHERE dataset_id=%s", (DATASET_ID,))
            conn.commit()
        clickhouse.delete_run(DATASET_ID, run_id, DIMENSION)
        with postgres.connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM run_segment_metadata WHERE run_id=%s", (run_id,))
            cur.execute("DELETE FROM run_segments WHERE run_id=%s", (run_id,))
            cur.execute("DELETE FROM run_videos WHERE run_id=%s", (run_id,))
            cur.execute("DELETE FROM ingest_runs WHERE run_id=%s", (run_id,))
            cur.execute("DELETE FROM datasets WHERE dataset_id=%s", (DATASET_ID,))
            conn.commit()


def _request(**overrides) -> SimpleNamespace:
    base = dict(
        query="otobüs ve yaya", dataset_id=DATASET_ID, backend="clickhouse", strategy="exact",
        dimension=DIMENSION, adaptive_mrl=SimpleNamespace(enabled=False, base_dim=256, top_n=100),
        metadata_filters={}, telemetry_filters={}, pattern="A", top_k=3, repeats=1,
        filter_execution_mode="pushdown", diagnose=True, parser_mode="rules",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_auto_soft_relaxation_rescues_an_underfilled_result(relaxation_corpus):
    """'otobüs ve yaya' -> rules parser infers bus_count>=1 AND person_count>=1. Only 1
    of 6 segments satisfies both, so top_k=3 is underfilled at Pass 1. auto_soft with
    min_results=3 must walk the ladder and rescue it by dropping person_count (both
    constraints are equal-confidence rules_parser output, so the ladder's pass-3 step
    -- which targets rules_parser/llm_parser sources -- drops both together, and the
    resulting bus_count-only filter satisfies min_results with 6 matches)."""
    response = engine.search(_request(filter_relaxation_mode="auto_soft", min_results=3))
    assert len(response["results"]) >= 3
    relaxation = response["diagnostics"]["filter_relaxation"]
    assert relaxation is not None
    assert relaxation["triggered"] is True
    assert relaxation["final_returned_count"] >= 3
    assert any(c["field"] == "person_count" for c in relaxation["relaxed_constraints"])


def test_diagnose_only_reports_the_same_finding_without_changing_returned_results(relaxation_corpus):
    """diagnose_only must compute the identical ladder outcome but never actually swap
    in the relaxed results -- the response still reflects Pass 1's real (underfilled)
    search."""
    response = engine.search(_request(filter_relaxation_mode="diagnose_only", min_results=3))
    assert len(response["results"]) == 1  # Pass 1's real, underfilled result -- unchanged
    relaxation = response["diagnostics"]["filter_relaxation"]
    assert relaxation["triggered"] is True
    assert relaxation["final_returned_count"] >= 3  # what auto_soft WOULD have returned


def test_relaxation_off_never_triggers_even_when_underfilled(relaxation_corpus):
    response = engine.search(_request(filter_relaxation_mode="off", min_results=3))
    assert len(response["results"]) == 1
    assert response["diagnostics"]["filter_relaxation"]["triggered"] is False


def test_relaxation_not_triggered_when_not_underfilled(relaxation_corpus):
    response = engine.search(_request(filter_relaxation_mode="auto_soft", min_results=1))
    assert response["diagnostics"]["filter_relaxation"]["triggered"] is False
    assert len(response["results"]) == 1
