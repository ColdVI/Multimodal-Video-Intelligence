from __future__ import annotations

import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from app.db import clickhouse, postgres
from app.search import engine

DATASET_ID = "advret_parser_integration_test"
DIMENSION = 512


def _vector(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIMENSION).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


@pytest.fixture
def rules_parser_corpus():
    """bus_count/vehicle_count/person_count filtering is only supported end-to-end for
    run-scoped data (app.search.filter_projection.POSTGRES_RUN_COLUMNS, used by both
    filter_run_segment_ids() and native ClickHouse/pgvector/Qdrant pushdown) -- the
    legacy no-active-run candidate path (postgres.filter_segment_ids) only supports
    event_category/split/video_id/a few telemetry fields by design. So proving the
    parser->engine wiring actually filters requires a real active run, not just a
    legacy dataset registration."""
    if not (clickhouse.health() and postgres.health()):
        pytest.skip("live ClickHouse/Postgres not reachable; parser->engine integration is NOT RUN")
    run_id = str(uuid.uuid4())
    started_now = "now()"
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
        videos = [(DATASET_ID, "parser_video", None, None, None, None)]
        segments = [(f"parser_seg_{i:03d}", DATASET_ID, "parser_video", float(i), float(i + 1), None) for i in range(10)]
        metadata = [
            (f"parser_seg_{i:03d}", 0, 1 if i < 4 else 0, 1 if i < 4 else 0, None, None, None)
            for i in range(10)
        ]
        postgres.write_run_metadata_chunk(run_id, DATASET_ID, 0, videos, segments, metadata, [])

    ch_rows = [
        {
            "run_id": run_id, "chunk_index": 0, "segment_id": f"parser_seg_{i:03d}", "dataset_id": DATASET_ID,
            "video_id": "parser_video", "t_start": float(i), "t_end": float(i + 1),
            "event_category": None, "split": None, "latitude": None, "longitude": None,
            "altitude_m": None, "velocity_mps": None, "roll": None, "pitch": None, "yaw": None,
            "yaw_rate": None, "gimbal_pitch": None, "gimbal_heading": None, "compass_heading": None,
            "person_count": 0, "vehicle_count": 1 if i < 4 else 0, "bus_count": 1 if i < 4 else 0,
            "is_night": 0, "embedding": _vector(i),
        }
        for i in range(10)
    ]
    clickhouse.write_run_chunk(run_id, DATASET_ID, DIMENSION, 0, ch_rows)
    try:
        yield run_id
    finally:
        # _ensure_inactive() (both clickhouse.delete_run and the Postgres-side FK chain)
        # forbids destructive cleanup of the *active* run -- deactivate in Postgres first,
        # then delete_run can proceed.
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
        query="otobüs bulunan sahne", dataset_id=DATASET_ID, backend="clickhouse", strategy="exact",
        dimension=DIMENSION, adaptive_mrl=SimpleNamespace(enabled=False, base_dim=256, top_n=100),
        metadata_filters={}, telemetry_filters={}, pattern="A", top_k=10, repeats=1,
        filter_execution_mode="pushdown", diagnose=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_rules_parser_mode_filters_live_results_by_ontology_derived_constraint(rules_parser_corpus):
    """No explicit metadata_filters at all -- 'otobüs' alone, parsed via parser_mode=
    'rules', must resolve to bus_count>=1 and restrict live ClickHouse results to
    exactly the 4 rows that actually have a bus, end to end through engine.search()."""
    response = engine.search(_request(parser_mode="rules"))
    returned_ids = {row["segment_id"] for row in response["results"]}
    assert returned_ids == {f"parser_seg_{i:03d}" for i in range(4)}
    assert response["diagnostics"]["parser"]["parser_mode"] == "rules"
    assert response["execution_policy"]["parser_mode"] == "rules"


def test_parser_mode_none_ignores_query_text_as_a_filter_source(rules_parser_corpus):
    """Default behavior must be completely unchanged: with parser_mode=none (the
    default), the same 'otobüs' query text must NOT filter anything -- only explicit
    metadata_filters would, and there are none here, so all 10 rows are candidates."""
    response = engine.search(_request())
    assert len(response["results"]) == 10
    assert response["diagnostics"]["parser"] is None
    assert response["execution_policy"]["parser_mode"] == "none"


def test_explicit_filter_overrides_parser_derived_constraint_on_same_field(rules_parser_corpus):
    """Explicit request filters must win over parser-derived soft constraints on the
    same field -- 'otobüs' alone (rules parser) would infer bus_count>=1 and restrict to
    4 rows, but an explicit bus_count>=0 (i.e. no real restriction) on the same field
    must not be shadowed by the parser's inference."""
    response = engine.search(_request(parser_mode="rules", metadata_filters={"bus_count": {"min": 0}}))
    assert len(response["results"]) == 10  # explicit bus_count>=0 wins over the parser's bus_count>=1
