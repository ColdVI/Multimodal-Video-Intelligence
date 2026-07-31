from __future__ import annotations

import uuid

import numpy as np
import pytest
from pytest import approx

from app.db import clickhouse, postgres

DATASET_ID = "advret_enrichment_write_test"
DIMENSION = 512


def _vector(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIMENSION).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


@pytest.fixture
def enrichment_run():
    if not (clickhouse.health() and postgres.health()):
        pytest.skip("live ClickHouse/Postgres not reachable; enrichment write test is NOT RUN")
    run_id = str(uuid.uuid4())
    with postgres.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO datasets(dataset_id, has_telemetry, has_captions, vector_provenance) "
            "VALUES (%s, false, false, 'synthetic') ON CONFLICT (dataset_id) DO NOTHING",
            (DATASET_ID,),
        )
        cur.execute(
            "INSERT INTO ingest_runs(run_id, dataset_id, dataset_version, status, vector_provenance, "
            "model_id, model_revision, source_commit, enabled_backends, enabled_dimensions, manifest_hash, started_at) "
            "VALUES (%s, %s, 'v1', 'validating', 'synthetic', 'm', 'r', 'c', '[\"clickhouse\"]', '[512]', 'hash', now())",
            (run_id, DATASET_ID),
        )
        conn.commit()
        videos = [(DATASET_ID, "enrich_video", None, None, None, None)]
        segments = [("enrich_seg_000", DATASET_ID, "enrich_video", 0.0, 1.0, None)]
        metadata = [("enrich_seg_000", 0, 0, 0, None, None, None)]
        postgres.write_run_metadata_chunk(run_id, DATASET_ID, 0, videos, segments, metadata, [])
    clickhouse.write_run_chunk(run_id, DATASET_ID, DIMENSION, 0, [{
        "run_id": run_id, "chunk_index": 0, "segment_id": "enrich_seg_000", "dataset_id": DATASET_ID,
        "video_id": "enrich_video", "t_start": 0.0, "t_end": 1.0,
        "event_category": None, "split": None, "latitude": None, "longitude": None,
        "altitude_m": None, "velocity_mps": None, "roll": None, "pitch": None, "yaw": None,
        "yaw_rate": None, "gimbal_pitch": None, "gimbal_heading": None, "compass_heading": None,
        "person_count": 0, "vehicle_count": 0, "bus_count": 0, "is_night": 0, "embedding": _vector(0),
    }])
    try:
        yield run_id
    finally:
        with postgres.connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM run_segment_metadata WHERE run_id=%s", (run_id,))
            cur.execute("DELETE FROM run_segments WHERE run_id=%s", (run_id,))
            cur.execute("DELETE FROM run_videos WHERE run_id=%s", (run_id,))
            cur.execute("DELETE FROM ingest_runs WHERE run_id=%s", (run_id,))
            cur.execute("DELETE FROM datasets WHERE dataset_id=%s", (DATASET_ID,))
            conn.commit()
        clickhouse.client().command(
            f"ALTER TABLE seg_ch_{DIMENSION}_runs DELETE WHERE run_id={{run_id:UUID}}",
            parameters={"run_id": run_id}, settings={"mutations_sync": 2},
        )


def test_postgres_enrichment_write_sets_canonical_columns(enrichment_run):
    postgres.write_run_detector_enrichment(enrichment_run, DATASET_ID, [("enrich_seg_000", 2.5, 0.8)])
    with postgres.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT median_visible_vehicle_count, detection_persistence_ratio FROM run_segment_metadata "
            "WHERE run_id=%s AND segment_id=%s", (enrichment_run, "enrich_seg_000"),
        )
        row = cur.fetchone()
    assert row == (2.5, 0.8)


def test_postgres_enrichment_write_null_means_not_measured(enrichment_run):
    postgres.write_run_detector_enrichment(enrichment_run, DATASET_ID, [("enrich_seg_000", None, None)])
    with postgres.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT median_visible_vehicle_count, detection_persistence_ratio FROM run_segment_metadata "
            "WHERE run_id=%s AND segment_id=%s", (enrichment_run, "enrich_seg_000"),
        )
        row = cur.fetchone()
    assert row == (None, None)


def test_clickhouse_enrichment_write_sets_canonical_columns(enrichment_run):
    clickhouse.write_run_detector_enrichment(enrichment_run, DATASET_ID, DIMENSION, [("enrich_seg_000", 3.0, 0.6)])
    result = clickhouse.client().query(
        f"SELECT median_visible_vehicle_count, detection_persistence_ratio FROM seg_ch_{DIMENSION}_runs "
        "WHERE run_id={run_id:UUID} AND segment_id={segment_id:String}",
        parameters={"run_id": enrichment_run, "segment_id": "enrich_seg_000"},
    )
    # ClickHouse stores these as Float32, so 0.6 round-trips as the nearest float32
    # value, not the Python float64 literal -- expected storage precision, not a bug.
    assert result.result_rows[0] == approx((3.0, 0.6))
