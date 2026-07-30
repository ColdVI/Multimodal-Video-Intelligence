from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.db import postgres, qdrant
from app.db.ingest_runs import ChunkSpec, RunCoordinator, RunSpec, legacy_run_id
from app.ingestion.gc_runs import select_gc_candidates


def _run(expected=3):
    return RunSpec(
        run_id="00000000-0000-0000-0000-000000000001", dataset_id="dataset",
        dataset_version="v1", vector_provenance="real", model_id="model",
        model_revision="revision", source_commit="commit", enabled_backends=("clickhouse",),
        enabled_dimensions=(512,), manifest_hash="manifest", expected_segments=expected,
    )


def _chunk(expected=3):
    return ChunkSpec(
        run_id=_run().run_id, dataset_id="dataset", video_id="video", video_path="videos/video.mp4",
        chunk_index=0, chunk_start_s=0.0, chunk_end_s=12.0, expected_segments=expected,
    )


class _Store:
    def __init__(self):
        self.active = "old-run"
        self.statuses = []
        self.chunk_statuses = []
        self.chunk_rows = [{"status": "committed", "expected_segments": 3}]
        self.metadata_rows = 3
        self.duplicates = 0

    def active_run_id(self, dataset_id):
        return self.active

    def set_run_status(self, run_id, status, *, error_summary=None):
        self.statuses.append((status, error_summary))

    def set_chunk_status(self, spec, status, backend_status):
        self.chunk_statuses.append((status, backend_status))

    def chunks(self, run_id):
        return self.chunk_rows

    def metadata_count(self, run_id):
        return self.metadata_rows

    def duplicate_count(self, run_id):
        return self.duplicates

    def activate(self, spec, rows_per_backend):
        self.active = spec.run_id
        self.statuses.append(("completed", rows_per_backend))


class _Backend:
    name = "clickhouse"

    def __init__(self):
        self.deleted = []
        self.rows = 3

    def delete_inactive_chunk(self, run_id, dataset_id, chunk_index, dimension):
        self.deleted.append((run_id, chunk_index, dimension))
        return 2

    def count_run(self, dataset_id, run_id, dimension):
        return self.rows


def test_retry_cleans_only_same_inactive_run_chunk_before_writing():
    store, backend = _Store(), _Backend()
    statuses = RunCoordinator(store, {"clickhouse": backend}).begin_chunk(_chunk(), _run())
    assert backend.deleted == [(_run().run_id, 0, 512)]
    assert statuses["clickhouse:512"]["deleted_before_retry"] == 2
    assert store.chunk_statuses[-1][0] == "writing"


def test_retry_of_active_run_is_forbidden_before_any_delete():
    store, backend = _Store(), _Backend()
    store.active = _run().run_id
    with pytest.raises(ValueError, match="active run"):
        RunCoordinator(store, {"clickhouse": backend}).begin_chunk(_chunk(), _run())
    assert backend.deleted == []


def test_chunk_count_mismatch_is_failed_and_not_silently_committed():
    store, backend = _Store(), _Backend()
    with pytest.raises(ValueError, match="row-count mismatch"):
        RunCoordinator(store, {"clickhouse": backend}).commit_chunk(
            _chunk(), _run(), {"clickhouse:512": 2}, metadata_rows=3,
        )
    assert store.chunk_statuses[-1][0] == "failed"


def test_finalize_failure_preserves_old_active_run():
    store, backend = _Store(), _Backend()
    backend.rows = 2
    result = RunCoordinator(store, {"clickhouse": backend}).finalize_run(_run())
    assert result["status"] == "failed" and result["active_changed"] is False
    assert store.active == "old-run"
    assert store.statuses[-1][0] == "failed"


def test_finalize_success_activates_only_after_all_counts_match():
    store, backend = _Store(), _Backend()
    result = RunCoordinator(store, {"clickhouse": backend}).finalize_run(_run())
    assert result["status"] == "completed" and result["active_changed"] is True
    assert store.active == _run().run_id


def test_run_ids_and_qdrant_ids_are_deterministic_and_run_scoped():
    assert legacy_run_id("d", "h") == legacy_run_id("d", "h")
    assert legacy_run_id("d", "h") != legacy_run_id("d", "other")
    assert qdrant.point_id("segment", "run-a") != qdrant.point_id("segment", "run-b")
    assert qdrant.point_id("segment", "run-a") == qdrant.point_id("segment", "run-a")


def test_schema_contains_all_run_scoped_control_and_metadata_tables():
    for table in (
        "ingest_runs", "dataset_active_runs", "ingest_chunks", "run_videos", "run_segments",
        "run_segment_metadata", "run_segment_telemetry", "run_retrieval_groundtruth",
        "telemetry_field_registry",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in postgres.SCHEMA_SQL


def test_gc_never_selects_active_running_or_previous_completed():
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=3)
    runs = [
        {"run_id": "active", "dataset_id": "d", "status": "completed", "is_active": True, "started_at": old, "finished_at": old},
        {"run_id": "previous", "dataset_id": "d", "status": "completed", "is_active": False, "started_at": old, "finished_at": old + timedelta(hours=1)},
        {"run_id": "older", "dataset_id": "d", "status": "completed", "is_active": False, "started_at": old, "finished_at": old},
        {"run_id": "running", "dataset_id": "d", "status": "ingesting", "is_active": False, "started_at": old, "finished_at": None},
        {"run_id": "failed", "dataset_id": "d", "status": "failed", "is_active": False, "started_at": old, "finished_at": old},
        {"run_id": "recent-failed", "dataset_id": "d", "status": "failed", "is_active": False, "started_at": now, "finished_at": now},
    ]
    selected = select_gc_candidates(runs, now=now, retain_previous_completed=1, min_age_hours=24)
    assert {row["run_id"] for row in selected} == {"older", "failed"}
