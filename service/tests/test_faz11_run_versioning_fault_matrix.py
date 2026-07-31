from __future__ import annotations

import numpy as np
import pytest
import yaml
from PIL import Image

from app.config import Settings
from app.db.ingest_runs import RunSpec
from app.ingestion.generic_loader import WindowRecord
from app.ingestion.ingest import GenericIngestor
from app.ingestion.manifest import load_manifest


def _manifest(tmp_path):
    payload = {
        "schema_version": 1, "dataset_id": "fixture", "display_name": "Fixture",
        "source": {"videos_glob": "videos/*.mp4", "video_id_from": "filename_stem"},
        "pairing": {"strategy": "filename_stem", "telemetry_glob": None, "telemetry_id_from": "filename_stem"},
        "time_alignment": {"video_clock": "pts", "telemetry_clock": "relative_s", "offset_s": 0, "max_gap_s": 1},
        "window": {"size_s": 4, "stride_s": 4, "frames_per_item": 1, "partial_window_policy": "drop_partial"},
        "telemetry": {"format": "generic_csv", "timestamp_column": "timestamp", "fields": {}, "extra": {}},
        "media": {"enabled": True, "clip_cache": True}, "policy": {"fail_on_video_error": True},
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return load_manifest(path)


def _run(**overrides):
    base = dict(
        run_id="00000000-0000-0000-0000-000000000010", dataset_id="fixture",
        dataset_version="v1", vector_provenance="real", model_id="model",
        model_revision="revision", source_commit="commit", enabled_backends=("clickhouse",),
        enabled_dimensions=(512,), manifest_hash="manifest", expected_segments=None,
    )
    base.update(overrides)
    return RunSpec(**base)


def _records(count, *, chunk_index=0):
    return [
        WindowRecord(
            dataset_id="fixture", video_id="video", segment_id=f"fixture:video:{index}",
            chunk_index=chunk_index, t_start=float(index * 4), t_end=float(index * 4 + 4),
            frames=[Image.new("RGB", (2, 2), color=(index % 255, 0, 0))],
            metadata={"source_path": "/data/videos/video.mp4", "video_duration_s": float(count * 4), "fps": 1.0, "codec": "fixture"},
            telemetry={"altitude_m": float(index), "person_count": index}, extra={},
        )
        for index in range(count)
    ]


class _Store:
    def __init__(self):
        self.active = "old-run"
        self.ledger = {}
        self.metadata = set()
        self.run_status = None

    def active_run_id(self, dataset_id):
        return self.active

    def set_run_status(self, run_id, status, *, error_summary=None):
        self.run_status = status

    def set_chunk_status(self, spec, status, backend_status):
        self.ledger[(spec.video_id, spec.chunk_index)] = {
            "video_id": spec.video_id, "chunk_index": spec.chunk_index, "status": status,
            "expected_segments": spec.expected_segments, "backend_status": backend_status,
        }

    def chunks(self, run_id):
        return list(self.ledger.values())

    def metadata_count(self, run_id):
        return len(self.metadata)

    def duplicate_count(self, run_id):
        return 0

    def update_expected_segments(self, run_id, value):
        pass

    def activate(self, spec, rows):
        self.active = spec.run_id
        self.run_status = "completed"


class _Backend:
    def __init__(self, name, *, fail_on_write_call=None):
        self.name = name
        self.rows = {}
        self.write_calls = 0
        self.fail_on_write_call = fail_on_write_call

    def delete_inactive_chunk(self, run_id, dataset_id, chunk_index, dimension):
        keys = [key for key in self.rows if key[:3] == (run_id, chunk_index, dimension)]
        for key in keys:
            del self.rows[key]
        return len(keys)

    def write_chunk(self, run_id, dataset_id, dimension, chunk_index, rows):
        self.write_calls += 1
        if self.fail_on_write_call is not None and self.write_calls == self.fail_on_write_call:
            raise RuntimeError(f"injected failure in backend {self.name}")
        for row in rows:
            self.rows[(run_id, chunk_index, dimension, row["segment_id"])] = row
        return len(rows)

    def count_run(self, dataset_id, run_id, dimension):
        return sum(key[0] == run_id and key[2] == dimension for key in self.rows)


def _identity_embed(batch):
    vectors = np.ones((len(batch), 2048), dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _patch_metadata(monkeypatch, store, *, fail=False):
    def delete_metadata(run_id, dataset_id, video_id, chunk_index):
        doomed = {value for value in store.metadata if value.startswith(f"{chunk_index}:")}
        store.metadata -= doomed
        return len(doomed)

    def write_metadata(run_id, dataset_id, chunk_index, videos, segments, metadata, telemetry):
        if fail:
            raise RuntimeError("injected metadata write failure")
        store.metadata.update(f"{chunk_index}:{row[0]}" for row in segments)
        return len(segments)

    monkeypatch.setattr("app.ingestion.ingest.postgres.delete_inactive_metadata_chunk", delete_metadata)
    monkeypatch.setattr("app.ingestion.ingest.postgres.write_run_metadata_chunk", write_metadata)


# 1/3: failure before any write (metadata or backend) has happened for the chunk.
def test_fault_1_and_3_failure_before_any_write_leaves_chunk_uncommitted(tmp_path, monkeypatch):
    manifest, store = _manifest(tmp_path), _Store()
    _patch_metadata(monkeypatch, store, fail=True)
    backend = _Backend("clickhouse")
    result = GenericIngestor(
        manifest, tmp_path, _run(), store=store, backends={"clickhouse": backend},
        embed_videos=_identity_embed, records=iter(_records(2)), report_root=tmp_path / "report",
    ).run()
    assert result["status"] == "failed"
    assert store.ledger[("video", 0)]["status"] == "failed"
    assert backend.write_calls == 0
    assert store.active == "old-run"


# 2/4: metadata write succeeds, first backend write then fails.
def test_fault_2_and_4_metadata_committed_but_first_backend_write_fails(tmp_path, monkeypatch):
    manifest, store = _manifest(tmp_path), _Store()
    _patch_metadata(monkeypatch, store)
    backend = _Backend("clickhouse", fail_on_write_call=1)
    result = GenericIngestor(
        manifest, tmp_path, _run(), store=store, backends={"clickhouse": backend},
        embed_videos=_identity_embed, records=iter(_records(2)), report_root=tmp_path / "report",
    ).run()
    assert result["status"] == "failed"
    assert store.ledger[("video", 0)]["status"] == "failed"
    assert store.active == "old-run"
    # metadata rows were written by _flush_pending before the backend call failed,
    # but the chunk is not committed - resume must clean this up, not leave it live.
    assert len(store.metadata) == 2


# 4/5: two backends enabled, first succeeds, second (a later dimension/backend
# combination within the same chunk) fails.
def test_fault_4_and_5_first_backend_succeeds_second_backend_fails(tmp_path, monkeypatch):
    manifest, store = _manifest(tmp_path), _Store()
    _patch_metadata(monkeypatch, store)
    backend_a = _Backend("clickhouse")
    backend_b = _Backend("qdrant", fail_on_write_call=1)
    run = _run(enabled_backends=("clickhouse", "qdrant"))
    result = GenericIngestor(
        manifest, tmp_path, run, store=store, backends={"clickhouse": backend_a, "qdrant": backend_b},
        embed_videos=_identity_embed, records=iter(_records(2)), report_root=tmp_path / "report",
    ).run()
    assert result["status"] == "failed"
    assert store.ledger[("video", 0)]["status"] == "failed"
    # backend_a already has rows for this (not-yet-committed) chunk; resume's
    # begin_chunk cleanup (delete_inactive_chunk) is responsible for wiping
    # them before redoing the chunk, not this run() call.
    assert backend_a.count_run("fixture", run.run_id, 512) == 2
    assert backend_b.count_run("fixture", run.run_id, 512) == 0
    assert store.active == "old-run"


# 5: two dimensions enabled, first dimension's writes (across all backends)
# succeed, second dimension's write fails.
def test_fault_5_first_dimension_succeeds_second_dimension_fails(tmp_path, monkeypatch):
    manifest, store = _manifest(tmp_path), _Store()
    _patch_metadata(monkeypatch, store)
    # write_chunk is called once per (dimension, backend) pair per flush; with a
    # single clickhouse backend and two dimensions, the second call is the
    # second dimension's write.
    backend = _Backend("clickhouse", fail_on_write_call=2)
    run = _run(enabled_dimensions=(512, 256))
    result = GenericIngestor(
        manifest, tmp_path, run, store=store, backends={"clickhouse": backend},
        embed_videos=_identity_embed, records=iter(_records(2)), report_root=tmp_path / "report",
    ).run()
    assert result["status"] == "failed"
    assert backend.count_run("fixture", run.run_id, 512) == 2
    assert backend.count_run("fixture", run.run_id, 256) == 0
    assert store.active == "old-run"


# 6: failure in the middle of a chunk's DB write batching (some flushes already
# committed to backends, a later flush within the same still-uncommitted chunk fails).
def test_fault_6_failure_mid_db_write_batch(tmp_path, monkeypatch):
    manifest, store = _manifest(tmp_path), _Store()
    _patch_metadata(monkeypatch, store)
    monkeypatch.setattr(
        "app.ingestion.ingest.settings",
        Settings.from_env({
            "DECODE_PREFETCH_WINDOWS": "8", "EMBED_BATCH_SIZE": "2", "DB_WRITE_BATCH_SIZE": "2",
            "ENABLED_VECTOR_BACKENDS": "clickhouse", "ENABLED_DIMENSIONS": "512",
        }),
    )
    backend = _Backend("clickhouse", fail_on_write_call=2)  # second DB_WRITE_BATCH_SIZE flush fails
    result = GenericIngestor(
        manifest, tmp_path, _run(), store=store, backends={"clickhouse": backend},
        embed_videos=_identity_embed, records=iter(_records(4)), report_root=tmp_path / "report",
    ).run()
    assert result["status"] == "failed"
    assert store.ledger[("video", 0)]["status"] == "failed"
    assert backend.count_run("fixture", _run().run_id, 512) == 2  # first flush's rows remain until resume cleans up
    assert store.active == "old-run"


# 9/10: metadata + all backends/dimensions match expected, but the run store's
# activate() itself raises (simulating a failed activation transaction).
def test_fault_9_and_10_activation_failure_does_not_flip_active_pointer(tmp_path, monkeypatch):
    from app.db.ingest_runs import ChunkSpec, RunCoordinator

    class _FailingActivateStore(_Store):
        def activate(self, spec, rows):
            raise RuntimeError("injected activation transaction failure")

    store = _FailingActivateStore()
    store.ledger[("video", 0)] = {"status": "committed", "expected_segments": 2}
    store.metadata = {"a", "b"}
    backend = _Backend("clickhouse")
    backend.rows[("00000000-0000-0000-0000-000000000010", 0, 512, "a")] = {}
    backend.rows[("00000000-0000-0000-0000-000000000010", 0, 512, "b")] = {}
    run = _run(expected_segments=2)

    with pytest.raises(RuntimeError, match="injected activation"):
        RunCoordinator(store, {"clickhouse": backend}).finalize_run(run)
    # The old active run must still be in place: activate() raised before
    # doing anything the caller could observe as a successful flip, and a real
    # PostgreSQL connection.close() without commit() rolls back whatever the
    # UPDATE/INSERT pair inside activate() already executed.
    assert store.active == "old-run"
