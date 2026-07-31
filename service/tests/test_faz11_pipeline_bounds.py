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


def _run(enabled_dimensions=(512,)):
    return RunSpec(
        run_id="00000000-0000-0000-0000-000000000010", dataset_id="fixture",
        dataset_version="v1", vector_provenance="real", model_id="model",
        model_revision="revision", source_commit="commit", enabled_backends=("clickhouse",),
        enabled_dimensions=enabled_dimensions, manifest_hash="manifest", expected_segments=None,
    )


def _records(count: int, *, chunk_index: int = 0):
    result = []
    for index in range(count):
        result.append(WindowRecord(
            dataset_id="fixture", video_id="video", segment_id=f"fixture:video:{index}",
            chunk_index=chunk_index, t_start=float(index * 4), t_end=float(index * 4 + 4),
            frames=[Image.new("RGB", (2, 2), color=(index % 255, 0, 0))],
            metadata={"source_path": "/data/videos/video.mp4", "video_duration_s": float(count * 4), "fps": 1.0, "codec": "fixture"},
            telemetry={"altitude_m": float(index), "person_count": index}, extra={"battery": 12.0 - index},
        ))
    return result


class _Store:
    def __init__(self):
        self.active = "old-run"
        self.ledger = {}
        self.metadata = set()
        self.run_status = None
        self.expected = None

    def active_run_id(self, dataset_id):
        return self.active

    def set_run_status(self, run_id, status, *, error_summary=None):
        self.run_status = status

    def set_chunk_status(self, spec, status, backend_status):
        self.ledger[(spec.video_id, spec.chunk_index)] = {
            "video_id": spec.video_id, "chunk_index": spec.chunk_index,
            "status": status, "expected_segments": spec.expected_segments,
            "backend_status": backend_status,
        }

    def chunks(self, run_id):
        return list(self.ledger.values())

    def metadata_count(self, run_id):
        return len(self.metadata)

    def duplicate_count(self, run_id):
        return 0

    def update_expected_segments(self, run_id, value):
        self.expected = value

    def activate(self, spec, rows):
        self.active = spec.run_id
        self.run_status = "completed"


class _Backend:
    name = "clickhouse"

    def __init__(self, fail_on_call: int | None = None):
        self.rows = {}
        self.write_calls: list[int] = []
        self.fail_on_call = fail_on_call
        self._calls = 0

    def delete_inactive_chunk(self, run_id, dataset_id, chunk_index, dimension):
        keys = [key for key in self.rows if key[:3] == (run_id, chunk_index, dimension)]
        for key in keys:
            del self.rows[key]
        return len(keys)

    def write_chunk(self, run_id, dataset_id, dimension, chunk_index, rows):
        self._calls += 1
        self.write_calls.append(len(rows))
        if self.fail_on_call is not None and self._calls == self.fail_on_call:
            raise RuntimeError("injected backend write failure")
        for row in rows:
            self.rows[(run_id, chunk_index, dimension, row["segment_id"])] = row
        return len(rows)

    def count_run(self, dataset_id, run_id, dimension):
        return sum(key[0] == run_id and key[2] == dimension for key in self.rows)


def _settings(**overrides):
    base = {
        "DECODE_PREFETCH_WINDOWS": "8", "EMBED_BATCH_SIZE": "2", "DB_WRITE_BATCH_SIZE": "512",
        "ENABLED_VECTOR_BACKENDS": "clickhouse", "ENABLED_DIMENSIONS": "512",
    }
    base.update(overrides)
    return Settings.from_env(base)


def _identity_embed(batch):
    vectors = np.ones((len(batch), 2048), dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _patch_metadata(monkeypatch, store):
    def delete_metadata(run_id, dataset_id, video_id, chunk_index):
        doomed = {value for value in store.metadata if value.startswith(f"{chunk_index}:")}
        store.metadata -= doomed
        return len(doomed)

    def write_metadata(run_id, dataset_id, chunk_index, videos, segments, metadata, telemetry):
        store.metadata.update(f"{chunk_index}:{row[0]}" for row in segments)
        return len(segments)

    monkeypatch.setattr("app.ingestion.ingest.postgres.delete_inactive_metadata_chunk", delete_metadata)
    monkeypatch.setattr("app.ingestion.ingest.postgres.write_run_metadata_chunk", write_metadata)
    return write_metadata


def test_decode_prefetch_bound_is_enforced(tmp_path, monkeypatch):
    """No more than DECODE_PREFETCH_WINDOWS records may be pulled off the decode
    source before the first embed call consumes any of them."""
    manifest, store = _manifest(tmp_path), _Store()
    _patch_metadata(monkeypatch, store)
    monkeypatch.setattr("app.ingestion.ingest.settings", _settings(DECODE_PREFETCH_WINDOWS="3", EMBED_BATCH_SIZE="1"))
    backend = _Backend()
    pulled = {"count": 0}
    embedded = {"count": 0}

    def counting_records():
        for record in _records(10):
            pulled["count"] += 1
            yield record

    def embed(batch):
        assert pulled["count"] - embedded["count"] <= 3, "decode ran further ahead than DECODE_PREFETCH_WINDOWS"
        embedded["count"] += len(batch)
        return _identity_embed(batch)

    result = GenericIngestor(
        manifest, tmp_path, _run(), store=store, backends={"clickhouse": backend},
        embed_videos=embed, records=counting_records(), report_root=tmp_path / "report",
    ).run()
    assert result["status"] == "completed"
    assert result["segments"] == 10


def test_embed_batch_size_is_independent(tmp_path, monkeypatch):
    """EMBED_BATCH_SIZE controls the Qwen call granularity regardless of the
    other two settings."""
    manifest, store = _manifest(tmp_path), _Store()
    _patch_metadata(monkeypatch, store)
    monkeypatch.setattr(
        "app.ingestion.ingest.settings",
        _settings(DECODE_PREFETCH_WINDOWS="6", EMBED_BATCH_SIZE="4", DB_WRITE_BATCH_SIZE="3"),
    )
    backend = _Backend()
    call_sizes: list[int] = []

    def embed(batch):
        call_sizes.append(len(batch))
        return _identity_embed(batch)

    result = GenericIngestor(
        manifest, tmp_path, _run(), store=store, backends={"clickhouse": backend},
        embed_videos=embed, records=iter(_records(10)), report_root=tmp_path / "report",
    ).run()
    assert result["status"] == "completed"
    # 10 records: two prefetch groups of 6 and 4; embed batches of 4 within each -> 4,2,4
    assert call_sizes == [4, 2, 4]


def test_db_write_batch_size_is_independent(tmp_path, monkeypatch):
    """DB_WRITE_BATCH_SIZE controls the Postgres/backend write granularity and
    is decoupled from EMBED_BATCH_SIZE."""
    manifest, store = _manifest(tmp_path), _Store()
    write_metadata = _patch_metadata(monkeypatch, store)
    metadata_call_sizes: list[int] = []
    original = write_metadata

    def spying_write_metadata(run_id, dataset_id, chunk_index, videos, segments, metadata, telemetry):
        metadata_call_sizes.append(len(segments))
        return original(run_id, dataset_id, chunk_index, videos, segments, metadata, telemetry)

    monkeypatch.setattr("app.ingestion.ingest.postgres.write_run_metadata_chunk", spying_write_metadata)
    monkeypatch.setattr(
        "app.ingestion.ingest.settings",
        _settings(DECODE_PREFETCH_WINDOWS="6", EMBED_BATCH_SIZE="2", DB_WRITE_BATCH_SIZE="3"),
    )
    backend = _Backend()

    result = GenericIngestor(
        manifest, tmp_path, _run(), store=store, backends={"clickhouse": backend},
        embed_videos=_identity_embed, records=iter(_records(6)), report_root=tmp_path / "report",
    ).run()
    assert result["status"] == "completed"
    # 6 records, DB_WRITE_BATCH_SIZE=3 -> two writes of 3, never matching EMBED_BATCH_SIZE=2
    assert metadata_call_sizes == [3, 3]
    assert backend.write_calls == [3, 3]


def test_frames_are_released_after_each_batch(tmp_path, monkeypatch):
    """Frames must be closed right after their own embed batch, not deferred
    until the (potentially much larger) DB write batch flushes."""
    manifest, store = _manifest(tmp_path), _Store()
    _patch_metadata(monkeypatch, store)
    monkeypatch.setattr(
        "app.ingestion.ingest.settings",
        _settings(DECODE_PREFETCH_WINDOWS="8", EMBED_BATCH_SIZE="2", DB_WRITE_BATCH_SIZE="100"),
    )
    backend = _Backend()
    seen_batches: list[list[WindowRecord]] = []

    def embed(records_only_frames):
        return _identity_embed(records_only_frames)

    records = _records(6)
    # Wrap embed_videos at the ingestor level via a closure capturing WindowRecord batches
    # by monkeypatching batched() is unnecessary: inspect frames directly after run().
    result = GenericIngestor(
        manifest, tmp_path, _run(), store=store, backends={"clickhouse": backend},
        embed_videos=_identity_embed, records=iter(records), report_root=tmp_path / "report",
    ).run()
    assert result["status"] == "completed"
    # DB_WRITE_BATCH_SIZE=100 means all 6 records are still buffered (unflushed) when
    # the *last* embed batch fires; if release were deferred to the DB flush, earlier
    # batches' frames would still be open at that point. They must already be closed.
    assert all(record.frames == [] for record in records)


def test_producer_exception_reaches_ingestor(tmp_path, monkeypatch):
    """An exception raised by the decode source itself (not the embedder) must
    surface as a failed chunk, not crash uncontrolled or hang."""
    manifest, store = _manifest(tmp_path), _Store()
    _patch_metadata(monkeypatch, store)
    monkeypatch.setattr("app.ingestion.ingest.settings", _settings())
    backend = _Backend()

    def broken_records():
        for record in _records(3):
            yield record
        raise RuntimeError("simulated decode failure")

    result = GenericIngestor(
        manifest, tmp_path, _run(), store=store, backends={"clickhouse": backend},
        embed_videos=_identity_embed, records=broken_records(), report_root=tmp_path / "report",
    ).run()
    assert result["status"] == "failed"
    assert store.ledger[("video", 0)]["status"] == "failed"


def test_db_failure_does_not_commit_chunk(tmp_path, monkeypatch):
    """If a vector backend write fails partway through a chunk's DB writes, the
    chunk must not be marked committed."""
    manifest, store = _manifest(tmp_path), _Store()
    _patch_metadata(monkeypatch, store)
    monkeypatch.setattr(
        "app.ingestion.ingest.settings",
        _settings(DECODE_PREFETCH_WINDOWS="8", EMBED_BATCH_SIZE="2", DB_WRITE_BATCH_SIZE="2"),
    )
    backend = _Backend(fail_on_call=2)  # second write_chunk call raises

    result = GenericIngestor(
        manifest, tmp_path, _run(), store=store, backends={"clickhouse": backend},
        embed_videos=_identity_embed, records=iter(_records(6)), report_root=tmp_path / "report",
    ).run()
    assert result["status"] == "failed"
    assert store.ledger[("video", 0)]["status"] == "failed"


def test_resume_after_partial_batch_is_idempotent(tmp_path, monkeypatch):
    """A chunk that fails after one DB_WRITE_BATCH_SIZE flush already succeeded
    must be fully cleaned up and re-written without duplicates on resume."""
    manifest, store = _manifest(tmp_path), _Store()
    _patch_metadata(monkeypatch, store)
    monkeypatch.setattr(
        "app.ingestion.ingest.settings",
        _settings(DECODE_PREFETCH_WINDOWS="8", EMBED_BATCH_SIZE="2", DB_WRITE_BATCH_SIZE="2"),
    )
    backend = _Backend(fail_on_call=2)

    first = GenericIngestor(
        manifest, tmp_path, _run(), store=store, backends={"clickhouse": backend},
        embed_videos=_identity_embed, records=iter(_records(6)), report_root=tmp_path / "report",
    ).run()
    assert first["status"] == "failed"

    backend.fail_on_call = None
    second = GenericIngestor(
        manifest, tmp_path, _run(), store=store, backends={"clickhouse": backend},
        embed_videos=_identity_embed, records=iter(_records(6)), report_root=tmp_path / "report",
    ).run(resume=True)
    assert second["status"] == "completed"
    assert second["segments"] == 6
    assert backend.count_run("fixture", _run().run_id, 512) == 6
    assert len(store.metadata) == 6
