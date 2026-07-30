from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import yaml
from PIL import Image

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


def _run():
    return RunSpec(
        run_id="00000000-0000-0000-0000-000000000010", dataset_id="fixture",
        dataset_version="v1", vector_provenance="real", model_id="model",
        model_revision="revision", source_commit="commit", enabled_backends=("clickhouse",),
        enabled_dimensions=(512,), manifest_hash="manifest", expected_segments=None,
    )


def _records():
    result = []
    for index in range(4):
        chunk = index // 2
        result.append(WindowRecord(
            dataset_id="fixture", video_id="video", segment_id=f"fixture:video:{index}",
            chunk_index=chunk, t_start=float(index * 4), t_end=float(index * 4 + 4),
            frames=[Image.new("RGB", (2, 2), color=(index, 0, 0))],
            metadata={"source_path": "/data/videos/video.mp4", "video_duration_s": 16.0, "fps": 1.0, "codec": "fixture"},
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

    def __init__(self):
        self.rows = {}

    def delete_inactive_chunk(self, run_id, dataset_id, chunk_index, dimension):
        keys = [key for key in self.rows if key[:3] == (run_id, chunk_index, dimension)]
        for key in keys:
            del self.rows[key]
        return len(keys)

    def write_chunk(self, run_id, dataset_id, dimension, chunk_index, rows):
        for row in rows:
            self.rows[(run_id, chunk_index, dimension, row["segment_id"])] = row
        return len(rows)

    def count_run(self, dataset_id, run_id, dimension):
        return sum(key[0] == run_id and key[2] == dimension for key in self.rows)


def test_crash_then_resume_skips_committed_chunk_without_duplicates(tmp_path, monkeypatch):
    manifest, run, store, backend = _manifest(tmp_path), _run(), _Store(), _Backend()

    def delete_metadata(run_id, dataset_id, video_id, chunk_index):
        doomed = {value for value in store.metadata if value.startswith(f"{chunk_index}:")}
        store.metadata -= doomed
        return len(doomed)

    def write_metadata(run_id, dataset_id, chunk_index, videos, segments, metadata, telemetry):
        store.metadata.update(f"{chunk_index}:{row[0]}" for row in segments)
        return len(segments)

    monkeypatch.setattr("app.ingestion.ingest.postgres.delete_inactive_metadata_chunk", delete_metadata)
    monkeypatch.setattr("app.ingestion.ingest.postgres.write_run_metadata_chunk", write_metadata)
    calls = 0

    def crash_second_batch(videos):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected crash")
        base = np.ones((len(videos), 2048), dtype=np.float32)
        return base / np.linalg.norm(base, axis=1, keepdims=True)

    first = GenericIngestor(
        manifest, tmp_path, run, store=store, backends={"clickhouse": backend},
        embed_videos=crash_second_batch, records=_records(), report_root=tmp_path / "report",
    ).run()
    assert first["status"] == "failed"
    assert store.ledger[("video", 0)]["status"] == "committed"
    assert backend.count_run("fixture", run.run_id, 512) == 2

    def embed(videos):
        base = np.ones((len(videos), 2048), dtype=np.float32)
        return base / np.linalg.norm(base, axis=1, keepdims=True)

    resumed_records = _records()
    second = GenericIngestor(
        manifest, tmp_path, run, store=store, backends={"clickhouse": backend},
        embed_videos=embed, records=resumed_records, report_root=tmp_path / "report",
    ).run(resume=True)
    assert second["status"] == "completed"
    assert second["preexisting_committed_chunks"] == 1
    assert second["segments"] == 4
    assert backend.count_run("fixture", run.run_id, 512) == 4
    assert len(store.metadata) == 4
    assert all(not record.frames for record in resumed_records if record.chunk_index == 1)
    report = json.loads((tmp_path / "report" / "report.json").read_text(encoding="utf-8"))
    assert report["finalize"]["active_changed"] is True


def test_vector_projection_uses_only_enabled_dimension(tmp_path, monkeypatch):
    manifest, store, backend = _manifest(tmp_path), _Store(), _Backend()
    run = replace(_run(), enabled_dimensions=(256,))
    monkeypatch.setattr("app.ingestion.ingest.postgres.delete_inactive_metadata_chunk", lambda *args: 0)

    def write_metadata(run_id, dataset_id, chunk_index, videos, segments, metadata, telemetry):
        store.metadata.update(f"{chunk_index}:{row[0]}" for row in segments)
        return len(segments)

    monkeypatch.setattr("app.ingestion.ingest.postgres.write_run_metadata_chunk", write_metadata)
    records = _records()[:2]
    result = GenericIngestor(
        manifest, tmp_path, run, store=store, backends={"clickhouse": backend},
        embed_videos=lambda videos: np.eye(len(videos), 2048, dtype=np.float32), records=records,
        report_root=tmp_path / "report",
    ).run()
    assert result["status"] == "completed"
    assert {key[2] for key in backend.rows} == {256}
    assert all(len(row["embedding"]) == 256 for row in backend.rows.values())
