from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from app import preflight as preflight_module
from app.config import Settings
from app.ingestion.manifest import load_manifest
from app.preflight import run_data_preflight


def _payload():
    return {
        "schema_version": 1, "dataset_id": "fixture", "display_name": "Fixture",
        "source": {"videos_glob": "videos/*.mp4", "video_id_from": "filename_stem"},
        "pairing": {
            "strategy": "filename_stem", "telemetry_glob": "telemetry/*.csv",
            "telemetry_id_from": "filename_stem", "manifest_csv": None,
        },
        "time_alignment": {
            "video_clock": "pts", "telemetry_clock": "relative_s", "offset_s": 0.0, "max_gap_s": 1.0,
        },
        "window": {"size_s": 8.0, "stride_s": 4.0, "frames_per_item": 8, "partial_window_policy": "drop_partial"},
        "telemetry": {
            "format": "generic_csv", "timestamp_column": "timestamp",
            "fields": {"altitude_m": {"source": "alt", "unit": "m", "reference": "AGL", "type": "continuous"}},
            "extra": {},
        },
        "media": {"enabled": True, "clip_cache": True},
    }


def _manifest(tmp_path: Path):
    (tmp_path / "videos").mkdir()
    (tmp_path / "telemetry").mkdir()
    (tmp_path / "videos" / "flight.mp4").write_bytes(b"fixture-not-decoded")
    (tmp_path / "telemetry" / "flight.csv").write_text(
        "timestamp,alt\n0,10\n4,11\n", encoding="utf-8",
    )
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(_payload(), sort_keys=False), encoding="utf-8")
    return load_manifest(path)


def _settings(tmp_path: Path, **overrides):
    values = {
        "ARTIFACTS_ROOT": str(tmp_path / "artifacts"), "ENABLED_VECTOR_BACKENDS": "clickhouse",
        "ENABLED_DIMENSIONS": "512", "MODEL_BUNDLE_ROOT": str(tmp_path / "no-such-bundle"),
    }
    values.update(overrides)
    (tmp_path / "artifacts").mkdir(exist_ok=True)
    return Settings.from_env(values)


@dataclass
class _Probe:
    duration_s: float = 16.0
    fps: float = 25.0
    width: int = 640
    height: int = 480
    creation_time: datetime | None = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(root: Path) -> dict[str, tuple[int, int, str]]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns, _sha256(path))
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def _fixture_bundle(root: Path) -> None:
    source_dir, model_dir = root / "source", root / "model"
    source_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    (source_dir / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (model_dir / "weights.bin").write_bytes(b"weights")

    def _inventory(subroot: Path):
        files, total = [], 0
        for path in sorted(subroot.rglob("*")):
            if path.is_file():
                data = path.read_bytes()
                total += len(data)
                files.append({"path": path.relative_to(subroot).as_posix(), "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        return files, total

    source_files, source_size = _inventory(source_dir)
    model_files, model_size = _inventory(model_dir)
    (root / "source_manifest.json").write_text(json.dumps({
        "schema_version": 1, "kind": "source", "source_repo": "x", "source_commit": "c",
        "generated_at_utc": "t", "total_size_bytes": source_size, "files": source_files,
    }, sort_keys=True), encoding="utf-8")
    (root / "model_manifest.json").write_text(json.dumps({
        "schema_version": 1, "kind": "model", "model_id": "m", "model_revision": "r",
        "generated_at_utc": "t", "total_size_bytes": model_size, "files": model_files,
    }, sort_keys=True), encoding="utf-8")
    (root / "bundle_manifest.json").write_text(json.dumps({
        "schema_version": 1, "generated_at_utc": "t", "model_id": "m", "model_revision": "r",
        "source_repo": "x", "source_commit": "c",
        "source_manifest_sha256": _sha256(root / "source_manifest.json"),
        "model_manifest_sha256": _sha256(root / "model_manifest.json"),
        "total_size_bytes": source_size + model_size,
    }, sort_keys=True), encoding="utf-8")


def test_preflight_does_not_modify_dataset_files(tmp_path):
    manifest = _manifest(tmp_path)
    dataset_root = tmp_path
    before = _snapshot(dataset_root / "videos") | _snapshot(dataset_root / "telemetry")
    run_data_preflight(
        manifest.path, data_root=dataset_root, configured=_settings(tmp_path), probe_fn=lambda _: _Probe(),
    )
    after = _snapshot(dataset_root / "videos") | _snapshot(dataset_root / "telemetry")
    assert before == after


def test_preflight_does_not_modify_model_bundle(tmp_path):
    manifest = _manifest(tmp_path)
    bundle_root = tmp_path / "bundle"
    _fixture_bundle(bundle_root)
    before = _snapshot(bundle_root)
    assert before, "fixture bundle must contain files to make this test meaningful"
    configured = _settings(tmp_path, MODEL_BUNDLE_ROOT=str(bundle_root), EMBEDDING_MODE="real")
    run_data_preflight(manifest.path, data_root=tmp_path, configured=configured, probe_fn=lambda _: _Probe())
    after = _snapshot(bundle_root)
    assert before == after


def test_preflight_does_not_initialize_databases():
    """Static contract check: the container preflight module must never import
    or call into app.db.postgres / app.db.clickhouse - only PostgreSQL/ClickHouse
    schema init or writes would violate the read-only preflight guarantee, and
    those live behind entirely separate modules preflight.py never touches."""
    source = inspect.getsource(preflight_module)
    for forbidden in ("app.db.postgres", "app.db.clickhouse", "init_schema", "app.ingestion.ingest"):
        assert forbidden not in source, f"app/preflight.py must not reference {forbidden!r}"


def test_preflight_only_writes_requested_artifact(tmp_path):
    """run_data_preflight() itself returns a dict and writes nothing at all;
    only the CLI wrapper (scripts/preflight.py) writes the requested --json-out
    artifact. Confirm the whole tmp_path tree is unchanged by the library call."""
    manifest = _manifest(tmp_path)
    before = _snapshot(tmp_path)
    run_data_preflight(
        manifest.path, data_root=tmp_path, configured=_settings(tmp_path), probe_fn=lambda _: _Probe(),
    )
    after = _snapshot(tmp_path)
    assert before == after


def test_failed_preflight_has_no_side_effects(tmp_path):
    """An intentionally broken manifest (nonexistent telemetry) must fail
    cleanly without leaving any partial write behind."""
    manifest = _manifest(tmp_path)
    (tmp_path / "telemetry" / "flight.csv").unlink()
    before = _snapshot(tmp_path)
    report = run_data_preflight(
        manifest.path, data_root=tmp_path, configured=_settings(tmp_path), probe_fn=lambda _: _Probe(),
    )
    after = _snapshot(tmp_path)
    assert any(item["status"] == "fail" for item in report["checks"])
    assert before == after
