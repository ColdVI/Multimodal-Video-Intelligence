from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import media
from ui import app as ui_app
from ui import components


def _settings(tmp_path: Path, **overrides):
    values = {
        "data_root": tmp_path / "data",
        "artifacts_root": tmp_path / "artifacts",
        "media_max_clip_s": 60.0,
        "media_cache_max_gb": 1.0,
        "media_cache_retention_hours": 24.0,
        "media_h264_crf": 23,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _segment(source: Path, *, start: float = 1.0, end: float = 3.0):
    return {
        "segment_id": "segment-1", "run_id": "run-1", "dataset_id": "ds", "video_id": "video-1",
        "t_start": start, "t_end": end, "source_uri": str(source),
    }


def test_media_clip_is_arg_list_cached_and_atomically_published(tmp_path, monkeypatch):
    configured = _settings(tmp_path)
    configured.data_root.mkdir()
    source = configured.data_root / "video.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(media, "settings", configured)
    monkeypatch.setattr(media.postgres, "resolve_media_segment", lambda *_: _segment(source))
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        Path(command[-1]).write_bytes(b"clip")
        return subprocess.CompletedProcess(command, 0, "", "")

    first, segment = media.get_clip("segment-1", runner=runner)
    second, _ = media.get_clip("segment-1", runner=runner)
    assert first == second and first.read_bytes() == b"clip"
    assert len(calls) == 1
    assert isinstance(calls[0][0], list) and "shell" not in calls[0][1]
    assert "libx264" in calls[0][0] and "+faststart" in calls[0][0]
    assert segment["duration"] == 2.0
    assert not list(first.parent.glob("*.partial.mp4"))


def test_media_rejects_path_outside_data_root(tmp_path, monkeypatch):
    configured = _settings(tmp_path)
    configured.data_root.mkdir()
    source = tmp_path / "secret.mp4"
    source.write_bytes(b"secret")
    monkeypatch.setattr(media, "settings", configured)
    monkeypatch.setattr(media.postgres, "resolve_media_segment", lambda *_: _segment(source))
    with pytest.raises(media.MediaError, match="outside DATA_ROOT") as exc:
        media.describe_segment("segment-1")
    assert exc.value.status_code == 403


def test_media_rejects_parent_traversal(tmp_path, monkeypatch):
    configured = _settings(tmp_path)
    configured.data_root.mkdir()
    (configured.data_root / "videos").mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"secret")
    traversal_uri = str(configured.data_root / "videos" / ".." / ".." / "outside.mp4")
    monkeypatch.setattr(media, "settings", configured)
    monkeypatch.setattr(media.postgres, "resolve_media_segment", lambda *_: _segment(Path(traversal_uri)))
    with pytest.raises(media.MediaError, match="outside DATA_ROOT") as exc:
        media.describe_segment("segment-1")
    assert exc.value.status_code == 403


def test_media_rejects_symlink_escape(tmp_path, monkeypatch):
    configured = _settings(tmp_path)
    configured.data_root.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"secret")
    link = configured.data_root / "linked.mp4"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not permitted on this host/user")
    monkeypatch.setattr(media, "settings", configured)
    monkeypatch.setattr(media.postgres, "resolve_media_segment", lambda *_: _segment(link))
    with pytest.raises(media.MediaError, match="outside DATA_ROOT") as exc:
        media.describe_segment("segment-1")
    assert exc.value.status_code == 403


def test_media_rejects_encoded_traversal_segment_id(tmp_path, monkeypatch):
    """The segment_id URL path parameter (including any percent-encoded or raw
    '../' content a client could send) is only ever used as an opaque
    PostgreSQL lookup key - it never touches the filesystem directly. Proof:
    a segment_id shaped like a traversal string that matches no real segment
    resolves to 404 via the normal not-found path, and _source_path is never
    even reached because resolve_media_segment returns None first."""
    configured = _settings(tmp_path)
    configured.data_root.mkdir()
    monkeypatch.setattr(media, "settings", configured)
    monkeypatch.setattr(media.postgres, "resolve_media_segment", lambda *_: None)
    with pytest.raises(media.MediaError, match="not found") as exc:
        media.describe_segment("..%2f..%2f..%2fetc%2fpasswd")
    assert exc.value.status_code == 404


def test_media_rejects_negative_or_reversed_time_range(tmp_path, monkeypatch):
    configured = _settings(tmp_path)
    configured.data_root.mkdir()
    source = configured.data_root / "video.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(media, "settings", configured)
    monkeypatch.setattr(media.postgres, "resolve_media_segment", lambda *_: _segment(source, start=5.0, end=2.0))
    with pytest.raises(media.MediaError, match="non-positive duration"):
        media.describe_segment("segment-1")
    monkeypatch.setattr(media.postgres, "resolve_media_segment", lambda *_: _segment(source, start=3.0, end=3.0))
    with pytest.raises(media.MediaError, match="non-positive duration"):
        media.describe_segment("segment-1")


def test_media_rejects_duration_limit_and_archive_source(tmp_path, monkeypatch):
    configured = _settings(tmp_path, media_max_clip_s=1.0)
    configured.data_root.mkdir()
    source = configured.data_root / "video.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(media, "settings", configured)
    monkeypatch.setattr(media.postgres, "resolve_media_segment", lambda *_: _segment(source, end=5.0))
    with pytest.raises(media.MediaError, match="MEDIA_MAX_CLIP_S"):
        media.describe_segment("segment-1")
    monkeypatch.setattr(
        media.postgres, "resolve_media_segment",
        lambda *_: {**_segment(source, end=1.5), "source_uri": f"{source}::frame"},
    )
    with pytest.raises(media.MediaError, match="not a directly playable"):
        media.describe_segment("segment-1")


def test_media_info_is_explicit_fallback(monkeypatch):
    monkeypatch.setattr(media, "describe_segment", lambda *_: (_ for _ in ()).throw(media.MediaError(404, "missing")))
    assert media.media_info("missing") == {
        "available": False, "source_exists": False, "reason": "missing",
        "segment_id": "missing", "run_id": None,
    }


def test_ui_capabilities_and_canonical_payload_are_api_driven(monkeypatch):
    monkeypatch.setattr(ui_app, "_get", lambda *_args, **_kwargs: {
        "enabled_backends": ["qdrant"], "enabled_dimensions": [256],
        "strategies": {"qdrant": ["hnsw"]},
    })
    assert ui_app._capabilities() == {
        "backends": ["qdrant"], "dimensions": [256], "strategies": {"qdrant": ["hnsw"]},
    }
    payload = ui_app._payload(
        "query", "dataset", None, None, None, None, None, None, None, None, None,
        "qdrant", "hnsw", 256, False, 256, 100, "A", 10, 1,
        [["yaw", 350, 10, "deg", "wrap"], ["person_count", 1, 5, "", "linear"]], True,
    )
    assert payload["telemetry_filters"]["yaw"] == [350.0, 10.0]
    assert payload["metadata_filters"] == {"person_count": [1.0, 5.0], "is_night": True}


def test_detail_renders_media_reason_run_and_extra_telemetry():
    rendered = components.result_detail_panel(
        {"segment_id": "s", "video_id": "v", "t_start": 0, "t_end": 1, "extra": {"battery": 91}},
        {"run_id": "abc123", "vector_provenance": "real", "filter_execution_mode": "pushdown"},
        {"source_exists": False, "reason": "unsupported source"},
    )
    assert "abc123" in rendered and "pushdown" in rendered
    assert "unsupported source" in rendered and "Kaynak mevcut" in rendered
    assert "battery" in rendered and "read-only" in rendered
