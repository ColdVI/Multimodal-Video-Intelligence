from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from PIL import Image

from app.config import Settings
from app.ingestion import video
from app.ingestion.ingest import _metadata_rows
from app.ingestion.generic_loader import WindowRecord, iter_window_records, release_frames
from app.ingestion.manifest import load_manifest
from app.preflight import discover_pairs, run_data_preflight


ROOT = Path(__file__).resolve().parents[2]
OFFLINE_COMPOSE = ROOT / "docker-compose.offline-gpu.yml"
VIDEO_ONLY_MANIFEST = ROOT / "datasets" / "video_only_m2ts.yaml"
EXPECTED_SERVICES = {"pg", "ch", "api", "ui"}


def _compose_payload() -> dict:
    assert OFFLINE_COMPOSE.is_file(), "standalone offline compose is missing"
    return yaml.safe_load(OFFLINE_COMPOSE.read_text(encoding="utf-8"))


def test_offline_compose_has_no_build_sections():
    payload = _compose_payload()
    assert all("build" not in service for service in payload["services"].values())


def test_offline_compose_uses_pull_policy_never():
    payload = _compose_payload()
    assert all(service.get("pull_policy") == "never" for service in payload["services"].values())
    assert all(":latest" not in service["image"] for service in payload["services"].values())


def test_offline_compose_contains_only_pg_ch_api_ui():
    payload = _compose_payload()
    assert set(payload["services"]) == EXPECTED_SERVICES
    assert payload["services"]["api"]["environment"]["ENABLED_VECTOR_BACKENDS"] == "clickhouse"
    assert payload["services"]["api"]["environment"]["DEFAULT_VECTOR_BACKEND"] == "clickhouse"
    assert str(payload["services"]["api"]["environment"]["ENABLED_DIMENSIONS"]) == "512"
    assert payload["services"]["api"]["environment"]["EMBEDDING_MODE"] == "real"


def test_api_and_ui_use_same_application_image():
    payload = _compose_payload()
    assert payload["services"]["api"]["image"] == payload["services"]["ui"]["image"]
    assert payload["services"]["api"]["image"].startswith("mvi-app-gpu:")
    assert payload["services"]["ui"]["command"] == ["python3", "-m", "ui.app"]
    assert payload["services"]["ui"]["healthcheck"]["test"][1] == "python3"


def test_only_api_requests_gpu():
    payload = _compose_payload()
    assert payload["services"]["api"]["gpus"] == "all"
    assert "gpus" not in payload["services"]["ui"]
    api = payload["services"]["api"]
    assert all("mvi-model-bundle" not in str(item) for item in api.get("volumes", []))
    assert api["environment"]["MODEL_BUNDLE_ROOT"] == "/opt/mvi-model-bundle"
    assert api["environment"]["QWEN_REPO_PATH"] == "/opt/mvi-model-bundle/source"
    assert api["environment"]["QWEN_MODEL_PATH"] == "/opt/mvi-model-bundle/model"

def test_sibling_data_root_resolves_to_workspace_data(tmp_path):
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("docker CLI is unavailable; compose config resolution was not run")
    repository = tmp_path / "folder" / "multi-model"
    videos = tmp_path / "folder" / "videos"
    artifacts = repository / "artifacts"
    repository.mkdir(parents=True)
    videos.mkdir()
    artifacts.mkdir()
    compose = repository / OFFLINE_COMPOSE.name
    compose.write_text(OFFLINE_COMPOSE.read_text(encoding="utf-8"), encoding="utf-8")
    env_file = repository / ".env.offline"
    env_file.write_text(
        "\n".join([
            "MVI_IMAGE_TAG=deadbeef", "POSTGRES_PASSWORD=test-postgres",
            "CLICKHOUSE_PASSWORD=test-clickhouse", "DATA_ROOT=../videos",
            "ARTIFACTS_ROOT=./artifacts",
            "BIND_HOST=127.0.0.1", "API_TOKEN=",
        ]) + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [docker, "compose", "--env-file", str(env_file), "-f", str(compose), "config", "--format", "json"],
        cwd=repository, capture_output=True, text=True, check=False,
    )
    if result.returncode:
        pytest.fail(f"docker compose config failed: {result.stderr}")
    configured = json.loads(result.stdout)
    mounts = configured["services"]["api"]["volumes"]
    data_mount = next(item for item in mounts if item["target"] == "/workspace/data")
    assert Path(data_mount["source"]).resolve() == videos.resolve()
    assert data_mount["read_only"] is True


def test_video_only_manifest_requires_no_telemetry(tmp_path):
    manifest = load_manifest(VIDEO_ONLY_MANIFEST)
    assert manifest.telemetry_glob is None
    assert manifest.telemetry_fields == {}
    assert manifest.telemetry_extra == {}
    data_root = tmp_path / "data"
    nested = data_root / "nested"
    nested.mkdir(parents=True)
    (data_root / "first.m2ts").write_bytes(b"fixture")
    (nested / "SECOND.M2TS").write_bytes(b"fixture")
    (nested / "not-video.txt").write_bytes(b"fixture")
    pairs = discover_pairs(manifest, data_root)
    assert {pair.video_path.name for pair in pairs} == {"first.m2ts", "SECOND.M2TS"}
    assert all(pair.telemetry_path is None for pair in pairs)

    class Probe:
        duration_s = 12.0
        fps = 10.0
        width = 64
        height = 48
        codec = "mpeg2video"
        container_format = "mpegts"
        raw_stream_start_time_s = 1.4
        normalized_timestamp_origin_s = 1.4
        first_decoded_frame_timestamp_s = 1.4
        timestamps_monotonic = True
        creation_time = None

    settings = Settings.from_env({
        "ARTIFACTS_ROOT": str(tmp_path), "EMBEDDING_MODE": "synthetic",
        "ENABLED_VECTOR_BACKENDS": "clickhouse", "ENABLED_DIMENSIONS": "512",
    })
    report = run_data_preflight(VIDEO_ONLY_MANIFEST, data_root=data_root, configured=settings, probe_fn=lambda _: Probe())
    assert report["status"] == "not_run"
    assert not [item for item in report["checks"] if item["category"] == "data" and item["status"] == "fail"]
    assert report["telemetry_enabled"] is False
    assert report["video_count"] == 2
    assert report["sample_video"]["container"] == "mpegts"
    assert report["sample_video"]["codec"] == "mpeg2video"


def _ffmpeg_executable() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
    except ImportError:
        pytest.skip("FFmpeg is unavailable; M2TS fixture was not generated")
    return imageio_ffmpeg.get_ffmpeg_exe()


@pytest.fixture
def mp4(tmp_path: Path) -> Path:
    import cv2
    import numpy as np

    target = tmp_path / "unchanged.mp4"
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV MP4 writer is unavailable")
    for index in range(120):
        writer.write(np.full((48, 64, 3), index % 255, dtype=np.uint8))
    writer.release()
    return target


@pytest.fixture
def m2ts(tmp_path: Path) -> Path:
    pytest.importorskip("av", reason="PyAV is unavailable; native M2TS tests were not run")
    target = tmp_path / "nonzero_pts.m2ts"
    command = [
        _ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10", "-frames:v", "120",
        "-vf", "setpts=PTS+5/TB", "-c:v", "mpeg2video", "-g", "10",
        "-muxdelay", "0", "-muxpreload", "0", "-f", "mpegts", str(target),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        pytest.skip(f"FFmpeg MPEG-TS encoder is unavailable: {result.stderr.strip()}")
    return target


def test_m2ts_probe(m2ts):
    probe = video.probe_video(m2ts)
    assert "mpegts" in probe.container_format
    assert probe.codec == "mpeg2video"
    assert probe.duration_s == pytest.approx(12.0, rel=0.05)
    assert probe.raw_stream_start_time_s is not None
    assert probe.first_decoded_frame_timestamp_s is not None
    assert probe.timestamps_monotonic is True


def test_m2ts_nonzero_pts_is_normalized(m2ts):
    probe = video.probe_video(m2ts)
    assert probe.raw_stream_start_time_s > 0.5
    chunks = list(video.iter_video_chunks(m2ts, chunk_s=5.0, window_size_s=4.0))
    first = next(video.iter_chunk_windows(
        chunks[0], window_size_s=4.0, stride_s=2.0, frames_per_item=4,
    ))
    assert first.t_start == pytest.approx(0.0, abs=0.05)
    assert first.t_start < first.t_end
    assert len(first.frames) == 4
    release_frames([first])


def test_m2ts_chunk_window_generation(m2ts):
    chunks = list(video.iter_video_chunks(m2ts, chunk_s=5.0, window_size_s=4.0))
    windows = []
    for chunk in chunks:
        windows.extend(video.iter_chunk_windows(
            chunk, window_size_s=4.0, stride_s=2.0, frames_per_item=4,
        ))
    starts = [item.t_start for item in windows]
    assert starts == pytest.approx([0.0, 2.0, 4.0, 6.0, 8.0], abs=0.05)
    assert len(starts) == len(set(starts))
    assert next(item for item in windows if item.t_start == pytest.approx(4.0, abs=0.05)).chunk_index == 0
    assert all(item.t_start < item.t_end and len(item.frames) == 4 for item in windows)
    for item in windows:
        release_frames([item])


def test_mp4_behavior_is_unchanged(mp4):
    chunks = list(video.iter_video_chunks(mp4, chunk_s=5.0, window_size_s=4.0))
    windows = [window for chunk in chunks for window in video.iter_chunk_windows(
        chunk, window_size_s=4.0, stride_s=2.0, frames_per_item=4,
    )]
    assert [item.t_start for item in windows] == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert all(len(item.frames) == 4 for item in windows)
    for item in windows:
        release_frames([item])


def test_timestamp_discontinuity_is_explicit(monkeypatch):
    probe = video.VideoProbe(6.0, 1.0, 4, 4, "fake", None)
    chunk = video.VideoChunk(Path("fake.m2ts"), 0, 0.0, 6.0, 6.0, probe)

    def discontinuous(_):
        for timestamp in (0.0, 1.0, 0.5, 2.0, 3.0):
            yield timestamp, Image.new("RGB", (4, 4), "black")

    monkeypatch.setattr(video, "_iter_decoded_frames", discontinuous)
    with pytest.raises(video.TimestampDiscontinuityError, match="non-monotonic"):
        list(video.iter_chunk_windows(
            chunk, window_size_s=2.0, stride_s=1.0, frames_per_item=2,
        ))


def test_video_only_search_hydrates_without_telemetry():
    record = WindowRecord(
        dataset_id="video_only", video_id="v1", segment_id="s1", chunk_index=0,
        t_start=0.0, t_end=8.0, frames=[],
        metadata={"source_path": "/workspace/data/v1.m2ts", "video_duration_s": 12.0},
        telemetry={}, extra={},
    )
    videos, segments, metadata, telemetry = _metadata_rows([record])
    assert videos[0][1] == "v1"
    assert segments[0][3:5] == (0.0, 8.0)
    assert metadata[0][1:4] == (0, 0, 0)
    assert telemetry[0][3:14] == (None,) * 11
