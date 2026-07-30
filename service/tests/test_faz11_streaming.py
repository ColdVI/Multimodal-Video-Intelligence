from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from app.config import Settings
from app.ingestion.generic_loader import (
    batched, deterministic_segment_id, iter_window_records, release_frames,
)
from app.ingestion.manifest import load_manifest
from app.ingestion.telemetry import (
    AlignedTelemetryRecord,
    TelemetrySeries,
    circular_interpolate,
    circular_mean,
)
from app.ingestion import video


@pytest.fixture
def mp4(tmp_path):
    import cv2

    path = tmp_path / "fixture.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV MP4 writer is unavailable")
    for index in range(120):
        frame = np.full((48, 64, 3), index % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_probe_and_chunk_halo_window_ownership(mp4):
    probe = video.probe_video(mp4)
    assert probe.fps == pytest.approx(10.0, rel=0.02)
    assert probe.duration_s == pytest.approx(12.0, rel=0.03)
    chunks = list(video.iter_video_chunks(mp4, chunk_s=5.0, window_size_s=4.0))
    assert [(item.chunk_start_s, item.chunk_end_s) for item in chunks] == [(0.0, 5.0), (5.0, 10.0), (10.0, pytest.approx(12.0, rel=0.03))]
    windows = []
    for chunk in chunks:
        windows.extend(video.iter_chunk_windows(
            chunk, window_size_s=4.0, stride_s=2.0, frames_per_item=4,
        ))
    assert [item.t_start for item in windows] == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert len({item.t_start for item in windows}) == len(windows)
    assert next(item for item in windows if item.t_start == 4.0).chunk_index == 0
    assert all(len(item.frames) == 4 for item in windows)
    for item in windows:
        for frame in item.frames:
            frame.close()


def test_partial_final_window_drop_and_pad(mp4):
    chunks = list(video.iter_video_chunks(mp4, chunk_s=10.0, window_size_s=4.0))
    dropped = list(video.iter_chunk_windows(
        chunks[-1], window_size_s=4.0, stride_s=2.0, frames_per_item=4,
        partial_window_policy="drop_partial",
    ))
    assert dropped == []
    padded = list(video.iter_chunk_windows(
        chunks[-1], window_size_s=4.0, stride_s=2.0, frames_per_item=4,
        partial_window_policy="pad_last",
    ))
    assert len(padded) == 1 and padded[0].t_start == 10.0
    assert len(padded[0].frames) == 4
    for frame in padded[0].frames:
        frame.close()


def test_decoder_yields_before_whole_chunk_is_materialized(monkeypatch):
    probe = video.VideoProbe(20.0, 1.0, 4, 4, "fake", None)
    chunk = video.VideoChunk(Path("fake.mp4"), 0, 0.0, 10.0, 12.0, probe)
    decoded = []

    def fake_frames(_):
        for timestamp in range(12):
            decoded.append(timestamp)
            yield float(timestamp), Image.new("RGB", (4, 4), (timestamp, 0, 0))

    monkeypatch.setattr(video, "_iter_decoded_frames", fake_frames)
    iterator = video.iter_chunk_windows(
        chunk, window_size_s=2.0, stride_s=1.0, frames_per_item=2,
    )
    first = next(iterator)
    assert first.t_start == 0.0
    assert max(decoded) <= 1
    assert len(decoded) < 12
    for frame in first.frames:
        frame.close()
    iterator.close()


def _telemetry_manifest(tmp_path):
    payload = {
        "schema_version": 1, "dataset_id": "t", "display_name": "T",
        "source": {"videos_glob": "*.mp4", "video_id_from": "filename_stem"},
        "pairing": {"strategy": "filename_stem", "telemetry_glob": "*.csv", "telemetry_id_from": "filename_stem"},
        "time_alignment": {"video_clock": "pts", "telemetry_clock": "relative_s", "offset_s": 0, "max_gap_s": 10},
        "window": {"size_s": 2, "stride_s": 1, "frames_per_item": 2},
        "telemetry": {
            "format": "generic_csv", "timestamp_column": "t",
            "fields": {
                "compass_heading": {"source": "heading", "type": "circular_deg", "unit": "deg"},
                "altitude_m": {"source": "alt", "type": "continuous", "unit": "m", "reference": "AGL"},
            },
            "extra": {"mode": {"source": "mode", "type": "categorical"}},
        },
    }
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return load_manifest(path)


def test_circular_interpolation_and_mean_wrap_through_zero():
    assert circular_interpolate(359.0, 1.0, 0.5) == pytest.approx(0.0, abs=1e-6)
    value = circular_mean([359.0, 1.0])
    assert value == pytest.approx(0.0, abs=1e-6) or value == pytest.approx(360.0, abs=1e-6)


def test_telemetry_window_aggregation_preserves_extra_and_circular_semantics(tmp_path):
    manifest = _telemetry_manifest(tmp_path)
    series = TelemetrySeries([
        AlignedTelemetryRecord(0.0, {"compass_heading": 359.0, "altitude_m": 10.0}, {"mode": "EO"}),
        AlignedTelemetryRecord(2.0, {"compass_heading": 1.0, "altitude_m": 14.0}, {"mode": "IR"}),
    ], manifest)
    canonical, extra = series.aggregate_window(0.0, 2.0)
    assert canonical["altitude_m"] == pytest.approx(12.0)
    assert canonical["compass_heading"] == pytest.approx(0.0, abs=1e-6)
    assert extra["mode"] == "EO"


def test_deterministic_segment_id_and_batched_contract():
    assert deterministic_segment_id("d", "v", 1.2344, 9.8764) == "d:v:1234:9876"
    assert list(batched(range(5), 2)) == [[0, 1], [2, 3], [4]]


def test_window_record_iterator_is_lazy_and_carries_telemetry(mp4):
    manifest = _telemetry_manifest(mp4.parent)
    (mp4.parent / "fixture.csv").write_text(
        "t,heading,alt,mode\n0,359,10,EO\n2,1,14,IR\n",
        encoding="utf-8",
    )
    configured = Settings.from_env({
        "DECODE_CHUNK_S": "5", "DECODE_PREFETCH_WINDOWS": "2",
        "ARTIFACTS_ROOT": str(mp4.parent),
    })
    iterator = iter_window_records(manifest, data_root=mp4.parent, configured=configured)
    first = next(iterator)
    assert first.segment_id == "t:fixture:0:2000"
    assert first.chunk_index == 0
    assert len(first.frames) == 2
    assert first.telemetry["altitude_m"] == pytest.approx(12.0)
    assert first.extra["mode"] == "EO"
    release_frames([first])
    assert first.frames == []
    iterator.close()
