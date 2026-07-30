from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeVar
from zoneinfo import ZoneInfo

from PIL import Image

from app.config import Settings, settings
from app.ingestion.manifest import DatasetManifest, load_manifest
from app.ingestion.telemetry import TelemetrySeries
from app.ingestion.video import iter_chunk_windows, iter_video_chunks, probe_video
from app.preflight import SourcePair, discover_pairs


@dataclass
class WindowRecord:
    dataset_id: str
    video_id: str
    segment_id: str
    chunk_index: int
    t_start: float
    t_end: float
    frames: list[Image.Image]
    metadata: dict[str, Any]
    telemetry: dict[str, Any]
    extra: dict[str, Any]


def deterministic_segment_id(dataset_id: str, video_id: str, t_start: float, t_end: float) -> str:
    start_ms = int(round(t_start * 1000.0))
    end_ms = int(round(t_end * 1000.0))
    return f"{dataset_id}:{video_id}:{start_ms}:{end_ms}"


def _video_start_unix_s(manifest: DatasetManifest, pair: SourcePair) -> float | None:
    if not manifest.is_absolute_clock:
        return None
    if pair.video_start_unix_s is not None:
        return pair.video_start_unix_s
    if manifest.video_start_time_from == "container_creation_time":
        creation = probe_video(pair.video_path).creation_time
        if creation is None:
            raise ValueError(f"video has no container creation_time anchor: {pair.video_path}")
        return creation.timestamp()
    if manifest.video_start_time_from == "filename":
        if not manifest.filename_time_regex or not manifest.filename_time_format:
            raise ValueError("filename anchor requires regex and format")
        match = re.search(manifest.filename_time_regex, pair.video_path.name)
        if match is None:
            raise ValueError(f"filename time regex did not match {pair.video_path.name}")
        text = match.groupdict().get("timestamp") or match.group(0)
        value = datetime.strptime(text, manifest.filename_time_format)
        return value.replace(tzinfo=ZoneInfo(manifest.timezone)).timestamp()
    raise ValueError(f"absolute telemetry requires an anchor for {pair.video_id}")


def _telemetry_series(manifest: DatasetManifest, pair: SourcePair) -> TelemetrySeries | None:
    if pair.telemetry_path is None:
        return None
    if manifest.telemetry_format != "generic_csv":
        raise ValueError(
            f"telemetry format {manifest.telemetry_format!r} requires a TelemetryAdapter implementation"
        )
    return TelemetrySeries.from_csv(
        pair.telemetry_path,
        manifest,
        video_start_unix_s=_video_start_unix_s(manifest, pair),
        offset_s=pair.offset_s,
    )


def iter_pair_records(
    manifest: DatasetManifest,
    pair: SourcePair,
    *,
    configured: Settings = settings,
) -> Iterator[WindowRecord]:
    telemetry_series = _telemetry_series(manifest, pair)
    for chunk in iter_video_chunks(
        pair.video_path,
        chunk_s=configured.decode_chunk_s,
        window_size_s=manifest.window_size_s,
    ):
        for decoded in iter_chunk_windows(
            chunk,
            window_size_s=manifest.window_size_s,
            stride_s=manifest.stride_s,
            frames_per_item=manifest.frames_per_item,
            partial_window_policy=manifest.partial_window_policy,
        ):
            if telemetry_series is None:
                canonical, extra = {}, {}
            else:
                canonical, extra = telemetry_series.aggregate_window(decoded.t_start, decoded.t_end)
            yield WindowRecord(
                dataset_id=manifest.dataset_id,
                video_id=pair.video_id,
                segment_id=deterministic_segment_id(
                    manifest.dataset_id, pair.video_id, decoded.t_start, decoded.t_end,
                ),
                chunk_index=decoded.chunk_index,
                t_start=decoded.t_start,
                t_end=decoded.t_end,
                frames=decoded.frames,
                metadata={
                    "source_path": str(pair.video_path),
                    "video_duration_s": chunk.probe.duration_s,
                    "fps": chunk.probe.fps,
                    "codec": chunk.probe.codec,
                },
                telemetry=canonical,
                extra=extra,
            )


def iter_window_records(
    manifest_or_path: DatasetManifest | str | Path,
    *,
    data_root: Path,
    configured: Settings = settings,
) -> Iterator[WindowRecord]:
    manifest = (
        manifest_or_path
        if isinstance(manifest_or_path, DatasetManifest)
        else load_manifest(manifest_or_path)
    )
    for pair in discover_pairs(manifest, data_root):
        yield from iter_pair_records(manifest, pair, configured=configured)


T = TypeVar("T")


def batched(values: Iterable[T], size: int) -> Iterator[list[T]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    iterator = iter(values)
    while batch := list(itertools.islice(iterator, size)):
        yield batch


def release_frames(records: Iterable[WindowRecord]) -> None:
    for record in records:
        for frame in record.frames:
            frame.close()
        record.frames.clear()


__all__ = [
    "WindowRecord", "batched", "deterministic_segment_id", "iter_pair_records",
    "iter_window_records", "release_frames",
]
