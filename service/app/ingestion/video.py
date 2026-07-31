from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from PIL import Image


@dataclass(frozen=True)
class VideoProbe:
    duration_s: float
    fps: float
    width: int
    height: int
    codec: str
    creation_time: datetime | None


@dataclass(frozen=True)
class VideoChunk:
    path: Path
    chunk_index: int
    chunk_start_s: float
    chunk_end_s: float
    halo_end_s: float
    probe: VideoProbe


@dataclass
class DecodedWindow:
    chunk_index: int
    t_start: float
    t_end: float
    frames: list[Image.Image]


def _creation_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _probe_pyav(path: Path) -> VideoProbe:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.base_rate or stream.guessed_rate or 0)
        if stream.duration is not None and stream.time_base is not None:
            duration_s = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration_s = float(container.duration / av.time_base)
        else:
            duration_s = 0.0
        creation = container.metadata.get("creation_time") or stream.metadata.get("creation_time")
        return VideoProbe(
            duration_s=duration_s,
            fps=fps,
            width=int(stream.codec_context.width),
            height=int(stream.codec_context.height),
            codec=str(stream.codec_context.name or "unknown"),
            creation_time=_creation_time(creation),
        )


def _probe_opencv(path: Path) -> VideoProbe:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        code = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((code >> 8 * index) & 0xFF) for index in range(4)).strip("\x00")
        return VideoProbe(
            duration_s=(frames / fps if fps > 0 else 0.0),
            fps=fps,
            width=width,
            height=height,
            codec=codec or "unknown",
            creation_time=None,
        )
    finally:
        cap.release()


def probe_video(path: Path) -> VideoProbe:
    """Probe with PyAV first; use OpenCV only when PyAV is unavailable."""
    try:
        import av  # noqa: F401
    except ImportError:
        probe = _probe_opencv(path)
    else:
        probe = _probe_pyav(path)
    if probe.duration_s <= 0 or probe.fps <= 0 or probe.width <= 0 or probe.height <= 0:
        raise ValueError(f"invalid video probe for {path}: {probe}")
    return probe


def iter_video_chunks(
    path: Path,
    *,
    chunk_s: float,
    window_size_s: float,
) -> Iterator[VideoChunk]:
    if chunk_s <= 0 or window_size_s <= 0:
        raise ValueError("chunk_s and window_size_s must be positive")
    probe = probe_video(path)
    chunk_index = 0
    chunk_start = 0.0
    while chunk_start < probe.duration_s:
        chunk_end = min(probe.duration_s, chunk_start + chunk_s)
        yield VideoChunk(
            path=path,
            chunk_index=chunk_index,
            chunk_start_s=chunk_start,
            chunk_end_s=chunk_end,
            halo_end_s=min(probe.duration_s, chunk_end + window_size_s),
            probe=probe,
        )
        chunk_index += 1
        chunk_start += chunk_s


def _iter_frames_pyav(chunk: VideoChunk) -> Iterator[tuple[float, Image.Image]]:
    import av

    with av.open(str(chunk.path)) as container:
        stream = container.streams.video[0]
        if stream.time_base is None:
            raise ValueError("video stream has no time_base")
        seek_pts = max(0, int(chunk.chunk_start_s / float(stream.time_base)))
        container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * stream.time_base)
            if timestamp + 1e-9 < chunk.chunk_start_s:
                continue
            if timestamp >= chunk.halo_end_s:
                break
            yield timestamp, frame.to_image()


def _iter_frames_opencv(chunk: VideoChunk) -> Iterator[tuple[float, Image.Image]]:
    import cv2

    cap = cv2.VideoCapture(str(chunk.path))
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {chunk.path}")
        start_frame = max(0, int(math.floor(chunk.chunk_start_s * chunk.probe.fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_index = start_frame
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_index / chunk.probe.fps
            frame_index += 1
            if timestamp + 1e-9 < chunk.chunk_start_s:
                continue
            if timestamp >= chunk.halo_end_s:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield timestamp, Image.fromarray(rgb)
    finally:
        cap.release()


def _iter_decoded_frames(chunk: VideoChunk) -> Iterator[tuple[float, Image.Image]]:
    try:
        import av  # noqa: F401
    except ImportError:
        yield from _iter_frames_opencv(chunk)
    else:
        yield from _iter_frames_pyav(chunk)


def _window_specs(
    chunk: VideoChunk,
    *,
    window_size_s: float,
    stride_s: float,
    frames_per_item: int,
    partial_window_policy: str,
) -> list[tuple[float, float, list[float]]]:
    if stride_s <= 0 or frames_per_item <= 0:
        raise ValueError("stride_s and frames_per_item must be positive")
    first_index = math.ceil((chunk.chunk_start_s - 1e-9) / stride_s)
    start = max(0.0, first_index * stride_s)
    result = []
    while start < chunk.chunk_end_s - 1e-9:
        nominal_end = start + window_size_s
        if nominal_end > chunk.probe.duration_s + 1e-9 and partial_window_policy == "drop_partial":
            start += stride_s
            continue
        end = min(nominal_end, chunk.probe.duration_s)
        sample_span = nominal_end - start
        targets = [start + sample_span * index / frames_per_item for index in range(frames_per_item)]
        result.append((start, end, targets))
        start += stride_s
    return result


def iter_chunk_windows(
    chunk: VideoChunk,
    *,
    window_size_s: float,
    stride_s: float,
    frames_per_item: int,
    partial_window_policy: str = "drop_partial",
) -> Iterator[DecodedWindow]:
    """Decode a chunk once and emit only windows whose ``t_start`` it owns."""
    if partial_window_policy not in {"drop_partial", "pad_last"}:
        raise ValueError("partial_window_policy must be drop_partial or pad_last")
    specs = _window_specs(
        chunk,
        window_size_s=window_size_s,
        stride_s=stride_s,
        frames_per_item=frames_per_item,
        partial_window_policy=partial_window_policy,
    )
    collected: list[list[Image.Image]] = [[] for _ in specs]
    target_positions = [0 for _ in specs]
    last_frame: Image.Image | None = None
    next_emit = 0
    try:
        for timestamp, frame in _iter_decoded_frames(chunk):
            for index, (_, _, targets) in enumerate(specs):
                if index < next_emit or targets[0] > timestamp + 1e-9:
                    continue
                position = target_positions[index]
                while position < len(targets) and targets[position] <= timestamp + 1e-9:
                    collected[index].append(frame.copy())
                    position += 1
                target_positions[index] = position
            # The source decode frame is only needed to produce copies for matching
            # windows and as the pad_last fallback; once superseded it is closed
            # explicitly rather than left for GC, per the explicit ownership contract.
            if last_frame is not None:
                last_frame.close()
            last_frame = frame
            while next_emit < len(specs) and len(collected[next_emit]) == len(specs[next_emit][2]):
                start, end, _ = specs[next_emit]
                frames = collected[next_emit]
                collected[next_emit] = []
                yield DecodedWindow(chunk.chunk_index, start, end, frames)
                next_emit += 1
        for index in range(next_emit, len(specs)):
            start, end, targets = specs[index]
            frames = collected[index]
            collected[index] = []
            if len(frames) < len(targets) and partial_window_policy == "pad_last" and (frames or last_frame):
                pad = frames[-1] if frames else last_frame
                while len(frames) < len(targets):
                    frames.append(pad.copy())  # type: ignore[union-attr]
            if len(frames) != len(targets):
                for frame in frames:
                    frame.close()
                continue
            yield DecodedWindow(chunk.chunk_index, start, end, frames)
    finally:
        # Reached on normal exhaustion (no-op: last_frame/collected are already
        # cleared above) and on early generator close (GeneratorExit from a
        # caller that stops consuming mid-chunk): release whatever is still held.
        if last_frame is not None:
            last_frame.close()
        for leftover in collected:
            for frame in leftover:
                frame.close()


__all__ = [
    "DecodedWindow", "VideoChunk", "VideoProbe", "iter_chunk_windows",
    "iter_video_chunks", "probe_video",
]
