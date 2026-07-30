from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from app.config import settings
from app.db import postgres


class MediaError(RuntimeError):
    def __init__(self, status_code: int, reason: str):
        super().__init__(reason)
        self.status_code = status_code
        self.reason = reason


def _source_path(source_uri: str) -> Path:
    if "::" in source_uri or "://" in source_uri:
        raise MediaError(422, "source is not a directly playable local video file")
    source = Path(source_uri).expanduser().resolve()
    data_root = settings.data_root.expanduser().resolve()
    try:
        source.relative_to(data_root)
    except ValueError as exc:
        raise MediaError(403, "media source is outside DATA_ROOT") from exc
    if not source.is_file():
        raise MediaError(404, "media source does not exist")
    return source


def describe_segment(segment_id: str, run_id: str | None = None) -> dict[str, Any]:
    segment = postgres.resolve_media_segment(segment_id, run_id)
    if segment is None:
        raise MediaError(404, "segment was not found in the requested or active run")
    duration = float(segment["t_end"]) - float(segment["t_start"])
    if duration <= 0:
        raise MediaError(422, "segment has a non-positive duration")
    if duration > settings.media_max_clip_s:
        raise MediaError(
            422,
            f"segment duration {duration:.3f}s exceeds MEDIA_MAX_CLIP_S={settings.media_max_clip_s:g}",
        )
    source = _source_path(str(segment["source_uri"]))
    return {**segment, "source_path": source, "duration": duration}


def _cache_key(segment: dict[str, Any]) -> str:
    source: Path = segment["source_path"]
    stat = source.stat()
    material = {
        "source": str(source),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "t_start": float(segment["t_start"]),
        "t_end": float(segment["t_end"]),
        "codec": "libx264/yuv420p/faststart",
        "crf": settings.media_h264_crf,
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()


def _cache_root() -> Path:
    root = settings.artifacts_root.expanduser().resolve() / "media_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def prune_cache(root: Path | None = None, *, now: float | None = None, preserve: Path | None = None) -> None:
    cache = root or _cache_root()
    current = time.time() if now is None else now
    retention_s = settings.media_cache_retention_hours * 3600.0
    entries: list[tuple[Path, os.stat_result]] = []
    for path in cache.glob("*.mp4"):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        if path != preserve and current - stat.st_mtime > retention_s:
            path.unlink(missing_ok=True)
        else:
            entries.append((path, stat))
    size_limit = int(settings.media_cache_max_gb * 1024**3)
    total = sum(stat.st_size for _, stat in entries)
    for path, stat in sorted(entries, key=lambda item: item[1].st_mtime):
        if total <= size_limit:
            break
        if path == preserve:
            continue
        path.unlink(missing_ok=True)
        total -= stat.st_size


def get_clip(
    segment_id: str,
    run_id: str | None = None,
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> tuple[Path, dict[str, Any]]:
    segment = describe_segment(segment_id, run_id)
    cache = _cache_root()
    target = cache / f"{_cache_key(segment)}.mp4"
    if target.is_file() and target.stat().st_size > 0:
        os.utime(target, None)
        return target, segment

    temporary = cache / f".{target.stem}.{uuid.uuid4().hex}.partial.mp4"
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f'{float(segment["t_start"]):.6f}',
        "-i", str(segment["source_path"]),
        "-t", f'{float(segment["duration"]):.6f}',
        "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(settings.media_h264_crf),
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(temporary),
    ]
    try:
        result = runner(command, capture_output=True, text=True, timeout=max(30.0, segment["duration"] * 4.0))
        if result.returncode != 0:
            reason = (result.stderr or result.stdout or "ffmpeg failed").strip()[-1000:]
            raise MediaError(422, f"clip generation failed: {reason}")
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise MediaError(422, "clip generation produced no media")
        temporary.replace(target)
    except FileNotFoundError as exc:
        raise MediaError(503, "ffmpeg executable is unavailable") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError(504, "clip generation timed out") from exc
    finally:
        temporary.unlink(missing_ok=True)
    prune_cache(cache, preserve=target)
    return target, segment


def media_info(segment_id: str, run_id: str | None = None) -> dict[str, Any]:
    try:
        segment = describe_segment(segment_id, run_id)
    except MediaError as exc:
        return {
            "available": False,
            "source_exists": False,
            "reason": exc.reason,
            "segment_id": segment_id,
            "run_id": run_id,
        }
    try:
        get_clip(segment_id, run_id)
    except MediaError as exc:
        return {
            "available": False,
            "source_exists": True,
            "reason": exc.reason,
            "segment_id": segment_id,
            "run_id": segment.get("run_id"),
            "video_id": segment.get("video_id"),
            "t_start": float(segment["t_start"]),
            "t_end": float(segment["t_end"]),
        }
    from app.auth import signed_media_url
    return {
        "available": True,
        "source_exists": True,
        "reason": None,
        "segment_id": segment_id,
        "run_id": segment.get("run_id"),
        "video_id": segment.get("video_id"),
        "t_start": float(segment["t_start"]),
        "t_end": float(segment["t_end"]),
        "clip_url": signed_media_url(segment_id, segment.get("run_id")),
    }


__all__ = ["MediaError", "describe_segment", "get_clip", "media_info", "prune_cache"]
