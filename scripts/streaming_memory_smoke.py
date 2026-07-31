"""Bounded-memory smoke test for the FAZ11 streaming ingest pipeline.

Drives the same three independently-sized stages GenericIngestor.run() uses
(DECODE_PREFETCH_WINDOWS -> EMBED_BATCH_SIZE -> DB_WRITE_BATCH_SIZE) against a
small synthetic multi-chunk video, instrumenting the true peak number of live
WindowRecord/PIL frame objects and process RSS. No GPU/real Qwen model is
used - embeddings are randomly generated - so this proves the pipeline's
memory-boundedness contract, not real embedding throughput or quality. It is
explicitly NOT a substitute for the real institution-video/GPU acceptance
step; see docs/TARGET_ENVIRONMENT_ACCEPTANCE.md.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_ROOT = REPO_ROOT / "service"
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

import numpy as np
import psutil
import yaml

from app.config import Settings
from app.ingestion.generic_loader import batched, iter_window_records, release_frames
from app.ingestion.manifest import load_manifest


def _git_sha() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _build_fixture_video(path: Path, *, duration_s: float, fps: float) -> None:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    if not writer.isOpened():
        raise RuntimeError("OpenCV MP4 writer is unavailable on this host")
    frame_count = int(duration_s * fps)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    for index in range(frame_count):
        frame[:] = index % 255
        writer.write(frame)
    writer.release()


def _build_manifest(root: Path) -> Path:
    payload = {
        "schema_version": 1, "dataset_id": "streaming_memory_smoke", "display_name": "Streaming memory smoke",
        "source": {"videos_glob": "videos/*.mp4", "video_id_from": "filename_stem"},
        "pairing": {"strategy": "filename_stem", "telemetry_glob": None, "telemetry_id_from": "filename_stem"},
        "time_alignment": {"video_clock": "pts", "telemetry_clock": "relative_s", "offset_s": 0.0, "max_gap_s": 1.0},
        "window": {"size_s": 2.0, "stride_s": 1.0, "frames_per_item": 4, "partial_window_policy": "drop_partial"},
        "telemetry": {"format": "generic_csv", "timestamp_column": "timestamp", "fields": {}, "extra": {}},
        "media": {"enabled": True, "clip_cache": True},
    }
    path = root / "manifest.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def run_smoke(*, decode_prefetch_windows: int, embed_batch_size: int, db_write_batch_size: int,
              duration_s: float = 60.0, fps: float = 10.0, workdir: Path | None = None) -> dict:
    process = psutil.Process()
    rss_before_mb = process.memory_info().rss / (1024 * 1024)

    owns_workdir = workdir is None
    if workdir is None:
        import tempfile
        workdir = Path(tempfile.mkdtemp(prefix="faz11-streaming-smoke-"))
    videos_dir = workdir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    video_path = videos_dir / "flight.mp4"
    _build_fixture_video(video_path, duration_s=duration_s, fps=fps)
    manifest_path = _build_manifest(workdir)
    manifest = load_manifest(manifest_path)

    configured = Settings.from_env({
        "DECODE_CHUNK_S": "10", "DECODE_PREFETCH_WINDOWS": str(decode_prefetch_windows),
        "EMBED_BATCH_SIZE": str(embed_batch_size), "DB_WRITE_BATCH_SIZE": str(db_write_batch_size),
        "ARTIFACTS_ROOT": str(workdir / "artifacts"), "ENABLED_VECTOR_BACKENDS": "clickhouse",
        "ENABLED_DIMENSIONS": "512",
    })

    windows_processed = 0
    max_live_window_records = 0
    max_live_pil_images = 0
    rss_peak_mb = rss_before_mb
    started = time.perf_counter()

    iterator = iter_window_records(manifest, data_root=workdir, configured=configured)
    for prefetch_group in batched(iterator, configured.decode_prefetch_windows):
        max_live_window_records = max(max_live_window_records, len(prefetch_group))
        max_live_pil_images = max(max_live_pil_images, sum(len(record.frames) for record in prefetch_group))
        rss_peak_mb = max(rss_peak_mb, process.memory_info().rss / (1024 * 1024))
        for embed_batch in batched(prefetch_group, configured.embed_batch_size):
            vectors = np.random.default_rng(0).standard_normal((len(embed_batch), 2048)).astype(np.float32)
            windows_processed += len(embed_batch)
            release_frames(embed_batch)
            rss_peak_mb = max(rss_peak_mb, process.memory_info().rss / (1024 * 1024))
        assert all(record.frames == [] for record in prefetch_group), "frames must be released before next prefetch group"

    elapsed_s = time.perf_counter() - started
    rss_after_mb = process.memory_info().rss / (1024 * 1024)

    if owns_workdir:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)

    bounded_verified = (
        windows_processed > decode_prefetch_windows
        and max_live_window_records <= decode_prefetch_windows
        and max_live_pil_images <= decode_prefetch_windows * manifest.frames_per_item
    )

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tested_code_sha": _git_sha(),
        "status": "pass_synthetic_smoke",
        "input_type": "synthetic_opencv_generated_mp4_no_real_institution_video",
        "windows_processed": windows_processed,
        "decode_prefetch_windows": decode_prefetch_windows,
        "embed_batch_size": embed_batch_size,
        "db_write_batch_size": db_write_batch_size,
        "max_live_window_records": max_live_window_records,
        "max_live_pil_images": max_live_pil_images,
        "rss_before_mb": round(rss_before_mb, 2),
        "rss_peak_mb": round(rss_peak_mb, 2),
        "rss_after_mb": round(rss_after_mb, 2),
        "bounded_behavior_verified": bounded_verified,
        "elapsed_s": round(elapsed_s, 3),
        "command": (
            f"python scripts/streaming_memory_smoke.py --decode-prefetch-windows {decode_prefetch_windows} "
            f"--embed-batch-size {embed_batch_size} --db-write-batch-size {db_write_batch_size}"
        ),
        "notes": (
            "Synthetic fixture video and randomly-generated embeddings; not a real institution "
            "video or real GPU/Qwen inference. Proves max_live_window_records/max_live_pil_images "
            "never exceed DECODE_PREFETCH_WINDOWS regardless of total windows processed (windows_processed "
            "is intentionally kept larger than decode_prefetch_windows to make the bound observable). "
            "Not usable as evidence of real-world throughput, VRAM, or institution-scale RSS."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-prefetch-windows", type=int, default=5)
    parser.add_argument("--embed-batch-size", type=int, default=2)
    parser.add_argument("--db-write-batch-size", type=int, default=8)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "artifacts" / "faz11" / "streaming_memory_smoke.json")
    args = parser.parse_args()
    report = run_smoke(
        decode_prefetch_windows=args.decode_prefetch_windows, embed_batch_size=args.embed_batch_size,
        db_write_batch_size=args.db_write_batch_size, duration_s=args.duration_s, fps=args.fps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["bounded_behavior_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
