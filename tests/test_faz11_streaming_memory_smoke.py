from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

try:
    import cv2  # noqa: F401
except ImportError:
    cv2 = None

from streaming_memory_smoke import run_smoke  # noqa: E402


@pytest.mark.skipif(cv2 is None, reason="OpenCV MP4 writer is unavailable on this host")
def test_streaming_pipeline_stays_bounded_regardless_of_total_windows(tmp_path):
    report = run_smoke(
        decode_prefetch_windows=3, embed_batch_size=2, db_write_batch_size=4,
        duration_s=20.0, fps=10.0, workdir=tmp_path,
    )
    assert report["windows_processed"] > report["decode_prefetch_windows"]
    assert report["max_live_window_records"] <= report["decode_prefetch_windows"]
    assert report["bounded_behavior_verified"] is True
    assert report["status"] == "pass_synthetic_smoke"
