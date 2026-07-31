from __future__ import annotations

import pytest

from app.enrichment.aggregation import (
    DetectionResult, DetectorStrictFailure, build_canonical_and_sidecar, detection_persistence_ratio,
    max_visible_count, median_visible_count, p90_visible_count,
)


def test_median_max_p90_on_a_typical_frame_series():
    counts = [1, 2, 2, 3, 5]
    assert median_visible_count(counts) == 2.0
    assert max_visible_count(counts) == 5.0
    assert p90_visible_count(counts) == pytest.approx(4.2)


def test_empty_frame_counts_returns_none_not_zero():
    """The one invariant this module exists to protect: 'detector did not run' (None)
    must never collapse into 'detector ran and saw zero' (0.0)."""
    assert median_visible_count([]) is None
    assert max_visible_count([]) is None
    assert p90_visible_count([]) is None
    assert detection_persistence_ratio([]) is None


def test_all_zero_frame_counts_correctly_returns_zero_not_none():
    """The other half of the same invariant: a detector that ran and legitimately saw
    nothing in every frame must report 0.0, not None."""
    assert median_visible_count([0, 0, 0]) == 0.0
    assert detection_persistence_ratio([0, 0, 0]) == 0.0


def test_persistence_ratio_counts_fraction_of_frames_meeting_min_count():
    counts = [0, 1, 2, 0, 3]  # 3 of 5 frames have >=1
    assert detection_persistence_ratio(counts) == pytest.approx(0.6)
    assert detection_persistence_ratio(counts, min_count=2) == pytest.approx(0.4)  # 2 of 5 have >=2


def test_build_canonical_sums_vehicle_classes_per_frame():
    result = DetectionResult(
        segment_id="s1",
        class_counts={"car": [1, 2, 1], "truck": [0, 1, 0], "bus": [0, 0, 1], "person": [5, 5, 5]},
        confidence_distribution=[0.8, 0.9, 0.7],
    )
    canonical, sidecar = build_canonical_and_sidecar(result, "best_effort")
    # per-frame vehicle totals: [1+0+0, 2+1+0, 1+0+1] = [1, 3, 2]
    assert canonical.median_visible_vehicle_count == 2.0
    assert canonical.detection_persistence_ratio == pytest.approx(1.0)  # all 3 frames have >=1
    assert sidecar.max_visible_vehicle_count == 3.0
    assert sidecar.class_counts == result.class_counts  # person counts preserved in the sidecar, not lost


def test_build_canonical_best_effort_failure_returns_none_values_not_zero():
    result = DetectionResult(segment_id="s1", class_counts=None, confidence_distribution=[])
    canonical, sidecar = build_canonical_and_sidecar(result, "best_effort")
    assert canonical.median_visible_vehicle_count is None
    assert canonical.detection_persistence_ratio is None
    assert sidecar is None


def test_build_canonical_strict_failure_raises():
    result = DetectionResult(segment_id="s1", class_counts=None, confidence_distribution=[])
    with pytest.raises(DetectorStrictFailure, match="s1"):
        build_canonical_and_sidecar(result, "strict")


def test_build_canonical_handles_uneven_class_frame_counts():
    """A class that was never detected in later frames may have a shorter list than
    others (real detector output shape) -- must not crash, must treat missing indices
    as 0, not skip them."""
    result = DetectionResult(
        segment_id="s1", class_counts={"car": [1, 1, 1], "truck": [1]}, confidence_distribution=[],
    )
    canonical, sidecar = build_canonical_and_sidecar(result, "best_effort")
    # frame 0: car=1,truck=1 -> 2; frame 1: car=1,truck=0(missing) -> 1; frame 2: car=1,truck=0 -> 1
    assert canonical.median_visible_vehicle_count == 1.0
