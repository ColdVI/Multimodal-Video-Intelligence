"""Pure aggregation math over per-sampled-frame detector counts, per plan Sec.8.1/8.2.

Every function here is a pure function of a list of per-frame counts (one count per
sampled frame in a window, in chronological order) -- no model, no I/O, fully unit-tested
without a detector. detector.py is the only module that calls a real model and feeds its
output into these.

Null vs zero (Sec.8.2's "detector calismadi ile nesne yok birbirine karistirilmaz"): a
segment the detector never successfully processed must produce None here, not 0.0 --
median/max/p90/persistence of an empty frame_counts list all return None, never 0.0. A
caller passing frame_counts=[0, 0, 0] (detector ran, saw nothing every time) correctly
gets 0.0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.enrichment.contracts import CanonicalDetection, DetectorFailurePolicy, SidecarDetection


class DetectorStrictFailure(RuntimeError):
    pass


def median_visible_count(frame_counts: list[int]) -> float | None:
    return float(np.median(frame_counts)) if frame_counts else None


def max_visible_count(frame_counts: list[int]) -> float | None:
    return float(np.max(frame_counts)) if frame_counts else None


def p90_visible_count(frame_counts: list[int]) -> float | None:
    return float(np.percentile(frame_counts, 90)) if frame_counts else None


def detection_persistence_ratio(frame_counts: list[int], *, min_count: int = 1) -> float | None:
    """Fraction of sampled frames where the visible count was >= min_count -- "how
    consistently was at least one instance visible," not an average count."""
    if not frame_counts:
        return None
    return sum(1 for count in frame_counts if count >= min_count) / len(frame_counts)


@dataclass(frozen=True)
class DetectionResult:
    segment_id: str
    class_counts: dict[str, list[int]] | None  # None == detector failed for this segment
    confidence_distribution: list[float]


def build_canonical_and_sidecar(
    result: DetectionResult, policy: DetectorFailurePolicy, *, vehicle_classes: tuple[str, ...] = ("car", "truck", "bus"),
) -> tuple[CanonicalDetection, SidecarDetection | None]:
    if result.class_counts is None:
        if policy == "strict":
            raise DetectorStrictFailure(f"detector failed for segment {result.segment_id} under strict policy")
        return CanonicalDetection(result.segment_id, None, None), None

    per_frame_vehicle_counts: list[int] = []
    n_frames = max((len(counts) for counts in result.class_counts.values()), default=0)
    for frame_index in range(n_frames):
        total = 0
        for cls in vehicle_classes:
            class_counts = result.class_counts.get(cls, ())
            if frame_index < len(class_counts):
                total += class_counts[frame_index]
        per_frame_vehicle_counts.append(total)

    canonical = CanonicalDetection(
        segment_id=result.segment_id,
        median_visible_vehicle_count=median_visible_count(per_frame_vehicle_counts),
        detection_persistence_ratio=detection_persistence_ratio(per_frame_vehicle_counts),
    )
    sidecar = SidecarDetection(
        segment_id=result.segment_id, class_counts=result.class_counts,
        confidence_distribution=result.confidence_distribution,
        max_visible_vehicle_count=max_visible_count(per_frame_vehicle_counts),
        p90_visible_vehicle_count=p90_visible_count(per_frame_vehicle_counts),
    )
    return canonical, sidecar


__all__ = [
    "DetectorStrictFailure", "DetectionResult", "median_visible_count", "max_visible_count",
    "p90_visible_count", "detection_persistence_ratio", "build_canonical_and_sidecar",
]
