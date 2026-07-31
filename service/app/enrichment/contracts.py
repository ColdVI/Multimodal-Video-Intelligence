"""Detector enrichment config/output contracts, per plan Sec.8.1/8.2.

Only two canonical, filterable columns exist (median_visible_vehicle_count,
detection_persistence_ratio) -- everything else the plan's v1 proposed (max/p90 visible
counts, unique_vehicle_tracks) is a sidecar artifact (SidecarDetection below), never a
migrated column, per Sec.8.1's "5 katı yüzey" rejection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DetectorFailurePolicy = Literal["strict", "best_effort"]


@dataclass(frozen=True)
class DetectorConfig:
    enabled: bool = False
    variant: str = "yolov8n_visdrone"
    frames_per_window: int = 6
    confidence: float = 0.25
    failure_policy: DetectorFailurePolicy = "best_effort"


@dataclass(frozen=True)
class CanonicalDetection:
    """The only two fields that ever reach segment_metadata/run_segment_metadata or
    seg_ch_{d}[_runs]. median_visible_vehicle_count/detection_persistence_ratio being
    None means "detector did not produce a value" (strict-policy failure, or the
    detector never ran) -- 0.0 means "ran successfully, found nothing." A caller must
    never coalesce None to 0 when writing these; see aggregation.py's docstring."""

    segment_id: str
    median_visible_vehicle_count: float | None
    detection_persistence_ratio: float | None


@dataclass(frozen=True)
class SidecarDetection:
    """Written to artifacts/enrichment/<run_id>/<segment_id>.json, never to a DB column.
    class_counts is per-sampled-frame, in detection order (frame 0, frame 1, ...)."""

    segment_id: str
    class_counts: dict[str, list[int]]
    confidence_distribution: list[float]
    max_visible_vehicle_count: float | None
    p90_visible_vehicle_count: float | None


@dataclass(frozen=True)
class DetectorRunOutcome:
    canonical: tuple[CanonicalDetection, ...]
    sidecars: tuple[SidecarDetection, ...]
    failure_policy: DetectorFailurePolicy
    failed_segment_ids: tuple[str, ...]
    status: Literal["completed", "failed"]


__all__ = [
    "DetectorFailurePolicy", "DetectorConfig", "CanonicalDetection",
    "SidecarDetection", "DetectorRunOutcome",
]
