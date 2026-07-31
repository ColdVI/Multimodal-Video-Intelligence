"""Optional tracking (plan Sec.8/16), gated on TRACKING_ENRICHMENT_ENABLED (default
false). DEFERRED this session: the plan itself requires the unique-object-count use
case to be validated before tracking is implemented ("yalniz benzersiz nesne sayisi use
case'i dogrulanirsa uygulanmali", Sec.16) -- no such validation was run here (would need
a labeled multi-object-instance video benchmark this environment does not have). This
module exists so the contract shape is fixed and importable now, and so
unique_vehicle_tracks is clearly NOT a canonical hard-filterable field until that gate is
met -- it must never be written to segment_metadata/seg_ch_{d} directly.

lazy-imports bytetrack-equivalent tracking code exactly like detector.py/text_cpu.py do;
importing this module never requires that dependency to be installed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool = False
    tracker: str = "bytetrack"
    min_track_frames: int = 3
    max_track_gap: int = 2


@dataclass(frozen=True)
class TrackingResult:
    segment_id: str
    unique_vehicle_tracks: int | None  # sidecar-only; never a canonical filter field


def run_tracking_not_implemented(*_args, **_kwargs) -> None:
    raise NotImplementedError(
        "Tracking is deferred pending unique-object-count use-case validation "
        "(plan Sec.16) -- see docs/operations/KNOWN_LIMITATIONS.md"
    )


__all__ = ["TrackingConfig", "TrackingResult", "run_tracking_not_implemented"]
