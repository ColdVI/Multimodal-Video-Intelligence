"""Optional caption enrichment (plan Sec.10/18), gated on CAPTION_MODE (default "off").

Captions are lexical/explanatory aids and hard-negative-analysis input only. This module
enforces the plan's explicit boundary: caption output is never converted into an
authoritative speed/altitude/color/exact-object-count filter -- see
CAPTION_AUTHORITATIVE_FIELDS_FORBIDDEN and assert_caption_not_authoritative(). A caption
model failure must not break core ingest or search: run_caption() catches and reports
rather than propagating, matching detector.py's best_effort default (there is no "strict"
mode for captions -- the plan does not ask for one, since nothing structural depends on
caption output the way relaxation logic depends on detector confidence).

Lazy-imports its model exactly like detector.py/text_cpu.py; importing this module never
requires a captioning dependency to be installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CaptionMode = Literal["off", "sampled", "event_only"]

# These are exactly the fields the plan (Sec.10) forbids a caption from authoritatively
# producing -- telemetry/count fields that already have a trusted source (telemetry
# registry, detector). A caption CAN mention them in free text; it must never become the
# thing normalize_filters() applies as a hard/soft numeric constraint.
CAPTION_AUTHORITATIVE_FIELDS_FORBIDDEN = frozenset({
    "velocity_mps", "altitude_m", "person_count", "vehicle_count", "bus_count", "is_night",
})


@dataclass(frozen=True)
class CaptionRecord:
    segment_id: str
    text: str | None  # None means the caption model failed/was skipped for this segment
    mode: CaptionMode


def assert_caption_not_authoritative(field: str) -> None:
    """Call this at the one place a caller might be tempted to build a StructuredConstraint
    from caption text -- raises immediately rather than letting a caption-derived filter
    reach normalize_filters(). There is currently no code path that does this; this
    function exists so one introduced later fails loudly instead of silently."""
    if field in CAPTION_AUTHORITATIVE_FIELDS_FORBIDDEN:
        raise ValueError(
            f"caption output must never authoritatively set {field!r} -- plan Sec.10/18 forbids this; "
            "use the telemetry registry or detector enrichment instead"
        )


def run_caption(segment_id: str, mode: CaptionMode, caption_fn) -> CaptionRecord:
    if mode == "off":
        raise ValueError("run_caption should not be called when caption_mode='off'")
    try:
        text = caption_fn(segment_id)
        return CaptionRecord(segment_id, text, mode)
    except Exception:
        return CaptionRecord(segment_id, None, mode)


__all__ = [
    "CaptionMode", "CaptionRecord", "CAPTION_AUTHORITATIVE_FIELDS_FORBIDDEN",
    "assert_caption_not_authoritative", "run_caption",
]
