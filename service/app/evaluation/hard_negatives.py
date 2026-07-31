"""Layered hard-negative benchmark, per plan Sec.9/17.

Tier A (auto-derivable from existing count/class metadata, binding): count-difference and
class-proximity pairs, generated here from whatever segment metadata is already ingested
-- no new annotation effort. Tier B (metadata-derivable: day/night, camera motion,
brightness): same idea, different fields, not implemented this session (needs
camera_motion/brightness populated, which existing datasets may not have -- see
KNOWN_LIMITATIONS.md). Tier C (color/direction/fine-detail, human-verified): cannot be
fabricated by this agent; attribute_reachability_precheck() is the gate the plan requires
BEFORE spending human annotation effort on a Tier C attribute, so a query never enters the
binding set testing something the frame resolution can't actually support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

AnnotationMethod = Literal["auto_derived", "human_verified"]
Evaluation = Literal["binding", "exploratory"]


@dataclass(frozen=True)
class HardNegativeQuery:
    query_id: str
    query_tr: str
    query_en: str
    positive_segment_ids: tuple[str, ...]
    hard_negative_segment_ids: tuple[str, ...]
    tested_attributes: tuple[str, ...]
    annotation_method: AnnotationMethod
    attribute_reachable: bool
    evaluation: Evaluation = "binding"

    def as_dict(self) -> dict:
        return {
            "query_id": self.query_id, "query_tr": self.query_tr, "query_en": self.query_en,
            "positive_segment_ids": list(self.positive_segment_ids),
            "hard_negative_segment_ids": list(self.hard_negative_segment_ids),
            "tested_attributes": list(self.tested_attributes),
            "annotation_method": self.annotation_method, "attribute_reachable": self.attribute_reachable,
            "evaluation": self.evaluation,
        }


@dataclass(frozen=True)
class SegmentCountRecord:
    segment_id: str
    counts: dict[str, int]  # e.g. {"person_count": 3, "vehicle_count": 1}


def attribute_reachability_precheck(*, frames_per_window: int, attribute: str, min_frames_for_reachability: int = 3) -> bool:
    """The plan's Sec.9.2/17 gate: 'can a human tell positive from negative using only
    the frames the model actually sees.' This function cannot see pixels -- it can only
    enforce the structural precondition (enough sampled frames exist to judge anything
    at all) and is deliberately conservative: a caller must still have a human actually
    look at sampled frames for color/direction/fine-detail attributes before setting
    attribute_reachable=True on a real HardNegativeQuery. This function alone can never
    justify attribute_reachable=True for a Tier C attribute -- it only catches the
    trivial failure (too few frames to judge anything)."""
    if frames_per_window < min_frames_for_reachability:
        return False
    return attribute in {"count_difference", "class_proximity", "day_night", "camera_motion", "brightness"}


def build_tier_a_count_difference_pairs(
    records: list[SegmentCountRecord], *, count_field: str, id_prefix: str,
) -> list[HardNegativeQuery]:
    """Auto-derivable, binding (plan Sec.9.3 Tier A: 'sayi farki'). For each distinct
    positive count value >0, segments with that count are positives and segments with
    count==0 are hard negatives -- no human judgment needed, the count field already
    settles it."""
    by_count: dict[int, list[str]] = {}
    for record in records:
        by_count.setdefault(record.counts.get(count_field, 0), []).append(record.segment_id)
    negatives = tuple(by_count.get(0, ()))
    if not negatives:
        return []  # no zero-count segments -> nothing to contrast positives against
    queries = []
    for count_value, segment_ids in sorted(by_count.items()):
        if count_value == 0 or not segment_ids:
            continue
        queries.append(HardNegativeQuery(
            query_id=f"{id_prefix}_count_{count_value}",
            query_tr=f"{count_field} alaninda tam {count_value} olan sahne",
            query_en=f"a scene with exactly {count_value} {count_field}",
            positive_segment_ids=tuple(segment_ids), hard_negative_segment_ids=negatives,
            tested_attributes=("count_difference",), annotation_method="auto_derived",
            attribute_reachable=True, evaluation="binding",
        ))
    return queries


def build_tier_a_class_proximity_pairs(
    records: list[SegmentCountRecord], *, primary_field: str, secondary_field: str, id_prefix: str,
) -> list[HardNegativeQuery]:
    """plan Sec.9.3: 'kisi+arac vs yalniz arac' -- positives have both classes present,
    hard negatives have the secondary class only (a plausible confusion: 'a scene with
    people and vehicles' vs 'a scene with vehicles but no people')."""
    both = [r.segment_id for r in records if r.counts.get(primary_field, 0) > 0 and r.counts.get(secondary_field, 0) > 0]
    secondary_only = [r.segment_id for r in records if r.counts.get(primary_field, 0) == 0 and r.counts.get(secondary_field, 0) > 0]
    if not both or not secondary_only:
        return []
    return [HardNegativeQuery(
        query_id=f"{id_prefix}_class_proximity",
        query_tr=f"{primary_field} ve {secondary_field} birlikte gorulen sahne",
        query_en=f"a scene with both {primary_field} and {secondary_field}",
        positive_segment_ids=tuple(both), hard_negative_segment_ids=tuple(secondary_only),
        tested_attributes=("class_proximity",), annotation_method="auto_derived",
        attribute_reachable=True, evaluation="binding",
    )]


__all__ = [
    "AnnotationMethod", "Evaluation", "HardNegativeQuery", "SegmentCountRecord",
    "attribute_reachability_precheck", "build_tier_a_count_difference_pairs",
    "build_tier_a_class_proximity_pairs",
]
