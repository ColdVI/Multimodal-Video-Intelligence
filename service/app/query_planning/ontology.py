"""Canonical concept ontology: user language -> detector taxonomy -> canonical product
filter field, per docs/planning/ADVANCED_RETRIEVAL_FINAL_PLAN_v2.1.md Sec.6.1/11.

Ontology runs BEFORE the parser can be considered correct: the research-plane parser
(search/parser.py) produces `car_count`/`truck_count`, but the product schema
(app.search.filter_schema.CANONICAL_FILTER_FIELDS) only has `vehicle_count`/`bus_count`.
Without this mapping, a parser wired straight to the product schema would 400 on the very
first "araba" query (normalize_filters() rejects unknown fields by design -- see
app.search.pushdown). This module is the thing that makes a parser output well-formed at
all, not an enhancement layered on top of one.

Every canonical_field value here MUST be a real key in CANONICAL_FILTER_FIELDS; that
invariant is enforced at import time (see the assertion below) so a typo here fails loud
in CI rather than producing a mysterious 400 at request time.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.search.filter_schema import CANONICAL_FILTER_FIELDS


@dataclass(frozen=True)
class OntologyConcept:
    concept_id: str
    family: str  # concepts only compete for specificity precedence within the same family
    canonical_field: str
    specificity: int  # higher wins when multiple concepts in the same family match
    detector_classes: tuple[str, ...] = ()
    synonyms_tr: tuple[str, ...] = ()
    synonyms_en: tuple[str, ...] = ()
    widened_from: str | None = None  # set when canonical_field is a fallback, not this concept's own column


ONTOLOGY: dict[str, OntologyConcept] = {
    "vehicle": OntologyConcept(
        concept_id="vehicle", family="vehicle", canonical_field="vehicle_count", specificity=1,
        detector_classes=("car", "truck", "bus", "motorcycle"),
        # Explicit plural/suffix forms, not a looser regex: Turkish agglutinates (araç ->
        # araçlar) and English pluralizes (car -> cars), but matching bare stems with a
        # relaxed boundary would false-positive on unrelated words sharing the same
        # prefix (car -> "care"/"card"/"careful"). Listing the common forms explicitly
        # keeps double word-boundary matching safe. Not full morphological coverage --
        # this is deliberately the "kural tabanlı, dar kapsam" (rules-based, narrow
        # scope) tradeoff CURRENT_SYSTEM.md already documents for the research parser.
        synonyms_tr=("araç", "araçlar", "taşıt", "taşıtlar", "otomobil", "otomobiller", "araba", "arabalar"),
        synonyms_en=("vehicle", "vehicles", "car", "cars", "automobile", "automobiles"),
    ),
    "bus": OntologyConcept(
        concept_id="bus", family="vehicle", canonical_field="bus_count", specificity=2,
        detector_classes=("bus",), synonyms_tr=("otobüs", "otobüsler"), synonyms_en=("bus", "buses"),
    ),
    "truck": OntologyConcept(
        concept_id="truck", family="vehicle", canonical_field="vehicle_count", specificity=2,
        detector_classes=("truck",),
        synonyms_tr=("kamyon", "kamyonlar", "kamyonet", "kamyonetler"), synonyms_en=("truck", "trucks"),
        widened_from="truck",  # no truck_count column exists yet (plan Sec.8.1) -- falls back to vehicle_count
    ),
    "person": OntologyConcept(
        concept_id="person", family="person", canonical_field="person_count", specificity=1,
        detector_classes=("person",),
        synonyms_tr=("insan", "insanlar", "kişi", "kişiler", "yaya", "yayalar", "adam", "adamlar"),
        synonyms_en=("person", "people", "pedestrian", "pedestrians"),
    ),
    "night": OntologyConcept(
        concept_id="night", family="lighting", canonical_field="is_night", specificity=1,
        synonyms_tr=("gece",), synonyms_en=("night", "nighttime"),
    ),
}

_unknown_fields = sorted({
    concept.canonical_field for concept in ONTOLOGY.values() if concept.canonical_field not in CANONICAL_FILTER_FIELDS
})
if _unknown_fields:
    raise RuntimeError(
        f"ontology concept(s) reference canonical_field(s) not in CANONICAL_FILTER_FIELDS: {_unknown_fields}"
    )


@dataclass(frozen=True)
class SpecificityResolution:
    selected: tuple[OntologyConcept, ...]
    elided: tuple[OntologyConcept, ...]  # lower-specificity matches in the same family as a selected concept
    widened: tuple[OntologyConcept, ...]  # subset of `selected` whose canonical_field is a fallback


def resolve_specificity(matched: "list[OntologyConcept] | tuple[OntologyConcept, ...]") -> SpecificityResolution:
    """Within each family, only the highest-specificity match survives (plan example:
    'otobüs' must select bus_count alone, never also add a redundant vehicle_count>=1).
    Concepts in different families never compete -- 'otobüs ve yaya' keeps both bus and
    person. Ties within a family (equal specificity) are all kept, since there is no
    principled way to break a tie -- callers see every tied concept in `selected`."""
    by_family: dict[str, list[OntologyConcept]] = {}
    for concept in matched:
        by_family.setdefault(concept.family, []).append(concept)
    selected: list[OntologyConcept] = []
    elided: list[OntologyConcept] = []
    for concepts in by_family.values():
        best = max(concept.specificity for concept in concepts)
        for concept in concepts:
            (selected if concept.specificity == best else elided).append(concept)
    widened = tuple(concept for concept in selected if concept.widened_from is not None)
    return SpecificityResolution(selected=tuple(selected), elided=tuple(elided), widened=widened)


def ontology_widened_notes(resolution: SpecificityResolution) -> tuple[str, ...]:
    return tuple(f"{concept.widened_from}->{concept.canonical_field}" for concept in resolution.widened)


__all__ = ["OntologyConcept", "ONTOLOGY", "SpecificityResolution", "resolve_specificity", "ontology_widened_notes"]
