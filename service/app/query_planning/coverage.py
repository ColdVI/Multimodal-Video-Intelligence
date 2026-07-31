"""Dataset field-coverage check: a structured constraint is only turned into a hard
filter if its field is actually available for the active run (or, with no active run, the
legacy dataset). Otherwise it falls back into the semantic residual and the diagnostics
carry an explicit field_unavailable entry -- never a silent DB query against a column that
doesn't exist for this dataset. Mirrors the field-availability logic already exposed by
GET /datasets/{id}/filter-schema (app.main.filter_schema) so parser code has one source of
truth to call instead of growing its own catalog, per plan Sec.6.4.
"""

from __future__ import annotations

from app.query_planning.models import StructuredConstraint
from app.search.filter_schema import CANONICAL_FILTER_FIELDS


def available_field_names(dataset_id: str) -> frozenset[str]:
    """Same three information sources GET /datasets/{id}/filter-schema already uses
    (active-run telemetry registry, or legacy facet presence when there is no active
    run) -- kept independent of that endpoint's response-shaping code (bounds/values for
    the UI) since this function only needs the set of *names*."""
    from app.db import postgres
    from app.db.telemetry_registry import fields_for_run

    snapshot = postgres.get_active_run_snapshot(dataset_id)
    facet_data = postgres.facets(dataset_id)
    if snapshot is None:
        names = {"event_category", "split", "video_id"}
        names.update(name for name, bounds in facet_data.get("telemetry", {}).items() if bounds)
        names.update(name for name, bounds in facet_data.get("counts", {}).items() if bounds)
        return frozenset(names)
    run_id = str(snapshot["run_id"])
    registered = {field.name for field in fields_for_run(dataset_id, run_id)}
    for name in ("event_category", "split", "video_id", "person_count", "vehicle_count", "bus_count", "is_night"):
        available = (
            name in {"event_category", "split", "video_id"}
            or facet_data.get("counts", {}).get(name) is not None
            or (name == "is_night" and bool(facet_data.get("booleans", {}).get(name)))
        )
        if available:
            registered.add(name)
    return frozenset(registered)


def split_by_coverage(
    constraints: "tuple[StructuredConstraint, ...] | list[StructuredConstraint]",
    available_fields: frozenset[str],
) -> "tuple[tuple[StructuredConstraint, ...], tuple[StructuredConstraint, ...]]":
    """Returns (covered, uncovered). A field outside CANONICAL_FILTER_FIELDS entirely
    should never reach here (ontology/rules only ever emit canonical fields), but is
    treated as uncovered rather than raising, since coverage is meant to be the forgiving
    half of validation -- normalize_filters() is the strict allow-list gate."""
    covered = tuple(
        c for c in constraints if c.field in available_fields and c.field in CANONICAL_FILTER_FIELDS
    )
    uncovered = tuple(c for c in constraints if c not in covered)
    return covered, uncovered


def field_unavailable_notes(uncovered: "tuple[StructuredConstraint, ...]") -> tuple[str, ...]:
    return tuple(dict.fromkeys(c.field for c in uncovered))  # dedupe, preserve order


__all__ = ["available_field_names", "split_by_coverage", "field_unavailable_notes"]
