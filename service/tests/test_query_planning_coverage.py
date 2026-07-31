from __future__ import annotations

from app.query_planning.coverage import available_field_names, field_unavailable_notes, split_by_coverage
from app.query_planning.models import ConstraintProvenance, StructuredConstraint


def _soft(field: str) -> StructuredConstraint:
    return StructuredConstraint(field, "gte", 1, ConstraintProvenance(source="llm_parser", hard=False, relaxable=True))


def test_split_by_coverage_separates_available_from_unavailable_fields():
    constraints = [_soft("vehicle_count"), _soft("velocity_mps")]
    covered, uncovered = split_by_coverage(constraints, frozenset({"vehicle_count"}))
    assert [c.field for c in covered] == ["vehicle_count"]
    assert [c.field for c in uncovered] == ["velocity_mps"]


def test_split_by_coverage_treats_non_canonical_field_as_uncovered_even_if_listed_available():
    constraints = [_soft("not_a_real_field")]
    covered, uncovered = split_by_coverage(constraints, frozenset({"not_a_real_field"}))
    assert covered == ()
    assert [c.field for c in uncovered] == ["not_a_real_field"]


def test_field_unavailable_notes_dedupes_and_preserves_first_seen_order():
    constraints = [_soft("velocity_mps"), _soft("altitude_m"), _soft("velocity_mps")]
    notes = field_unavailable_notes(tuple(constraints))
    assert notes == ("velocity_mps", "altitude_m")


def test_available_field_names_reflects_legacy_dataset_facets(monkeypatch):
    from app.query_planning import coverage as coverage_module

    class FakePostgres:
        @staticmethod
        def get_active_run_snapshot(dataset_id):
            return None

        @staticmethod
        def facets(dataset_id):
            return {
                "telemetry": {"altitude_m": [0.0, 100.0], "velocity_mps": None},
                "counts": {"person_count": [0, 5], "vehicle_count": None},
            }

    monkeypatch.setattr("app.db.postgres.get_active_run_snapshot", FakePostgres.get_active_run_snapshot)
    monkeypatch.setattr("app.db.postgres.facets", FakePostgres.facets)
    names = coverage_module.available_field_names("mini")
    assert "altitude_m" in names  # has real bounds
    assert "velocity_mps" not in names  # bounds are None -> not actually available
    assert "person_count" in names
    assert "vehicle_count" not in names
    assert {"event_category", "split", "video_id"} <= names
