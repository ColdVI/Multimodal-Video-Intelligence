from __future__ import annotations

import pytest

from app.query_planning.models import (
    ConstraintProvenance, ParsedQuery, QueryExecutionPlan, RelaxationPolicy,
    StructuredConstraint, constraints_to_filter_dict,
)
from app.search.pushdown import normalize_filters


def test_explicit_request_provenance_must_be_hard_and_not_relaxable():
    with pytest.raises(ValueError):
        ConstraintProvenance(source="explicit_request", hard=False, relaxable=False)
    with pytest.raises(ValueError):
        ConstraintProvenance(source="explicit_request", hard=True, relaxable=True)
    ConstraintProvenance(source="explicit_request", hard=True, relaxable=False)  # ok


def test_confidence_must_be_in_unit_interval():
    with pytest.raises(ValueError):
        ConstraintProvenance(source="llm_parser", confidence=1.5, hard=False, relaxable=True)
    with pytest.raises(ValueError):
        ConstraintProvenance(source="llm_parser", confidence=-0.1, hard=False, relaxable=True)


def test_structured_constraint_explicit_factory_matches_plan_example_shape():
    constraint = StructuredConstraint.explicit("vehicle_count", "gte", 1)
    payload = constraint.as_dict()
    assert payload == {
        "field": "vehicle_count", "operator": "gte", "value": 1,
        "source": "explicit_request", "data_source": "user",
        "confidence": 1.0, "hard": True, "relaxable": False,
    }


def test_constraints_to_filter_dict_handles_eq_gte_lte_range():
    constraints = [
        StructuredConstraint("event_category", "eq", "traffic", ConstraintProvenance(source="explicit_request")),
        StructuredConstraint("vehicle_count", "gte", 1, ConstraintProvenance(source="llm_parser", hard=False, relaxable=True)),
        StructuredConstraint("altitude_m", "lte", 100.0, ConstraintProvenance(source="llm_parser", hard=False, relaxable=True)),
        StructuredConstraint("velocity_mps", "range", (5.0, 10.0), ConstraintProvenance(source="telemetry_derived")),
    ]
    result = constraints_to_filter_dict(constraints)
    assert result["event_category"] == "traffic"
    assert result["vehicle_count"] == {"min": 1}
    assert result["altitude_m"] == {"max": 100.0}
    assert result["velocity_mps"] == {"min": 5.0, "max": 10.0}


def test_constraints_to_filter_dict_merges_multiple_constraints_on_same_field_to_tightest_bound():
    """A detector-derived gte=1 and a parser-derived gte=3 on vehicle_count must merge to
    the tighter gte=3, not silently pick whichever constraint happened to come last."""
    constraints = [
        StructuredConstraint("vehicle_count", "gte", 1, ConstraintProvenance(source="detector_derived", hard=False, relaxable=True)),
        StructuredConstraint("vehicle_count", "gte", 3, ConstraintProvenance(source="llm_parser", hard=False, relaxable=True)),
        StructuredConstraint("vehicle_count", "lte", 10, ConstraintProvenance(source="llm_parser", hard=False, relaxable=True)),
        StructuredConstraint("vehicle_count", "lte", 8, ConstraintProvenance(source="detector_derived", hard=False, relaxable=True)),
    ]
    result = constraints_to_filter_dict(constraints)
    assert result["vehicle_count"] == {"min": 3, "max": 8}


def test_constraints_to_filter_dict_output_is_accepted_by_normalize_filters():
    """The whole point of this bridge: its output must be exactly what the existing,
    already-safe pushdown layer expects -- no parallel filter-building path."""
    constraints = [
        StructuredConstraint.explicit("bus_count", "gte", 1),
        StructuredConstraint("is_night", "eq", True, ConstraintProvenance(source="rules_parser", hard=False, relaxable=True)),
    ]
    filter_dict = constraints_to_filter_dict(constraints)
    predicates = normalize_filters(filter_dict, None)
    assert len(predicates) == 2


def test_constraints_to_filter_dict_rejects_unknown_field_via_normalize_filters():
    constraints = [StructuredConstraint("not_a_real_field", "eq", 1, ConstraintProvenance(source="llm_parser", hard=False, relaxable=True))]
    filter_dict = constraints_to_filter_dict(constraints)
    with pytest.raises(ValueError, match="unknown or non-filterable field"):
        normalize_filters(filter_dict, None)


def test_parsed_query_unparsed_is_the_none_mode_default():
    parsed = ParsedQuery.unparsed("kalabalık trafik")
    assert parsed.structured_constraints == ()
    assert parsed.semantic_residual == "kalabalık trafik"
    assert parsed.residual_policy == "full_query"
    assert parsed.diagnostics.parser_mode == "none"


def test_relaxation_policy_rejects_invalid_values():
    with pytest.raises(ValueError):
        RelaxationPolicy(max_relaxation_passes=0)
    with pytest.raises(ValueError):
        RelaxationPolicy(relaxation_timeout_ms=0)
    with pytest.raises(ValueError):
        RelaxationPolicy(min_results=-1)
    RelaxationPolicy()  # defaults are valid


def test_query_execution_plan_splits_hard_and_soft_constraints():
    parsed = ParsedQuery(
        raw_query="q", semantic_residual="q", residual_policy="full_query",
        diagnostics=ParsedQuery.unparsed("q").diagnostics,
        structured_constraints=(
            StructuredConstraint.explicit("bus_count", "gte", 1),
            StructuredConstraint("vehicle_count", "gte", 1, ConstraintProvenance(source="llm_parser", hard=False, relaxable=True)),
            StructuredConstraint("person_count", "gte", 1, ConstraintProvenance(source="detector_derived", hard=False, relaxable=True)),
        ),
    )
    plan = QueryExecutionPlan.from_parsed_query(parsed, RelaxationPolicy(mode="auto_soft"))
    assert len(plan.hard_constraints) == 1
    assert plan.hard_constraints[0].field == "bus_count"
    assert {c.field for c in plan.soft_constraints} == {"vehicle_count", "person_count"}
    assert plan.active_filter_dict() == {
        "bus_count": {"min": 1}, "vehicle_count": {"min": 1}, "person_count": {"min": 1},
    }
