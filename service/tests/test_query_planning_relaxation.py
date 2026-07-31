from __future__ import annotations

import pytest

from app.query_planning.models import ConstraintProvenance, RelaxationPolicy, StructuredConstraint
from app.query_planning.relaxation import run_relaxation_ladder

HARD = (StructuredConstraint.explicit("bus_count", "gte", 1),)
DETECTOR_LOW = StructuredConstraint("person_count", "gte", 1, ConstraintProvenance(
    source="detector_derived", confidence=0.3, hard=False, relaxable=True,
))
DETECTOR_HIGH = StructuredConstraint("vehicle_count", "gte", 1, ConstraintProvenance(
    source="detector_derived", confidence=0.95, hard=False, relaxable=True,
))
PARSER_LOW = StructuredConstraint("is_night", "eq", True, ConstraintProvenance(
    source="llm_parser", confidence=0.4, hard=False, relaxable=True,
))


def test_mode_off_runs_a_single_pass_and_never_relaxes():
    outcome = run_relaxation_ladder(HARD, (DETECTOR_LOW,), RelaxationPolicy(mode="off"), lambda c: 0)
    assert outcome.triggered is False
    assert len(outcome.passes) == 1
    assert outcome.exact_filter_match is True
    assert outcome.relaxed_constraints() == ()


def test_not_triggered_when_pass_1_already_satisfies_min_results():
    outcome = run_relaxation_ladder(
        HARD, (DETECTOR_HIGH,), RelaxationPolicy(mode="auto_soft", min_results=5),
        lambda c: 10,
    )
    assert outcome.triggered is False
    assert outcome.selected_pass == 1
    assert outcome.stopped_reason == "not_triggered"


def test_manual_constraints_are_present_in_every_pass_including_semantic_only():
    """The one safety invariant this whole module exists to protect ("Manual filtre
    hicbir pass'te otomatik kaldirilmaz", plan Sec.7): an explicit_request constraint is
    never in any pass's relaxed_this_pass, including pass 5 -- even though pass 5 does
    drop OTHER hard constraints (trusted telemetry)."""
    outcome = run_relaxation_ladder(
        HARD, (DETECTOR_LOW, PARSER_LOW),
        RelaxationPolicy(mode="auto_soft", min_results=100, allow_semantic_only_fallback=True),
        lambda c: 0,  # never satisfied -> walks the whole ladder including pass 5
    )
    for step in outcome.passes:
        assert set(HARD) <= set(step.active_constraints)
        assert set(HARD).isdisjoint(step.relaxed_this_pass)


def test_pass_2_drops_only_low_confidence_detector_constraints():
    outcome = run_relaxation_ladder(
        HARD, (DETECTOR_LOW, DETECTOR_HIGH),
        RelaxationPolicy(mode="auto_soft", min_results=5),
        lambda c: 5 if DETECTOR_LOW not in c else 0,
    )
    assert outcome.selected_pass == 2
    assert outcome.passes[1].relaxed_this_pass == (DETECTOR_LOW,)
    assert DETECTOR_HIGH in outcome.passes[1].active_constraints


def test_pass_3_drops_low_confidence_parser_constraints_after_pass_2():
    def fake_search(constraints):
        if PARSER_LOW in constraints:
            return 0
        return 5

    outcome = run_relaxation_ladder(
        HARD, (DETECTOR_LOW, PARSER_LOW),
        RelaxationPolicy(mode="auto_soft", min_results=5),
        fake_search,
    )
    assert outcome.selected_pass == 3
    relaxed = outcome.relaxed_constraints()
    assert DETECTOR_LOW in relaxed
    assert PARSER_LOW in relaxed


def test_pass_4_drops_all_remaining_soft_constraints():
    high_confidence_soft = StructuredConstraint("vehicle_count", "gte", 1, ConstraintProvenance(
        source="rules_parser", confidence=1.0, hard=False, relaxable=True,
    ))
    outcome = run_relaxation_ladder(
        HARD, (high_confidence_soft,), RelaxationPolicy(mode="auto_soft", min_results=5),
        lambda c: 5 if high_confidence_soft not in c else 0,
    )
    assert outcome.selected_pass == 4
    assert high_confidence_soft in outcome.relaxed_constraints()
    assert outcome.passes[-1].active_constraints == HARD


def test_pass_5_only_runs_when_allow_semantic_only_fallback_is_true():
    outcome = run_relaxation_ladder(
        HARD, (DETECTOR_LOW,), RelaxationPolicy(mode="auto_soft", min_results=100, allow_semantic_only_fallback=False),
        lambda c: 0,
    )
    assert outcome.selected_pass == 4  # stops at hard-only, never tries semantic-only
    assert outcome.stopped_reason == "no_semantic_fallback"
    assert all(step.pass_number != 5 for step in outcome.passes)


def test_pass_5_drops_trusted_telemetry_but_keeps_manual_constraints():
    trusted_telemetry = StructuredConstraint("altitude_m", "lte", 100.0, ConstraintProvenance(
        source="telemetry_derived", confidence=1.0, hard=True, relaxable=False,
    ))
    hard = HARD + (trusted_telemetry,)
    outcome = run_relaxation_ladder(
        hard, (DETECTOR_LOW,),
        RelaxationPolicy(mode="auto_soft", min_results=1, allow_semantic_only_fallback=True),
        lambda c: 1 if trusted_telemetry not in c else 0,
    )
    assert outcome.selected_pass == 5
    assert outcome.passes[-1].active_constraints == HARD  # manual survives, telemetry dropped
    assert trusted_telemetry in outcome.passes[-1].relaxed_this_pass
    assert HARD[0] not in outcome.passes[-1].relaxed_this_pass
    assert outcome.exact_filter_match is False


def test_pass_5_is_a_no_op_relative_to_pass_4_when_there_is_no_trusted_telemetry_to_drop():
    """HARD contains only an explicit_request constraint -- pass 5 has nothing eligible
    to drop, so it must not touch it, and the ladder correctly reports passes_exhausted
    rather than fabricating a change."""
    outcome = run_relaxation_ladder(
        HARD, (DETECTOR_LOW,),
        RelaxationPolicy(mode="auto_soft", min_results=100, allow_semantic_only_fallback=True),
        lambda c: 0,
    )
    assert outcome.selected_pass == 5
    assert outcome.passes[-1].active_constraints == HARD
    assert outcome.passes[-1].relaxed_this_pass == ()
    assert outcome.stopped_reason == "passes_exhausted"


def test_max_relaxation_passes_budget_stops_the_ladder_early():
    outcome = run_relaxation_ladder(
        HARD, (DETECTOR_LOW, PARSER_LOW),
        RelaxationPolicy(mode="auto_soft", min_results=100, max_relaxation_passes=2),
        lambda c: 0,
    )
    assert outcome.stopped_reason == "passes_exhausted"
    assert max(step.pass_number for step in outcome.passes) == 2


def test_timeout_budget_stops_the_ladder_early():
    import time

    def slow_search(constraints):
        time.sleep(0.02)
        return 0

    outcome = run_relaxation_ladder(
        HARD, (DETECTOR_LOW, PARSER_LOW),
        RelaxationPolicy(mode="auto_soft", min_results=100, relaxation_timeout_ms=15.0),
        slow_search,
    )
    assert outcome.stopped_reason == "timeout"


def test_known_pass1_count_skips_a_redundant_search_call():
    calls = []

    def counting_search(constraints):
        calls.append(constraints)
        return 10

    outcome = run_relaxation_ladder(
        HARD, (DETECTOR_LOW,), RelaxationPolicy(mode="auto_soft", min_results=5),
        counting_search, known_pass1_count=10,
    )
    assert calls == []  # never called -- satisfied immediately using the known count
    assert outcome.passes[0].returned_count == 10
    assert outcome.triggered is False


def test_known_pass1_count_also_applies_to_mode_off():
    def failing_search(constraints):
        raise AssertionError("mode=off must not call search_fn when the count is already known")

    outcome = run_relaxation_ladder(
        HARD, (), RelaxationPolicy(mode="off"), failing_search, known_pass1_count=7,
    )
    assert outcome.passes[0].returned_count == 7
    assert outcome.triggered is False


def test_as_diagnostics_matches_the_plan_example_shape():
    outcome = run_relaxation_ladder(
        HARD, (DETECTOR_LOW,), RelaxationPolicy(mode="auto_soft", min_results=5),
        lambda c: 10 if DETECTOR_LOW not in c else 0,
    )
    diagnostics = outcome.as_diagnostics()
    assert set(diagnostics) == {
        "mode", "triggered", "passes_executed", "initial_returned_count",
        "final_returned_count", "relaxed_constraints", "exact_filter_match", "stopped_reason",
    }
    assert diagnostics["triggered"] is True
    assert diagnostics["final_returned_count"] == 10
