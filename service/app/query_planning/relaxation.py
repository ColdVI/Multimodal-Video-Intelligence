"""Provenance-aware filter relaxation ladder, per
docs/planning/ADVANCED_RETRIEVAL_FINAL_PLAN_v2.1.md Sec.7/13.

Ladder (manual/explicit-hard constraints are NEVER removed by any pass):

  Pass 1: everything active (hard + soft, as planned)
  Pass 2: drop detector_derived soft constraints below detector_confidence_threshold
  Pass 3: drop parser_derived (rules_parser/llm_parser) soft constraints below
          parser_confidence_threshold
  Pass 4: drop ALL remaining soft constraints (hard + trusted-telemetry only)
  Pass 5: drop trusted-telemetry hard constraints too (source="telemetry_derived") --
          semantic-only. Only runs if allow_semantic_only_fallback=True. Manual/
          explicit_request constraints are NEVER dropped by pass 5 either: "Manual
          filtre hicbir pass'te otomatik kaldirilmaz" (Sec.7) is unconditional, and
          "semantic-only" specifically means "drop everything except what the user
          explicitly typed," not "drop everything including that.\"

Triggered only when the search actually came back short (returned_count <
min_results) -- never runs preemptively. mode="diagnose_only" runs the exact same ladder
and reports what each pass WOULD have returned, but the actual response still uses Pass
1's real results -- nothing is auto-relaxed in that mode. mode="off" never enters this
module's loop at all (single pass, current behavior, unchanged).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from app.query_planning.models import RelaxationPolicy, StructuredConstraint

SearchFn = Callable[[tuple[StructuredConstraint, ...]], int]  # constraints -> returned_count


@dataclass(frozen=True)
class RelaxationPassResult:
    pass_number: int
    description: str
    active_constraints: tuple[StructuredConstraint, ...]
    relaxed_this_pass: tuple[StructuredConstraint, ...]
    returned_count: int


@dataclass(frozen=True)
class RelaxationOutcome:
    mode: str
    triggered: bool
    passes: tuple[RelaxationPassResult, ...]
    selected_pass: int  # 1 if never triggered/relaxed
    exact_filter_match: bool  # True iff the selected pass kept every original constraint
    stopped_reason: str  # "satisfied" | "passes_exhausted" | "timeout" | "not_triggered" | "no_semantic_fallback"

    def relaxed_constraints(self) -> tuple[StructuredConstraint, ...]:
        """All constraints relaxed on or before the selected pass, in the order they
        were dropped -- the plan's `relaxed_constraints` diagnostics field."""
        result: list[StructuredConstraint] = []
        for step in self.passes:
            if step.pass_number > self.selected_pass:
                break
            result.extend(step.relaxed_this_pass)
        return tuple(result)

    def as_diagnostics(self) -> dict:
        selected = next((p for p in self.passes if p.pass_number == self.selected_pass), self.passes[0])
        return {
            "mode": self.mode,
            "triggered": self.triggered,
            "passes_executed": len(self.passes),
            "initial_returned_count": self.passes[0].returned_count if self.passes else None,
            "final_returned_count": selected.returned_count,
            "relaxed_constraints": [c.as_dict() for c in self.relaxed_constraints()],
            "exact_filter_match": self.exact_filter_match,
            "stopped_reason": self.stopped_reason,
        }


def _split(constraints: tuple[StructuredConstraint, ...], predicate) -> tuple[tuple[StructuredConstraint, ...], tuple[StructuredConstraint, ...]]:
    kept = tuple(c for c in constraints if not predicate(c))
    dropped = tuple(c for c in constraints if predicate(c))
    return kept, dropped


def run_relaxation_ladder(
    hard_constraints: tuple[StructuredConstraint, ...],
    soft_constraints: tuple[StructuredConstraint, ...],
    policy: RelaxationPolicy,
    search_fn: SearchFn,
    *,
    detector_confidence_threshold: float = 0.7,
    parser_confidence_threshold: float = 0.7,
    known_pass1_count: int | None = None,
) -> RelaxationOutcome:
    """known_pass1_count lets a caller that already ran the Pass-1-equivalent search
    (engine.py always has -- it's the request's own primary search) skip a redundant
    duplicate query just to populate this module's diagnostics. Passing None (the
    default, and what every unit test above this line uses) makes this module call
    search_fn itself for Pass 1, e.g. for standalone/notebook use without an engine."""
    active = hard_constraints + soft_constraints
    if policy.mode == "off":
        count = known_pass1_count if known_pass1_count is not None else search_fn(active)
        pass1 = RelaxationPassResult(1, "all constraints active", active, (), count)
        return RelaxationOutcome("off", False, (pass1,), 1, True, "not_triggered")

    min_results = policy.min_results if policy.min_results is not None else 1
    started = time.perf_counter()
    passes: list[RelaxationPassResult] = []
    count = known_pass1_count if known_pass1_count is not None else search_fn(active)
    passes.append(RelaxationPassResult(1, "all constraints active", active, (), count))

    if count >= min_results:
        return RelaxationOutcome(policy.mode, False, tuple(passes), 1, True, "not_triggered")

    def _budget_ok(pass_number: int) -> tuple[bool, str]:
        if pass_number > policy.max_relaxation_passes:
            return False, "passes_exhausted"
        if (time.perf_counter() - started) * 1000.0 > policy.relaxation_timeout_ms:
            return False, "timeout"
        return True, ""

    remaining = soft_constraints
    step_definitions = [
        (2, "drop low-confidence detector-derived soft constraints",
         lambda c: c.provenance.relaxable and c.provenance.source == "detector_derived" and c.provenance.confidence < detector_confidence_threshold),
        (3, "drop low-confidence parser-derived soft constraints",
         lambda c: c.provenance.relaxable and c.provenance.source in ("rules_parser", "llm_parser") and c.provenance.confidence < parser_confidence_threshold),
        (4, "drop all remaining soft constraints (hard-only)",
         lambda c: c.provenance.relaxable),
    ]
    selected_pass = 1
    stopped_reason = "passes_exhausted"
    for pass_number, description, predicate in step_definitions:
        ok, reason = _budget_ok(pass_number)
        if not ok:
            stopped_reason = reason
            break
        remaining, dropped = _split(remaining, predicate)
        active = hard_constraints + remaining
        count = search_fn(active)
        passes.append(RelaxationPassResult(pass_number, description, active, dropped, count))
        selected_pass = pass_number
        if count >= min_results:
            stopped_reason = "satisfied"
            break
    else:
        # ran all 3 steps (2,3,4) without satisfying min_results
        if policy.allow_semantic_only_fallback:
            ok, reason = _budget_ok(5)
            if ok:
                # Manual/explicit_request constraints survive even here -- only
                # trusted-telemetry hard constraints are dropped. See module docstring.
                manual_only, trusted_telemetry = _split(hard_constraints, lambda c: c.provenance.source != "explicit_request")
                count = search_fn(manual_only)
                passes.append(RelaxationPassResult(
                    5, "semantic-only (trusted-telemetry hard constraints dropped; manual constraints kept)",
                    manual_only, trusted_telemetry, count,
                ))
                selected_pass = 5
                stopped_reason = "satisfied" if count >= min_results else "passes_exhausted"
            else:
                stopped_reason = reason
        else:
            stopped_reason = "no_semantic_fallback"

    exact_match = selected_pass == 1
    return RelaxationOutcome(policy.mode, True, tuple(passes), selected_pass, exact_match, stopped_reason)


__all__ = ["RelaxationPassResult", "RelaxationOutcome", "run_relaxation_ladder", "SearchFn"]
