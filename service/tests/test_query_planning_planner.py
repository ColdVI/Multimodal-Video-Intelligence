from __future__ import annotations

import app.query_planning.llm as llm_module
import app.query_planning.planner as planner_module
from app.query_planning.models import (
    ConstraintProvenance, ParsedQuery, ParserDiagnostics, RelaxationPolicy, StructuredConstraint,
)
from app.query_planning.planner import plan_query

ALL_FIELDS = frozenset({"vehicle_count", "bus_count", "person_count", "is_night", "altitude_m"})


def test_none_mode_produces_unparsed_query_with_no_constraints():
    plan = plan_query("kalabalık trafik", ALL_FIELDS, parser_mode="none")
    assert plan.parsed_query.structured_constraints == ()
    assert plan.parsed_query.diagnostics.parser_mode == "none"
    assert plan.hard_constraints == ()
    assert plan.soft_constraints == ()


def test_rules_mode_delegates_to_rules_parser():
    plan = plan_query("otobüs bulunan sahne", ALL_FIELDS, parser_mode="rules")
    assert [c.field for c in plan.soft_constraints] == ["bus_count"]
    assert plan.parsed_query.diagnostics.parser_mode == "rules"


def test_unknown_parser_mode_raises():
    import pytest
    with pytest.raises(ValueError, match="unknown parser_mode"):
        plan_query("q", ALL_FIELDS, parser_mode="not_a_mode")  # type: ignore[arg-type]


def test_llm_mode_success_path_uses_the_selected_provider(monkeypatch):
    def fake_transformers(query, available_fields, *, model_id):
        return ParsedQuery(
            raw_query=query, semantic_residual="", residual_policy="structured_terms_removed",
            structured_constraints=(StructuredConstraint(
                "vehicle_count", "gte", 1, ConstraintProvenance(source="llm_parser", confidence=0.7, hard=False, relaxable=True),
            ),),
            diagnostics=ParserDiagnostics(parser_mode="llm", provider="transformers_local"),
        )

    monkeypatch.setattr(planner_module, "parse_with_transformers_local", fake_transformers)
    plan = plan_query("q", ALL_FIELDS, parser_mode="llm", llm_provider="transformers_local", llm_model_id="fake")
    assert [c.field for c in plan.soft_constraints] == ["vehicle_count"]
    assert plan.parsed_query.diagnostics.fallback_triggered is False


def test_llm_mode_failure_falls_back_to_none_with_reason_recorded(monkeypatch):
    def raising_provider(query, available_fields, *, model_id):
        raise llm_module.LLMParserError("model unavailable in this environment")

    monkeypatch.setattr(planner_module, "parse_with_transformers_local", raising_provider)
    plan = plan_query("otobüs", ALL_FIELDS, parser_mode="llm", llm_provider="transformers_local", llm_model_id="fake")
    assert plan.parsed_query.structured_constraints == ()
    assert plan.parsed_query.semantic_residual == "otobüs"  # behaves exactly like parser_mode=none
    assert plan.parsed_query.diagnostics.fallback_triggered is True
    assert "model unavailable" in plan.parsed_query.diagnostics.fallback_reason


def test_llm_mode_unknown_provider_falls_back_safely_too(monkeypatch):
    plan = plan_query("q", ALL_FIELDS, parser_mode="llm", llm_provider="not_a_real_provider")  # type: ignore[arg-type]
    assert plan.parsed_query.diagnostics.fallback_triggered is True
    assert "unknown llm_provider" in plan.parsed_query.diagnostics.fallback_reason


def test_coverage_gate_drops_constraint_for_field_not_in_available_fields_and_records_it():
    plan = plan_query("otobüs", frozenset({"person_count"}), parser_mode="rules")
    assert plan.parsed_query.structured_constraints == ()
    assert plan.parsed_query.diagnostics.field_unavailable == ("bus_count",)


def test_relaxation_policy_is_threaded_through():
    policy = RelaxationPolicy(mode="auto_soft", min_results=5)
    plan = plan_query("q", ALL_FIELDS, parser_mode="none", relaxation_policy=policy)
    assert plan.relaxation_policy.mode == "auto_soft"
    assert plan.relaxation_policy.min_results == 5


def test_default_relaxation_policy_is_off():
    plan = plan_query("q", ALL_FIELDS, parser_mode="none")
    assert plan.relaxation_policy.mode == "off"
