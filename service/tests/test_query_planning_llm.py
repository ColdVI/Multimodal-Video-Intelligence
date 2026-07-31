from __future__ import annotations

import json

import pytest

from app.query_planning.llm import LLMParserError, _parse_llm_response, build_prompt

AVAILABLE = frozenset({"vehicle_count", "bus_count", "person_count", "is_night", "altitude_m"})


def _response(**overrides) -> str:
    payload = {
        "constraints": [{"field": "vehicle_count", "operator": "gte", "value": 1, "confidence": 0.8}],
        "semantic_residual": "traffic",
        "unsupported_concepts": [],
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_valid_response_produces_a_structured_constraint_with_llm_provenance():
    parsed = _parse_llm_response(_response(), AVAILABLE, provider="transformers_local", latency_ms=12.0)
    assert len(parsed.structured_constraints) == 1
    constraint = parsed.structured_constraints[0]
    assert constraint.field == "vehicle_count"
    assert constraint.value == 1.0
    assert constraint.provenance.source == "llm_parser"
    assert constraint.provenance.hard is False
    assert constraint.provenance.relaxable is True
    assert parsed.semantic_residual == "traffic"
    assert parsed.diagnostics.provider == "transformers_local"


def test_field_outside_allow_list_is_rejected_not_silently_widened():
    """Security-critical: an LLM hallucinating a field name must never reach
    normalize_filters() -- it must fail loudly right here."""
    raw = _response(constraints=[{"field": "'; DROP TABLE segments; --", "operator": "eq", "value": 1}])
    with pytest.raises(LLMParserError, match="outside the allow-list"):
        _parse_llm_response(raw, AVAILABLE, provider="transformers_local", latency_ms=1.0)


def test_operator_outside_allow_list_is_rejected():
    raw = _response(constraints=[{"field": "vehicle_count", "operator": "DROP", "value": 1}])
    with pytest.raises(LLMParserError, match="outside the allow-list"):
        _parse_llm_response(raw, AVAILABLE, provider="transformers_local", latency_ms=1.0)


def test_non_json_response_is_rejected_cleanly():
    with pytest.raises(LLMParserError, match="non-JSON"):
        _parse_llm_response("this is not json {{{", AVAILABLE, provider="transformers_local", latency_ms=1.0)


def test_range_operator_requires_two_element_value():
    raw = _response(constraints=[{"field": "altitude_m", "operator": "range", "value": 5}])
    with pytest.raises(LLMParserError, match="range operator requires"):
        _parse_llm_response(raw, AVAILABLE, provider="transformers_local", latency_ms=1.0)


def test_range_operator_with_valid_bounds_succeeds():
    raw = _response(constraints=[{"field": "altitude_m", "operator": "range", "value": [10, 50]}])
    parsed = _parse_llm_response(raw, AVAILABLE, provider="transformers_local", latency_ms=1.0)
    assert parsed.structured_constraints[0].value == (10.0, 50.0)


def test_keyword_or_boolean_field_rejects_non_eq_operator():
    raw = _response(constraints=[{"field": "is_night", "operator": "gte", "value": 1}])
    with pytest.raises(LLMParserError, match="only 'eq' is valid"):
        _parse_llm_response(raw, AVAILABLE, provider="transformers_local", latency_ms=1.0)


def test_boolean_field_with_eq_operator_succeeds():
    raw = _response(constraints=[{"field": "is_night", "operator": "eq", "value": True}])
    parsed = _parse_llm_response(raw, AVAILABLE, provider="transformers_local", latency_ms=1.0)
    assert parsed.structured_constraints[0].value is True


def test_field_not_in_available_fields_is_defensively_skipped_not_raised():
    raw = _response(constraints=[{"field": "gimbal_pitch", "operator": "gte", "value": 1}])
    parsed = _parse_llm_response(raw, AVAILABLE, provider="transformers_local", latency_ms=1.0)
    assert parsed.structured_constraints == ()


def test_unsupported_concepts_are_carried_into_diagnostics():
    raw = _response(unsupported_concepts=["velocity_mps"])
    parsed = _parse_llm_response(raw, AVAILABLE, provider="transformers_local", latency_ms=1.0)
    assert parsed.diagnostics.unsupported_concepts == ("velocity_mps",)


def test_build_prompt_lists_only_available_fields_sorted():
    prompt = build_prompt("kalabalık trafik", frozenset({"bus_count", "altitude_m"}))
    assert "altitude_m, bus_count" in prompt
    assert "kalabalık trafik" in prompt
