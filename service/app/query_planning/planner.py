"""plan_query(): the one entry point engine.py calls. Dispatches parser_mode, applies the
dataset-coverage gate, and is the single place a parser failure gets converted into a safe
parser_mode="none"-equivalent fallback rather than a request-failing exception, per plan
Sec.4 ("Parser timeout veya hata durumunda: parser_mode=none davranışına güvenli biçimde
dön. Hata sessiz olmamalı; diagnostics'e yazılmalı.").
"""

from __future__ import annotations

import dataclasses

from app.query_planning import coverage, rules
from app.query_planning.llm import LLMParserError, parse_with_transformers_local, parse_with_vllm_openai_compatible
from app.query_planning.models import ParsedQuery, ParserMode, ParserProvider, QueryExecutionPlan, RelaxationPolicy


def _fallback_to_none(query: str, reason: str) -> ParsedQuery:
    parsed = ParsedQuery.unparsed(query)
    return dataclasses.replace(
        parsed,
        diagnostics=dataclasses.replace(parsed.diagnostics, fallback_triggered=True, fallback_reason=reason),
    )


def _apply_coverage_gate(parsed: ParsedQuery, available_fields: frozenset[str]) -> ParsedQuery:
    covered, uncovered = coverage.split_by_coverage(parsed.structured_constraints, available_fields)
    if not uncovered:
        return parsed
    return dataclasses.replace(
        parsed, structured_constraints=covered,
        diagnostics=dataclasses.replace(
            parsed.diagnostics,
            field_unavailable=parsed.diagnostics.field_unavailable + coverage.field_unavailable_notes(uncovered),
        ),
    )


def plan_query(
    query: str,
    available_fields: frozenset[str],
    *,
    parser_mode: ParserMode = "none",
    llm_provider: ParserProvider = "transformers_local",
    llm_model_id: str = "",
    llm_base_url: str = "",
    relaxation_policy: RelaxationPolicy | None = None,
) -> QueryExecutionPlan:
    if parser_mode == "none":
        parsed = ParsedQuery.unparsed(query)
    elif parser_mode == "rules":
        parsed = rules.parse(query)
    elif parser_mode == "llm":
        try:
            if llm_provider == "transformers_local":
                parsed = parse_with_transformers_local(query, available_fields, model_id=llm_model_id)
            elif llm_provider == "vllm_openai_compatible":
                parsed = parse_with_vllm_openai_compatible(query, available_fields, base_url=llm_base_url, model_id=llm_model_id)
            else:
                raise LLMParserError(f"unknown llm_provider: {llm_provider!r}")
        except LLMParserError as exc:
            parsed = _fallback_to_none(query, f"llm_provider={llm_provider}: {exc}")
    else:
        raise ValueError(f"unknown parser_mode: {parser_mode!r}")

    parsed = _apply_coverage_gate(parsed, available_fields)
    return QueryExecutionPlan.from_parsed_query(parsed, relaxation_policy or RelaxationPolicy())


__all__ = ["plan_query"]
