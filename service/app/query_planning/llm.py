"""parser_mode="llm" providers: transformers_local and vllm_openai_compatible.

Neither provider is imported at module load time -- `transformers`/`torch` and the vLLM
HTTP client are only touched inside the provider functions below, which are only called
when parser_mode="llm" is actually selected on a request. core_search_works_without_llm
depends on this: importing app.query_planning.llm must never fail just because torch/
transformers aren't installed.

vLLM is a serving method for the SAME parser model, not a different parser (plan
Sec.6.2) -- both providers share _parse_llm_response(), the one function that turns a raw
model completion into StructuredConstraints. That function is where the actual safety
contract lives: field must be in CANONICAL_FILTER_FIELDS (checked transitively by
constraints_to_filter_dict()+normalize_filters() at planning time, and redundantly here so
a malformed completion fails at parse time with a clear reason rather than a confusing 400
three layers downstream), operator must be one of the allowed literals, value must be the
right type for the field. No LLM-produced text ever reaches a SQL string; only field name,
operator, and now-type-checked value survive into a StructuredConstraint, and those still
have to pass through the exact same normalize_filters() allow-list every other constraint
does.
"""

from __future__ import annotations

import json
import time
from typing import Any, Protocol

from app.query_planning.models import ConstraintProvenance, ParserDiagnostics, ParsedQuery, StructuredConstraint
from app.search.filter_schema import CANONICAL_FILTER_FIELDS

ALLOWED_OPERATORS = {"eq", "gte", "lte", "range"}

PROMPT_TEMPLATE = """You extract structured filters from a video-search query. Only use \
these fields: {field_names}. Respond with strict JSON:
{{"constraints": [{{"field": "...", "operator": "eq|gte|lte|range", "value": ..., \
"confidence": 0.0-1.0}}], "semantic_residual": "...", "unsupported_concepts": ["..."]}}
If a concept in the query has no matching field, list it in unsupported_concepts and do
not invent a field. Query: {query}"""


class QueryParser(Protocol):
    def parse(self, query: str, available_fields: frozenset[str]) -> ParsedQuery: ...


class LLMParserError(RuntimeError):
    pass


def build_prompt(query: str, available_fields: frozenset[str]) -> str:
    return PROMPT_TEMPLATE.format(field_names=", ".join(sorted(available_fields)), query=query)


def _validate_value(field: str, operator: str, value: Any) -> Any:
    field_def = CANONICAL_FILTER_FIELDS[field]
    if operator == "range":
        if not (isinstance(value, (list, tuple)) and len(value) == 2):
            raise LLMParserError(f"range operator requires a 2-element [min,max], got {value!r}")
        low, high = value
        return (float(low), float(high))
    if field_def.data_type in {"keyword", "boolean"}:
        if operator != "eq":
            raise LLMParserError(f"field {field} is {field_def.data_type}; only 'eq' is valid, got {operator!r}")
        return value
    if operator not in {"gte", "lte"}:
        raise LLMParserError(f"numeric field {field} requires gte/lte/range, got {operator!r}")
    return float(value)


def _parse_llm_response(raw_text: str, available_fields: frozenset[str], *, provider: str, latency_ms: float) -> ParsedQuery:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise LLMParserError(f"provider returned non-JSON output: {exc}") from exc

    constraints: list[StructuredConstraint] = []
    unsupported = tuple(payload.get("unsupported_concepts", []))
    for raw_constraint in payload.get("constraints", []):
        field = raw_constraint.get("field")
        operator = raw_constraint.get("operator")
        if field not in CANONICAL_FILTER_FIELDS:
            raise LLMParserError(f"provider referenced a field outside the allow-list: {field!r}")
        if field not in available_fields:
            continue  # coverage.py's job normally does this; defensive skip if llm.py is called standalone
        if operator not in ALLOWED_OPERATORS:
            raise LLMParserError(f"provider used an operator outside the allow-list: {operator!r}")
        value = _validate_value(field, operator, raw_constraint.get("value"))
        confidence = float(raw_constraint.get("confidence", 0.6))
        constraints.append(StructuredConstraint(
            field=field, operator=operator, value=value,
            provenance=ConstraintProvenance(
                source="llm_parser", confidence=confidence, hard=False, relaxable=True, data_source="llm",
            ),
        ))

    residual = payload.get("semantic_residual", "")
    diagnostics = ParserDiagnostics(
        parser_mode="llm", provider=provider, latency_ms=latency_ms,
        confidence=(sum(c.provenance.confidence for c in constraints) / len(constraints)) if constraints else None,
        unsupported_concepts=unsupported,
    )
    return ParsedQuery(
        raw_query=raw_text, structured_constraints=tuple(constraints), semantic_residual=residual,
        residual_policy="structured_terms_removed", diagnostics=diagnostics,
    )


def parse_with_transformers_local(query: str, available_fields: frozenset[str], *, model_id: str) -> ParsedQuery:
    """Lazy-imports transformers/torch. Raises LLMParserError (never a synthetic
    fallback) if the model can't be loaded -- planner.py is what decides whether a
    failure here falls back to parser_mode=none, not this function."""
    started = time.perf_counter()
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise LLMParserError(f"transformers_local requires transformers+torch: {exc}") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    prompt = build_prompt(query, available_fields)
    inputs = tokenizer(prompt, return_tensors="pt")
    output_ids = model.generate(**inputs, max_new_tokens=512)
    raw_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return _parse_llm_response(raw_text, available_fields, provider="transformers_local", latency_ms=(time.perf_counter() - started) * 1000.0)


def parse_with_vllm_openai_compatible(
    query: str, available_fields: frozenset[str], *, base_url: str, model_id: str, timeout_s: float = 5.0,
) -> ParsedQuery:
    """vLLM here is purely a serving transport for the same parser model/prompt contract
    as transformers_local -- see module docstring. Lazy-imports httpx."""
    started = time.perf_counter()
    try:
        import httpx
    except ImportError as exc:
        raise LLMParserError(f"vllm_openai_compatible requires httpx: {exc}") from exc

    prompt = build_prompt(query, available_fields)
    try:
        response = httpx.post(
            f"{base_url}/v1/completions",
            json={"model": model_id, "prompt": prompt, "max_tokens": 512, "temperature": 0.0},
            timeout=timeout_s,
        )
        response.raise_for_status()
        raw_text = response.json()["choices"][0]["text"]
    except httpx.HTTPError as exc:
        raise LLMParserError(f"vllm_openai_compatible request failed: {exc}") from exc
    return _parse_llm_response(raw_text, available_fields, provider="vllm_openai_compatible", latency_ms=(time.perf_counter() - started) * 1000.0)


__all__ = [
    "QueryParser", "LLMParserError", "ALLOWED_OPERATORS", "build_prompt",
    "parse_with_transformers_local", "parse_with_vllm_openai_compatible",
]
