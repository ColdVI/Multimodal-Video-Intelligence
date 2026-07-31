"""Production contracts for query planning: ParsedQuery, StructuredConstraint,
ConstraintProvenance, ParserDiagnostics, RelaxationPolicy, QueryExecutionPlan.

DatasetFilterSchema is deliberately NOT redefined here -- app.search.filter_schema already
provides it (CANONICAL_FILTER_FIELDS, manifest_filter_fields(), FilterField) and is wired
into GET /datasets/{id}/filter-schema. Constraint.field values in this module are always
validated against that same schema (see coverage.py), never a private copy of it.

Safety invariant this whole package exists to protect: a StructuredConstraint's field is
always one of app.search.filter_schema.CANONICAL_FILTER_FIELDS' keys (a closed allow-list),
its value always passes through app.search.pushdown.normalize_filters() before reaching any
backend, and no parser/ontology/relaxation code ever builds a SQL fragment itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

ConstraintSource = Literal[
    "explicit_request", "rules_parser", "llm_parser", "detector_derived", "telemetry_derived",
]
ConstraintOperator = Literal["eq", "gte", "lte", "range"]
ParserMode = Literal["none", "rules", "llm"]
ParserProvider = Literal["rules", "transformers_local", "vllm_openai_compatible"]
RelaxationMode = Literal["off", "diagnose_only", "auto_soft"]
SemanticResidualPolicy = Literal["full_query", "structured_terms_removed"]


@dataclass(frozen=True)
class ConstraintProvenance:
    """source=explicit_request constraints are always hard=True/relaxable=False by
    construction (see StructuredConstraint.explicit()) -- a manual/API filter is never
    silently dropped by the relaxation ladder, matching the plan's hard requirement."""

    source: ConstraintSource
    confidence: float = 1.0
    hard: bool = True
    relaxable: bool = False
    data_source: str | None = None  # e.g. "yolo_detector", "user", "telemetry_registry"

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")
        if self.source == "explicit_request" and (self.hard is not True or self.relaxable is not False):
            raise ValueError("explicit_request provenance must be hard=True, relaxable=False")


@dataclass(frozen=True)
class StructuredConstraint:
    field: str
    operator: ConstraintOperator
    value: Any
    provenance: ConstraintProvenance

    @classmethod
    def explicit(cls, field: str, operator: ConstraintOperator, value: Any, *, data_source: str = "user") -> "StructuredConstraint":
        return cls(field, operator, value, ConstraintProvenance(
            source="explicit_request", confidence=1.0, hard=True, relaxable=False, data_source=data_source,
        ))

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field, "operator": self.operator, "value": self.value,
            "source": self.provenance.source, "data_source": self.provenance.data_source,
            "confidence": self.provenance.confidence, "hard": self.provenance.hard,
            "relaxable": self.provenance.relaxable,
        }


def constraints_to_filter_dict(constraints: "tuple[StructuredConstraint, ...] | list[StructuredConstraint]") -> dict[str, Any]:
    """Bridge from the provenance-aware constraint list to the plain
    {field: value | {"min":..,"max":..}} shape app.search.pushdown.normalize_filters()
    already accepts. Multiple constraints on the same field (e.g. a detector-derived gte
    plus a parser-derived lte) are merged conservatively: the tightest bound wins. This
    never emits SQL -- normalize_filters() + Predicate + the backend's parametrized query
    builder are still the only path from here to a database."""
    merged: dict[str, dict[str, Any]] = {}
    equals: dict[str, Any] = {}
    for constraint in constraints:
        if constraint.operator == "eq":
            equals[constraint.field] = constraint.value
            continue
        bucket = merged.setdefault(constraint.field, {})
        if constraint.operator == "gte":
            bucket["min"] = constraint.value if "min" not in bucket else max(bucket["min"], constraint.value)
        elif constraint.operator == "lte":
            bucket["max"] = constraint.value if "max" not in bucket else min(bucket["max"], constraint.value)
        elif constraint.operator == "range":
            low, high = constraint.value
            bucket["min"] = low if "min" not in bucket else max(bucket["min"], low)
            bucket["max"] = high if "max" not in bucket else min(bucket["max"], high)
    result: dict[str, Any] = dict(equals)
    result.update(merged)
    return result


@dataclass(frozen=True)
class ParserDiagnostics:
    parser_mode: ParserMode
    provider: ParserProvider | None = None
    latency_ms: float | None = None
    confidence: float | None = None
    fallback_triggered: bool = False
    fallback_reason: str | None = None
    unsupported_concepts: tuple[str, ...] = ()
    field_unavailable: tuple[str, ...] = ()
    ontology_widened: tuple[str, ...] = ()  # e.g. "truck->vehicle_count"
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "parser_mode": self.parser_mode, "provider": self.provider,
            "latency_ms": self.latency_ms, "confidence": self.confidence,
            "fallback_triggered": self.fallback_triggered, "fallback_reason": self.fallback_reason,
            "unsupported_concepts": list(self.unsupported_concepts),
            "field_unavailable": list(self.field_unavailable),
            "ontology_widened": list(self.ontology_widened),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ParsedQuery:
    raw_query: str
    structured_constraints: tuple[StructuredConstraint, ...]
    semantic_residual: str
    residual_policy: SemanticResidualPolicy
    diagnostics: ParserDiagnostics

    @classmethod
    def unparsed(cls, raw_query: str) -> "ParsedQuery":
        """parser_mode=none -- the existing, default behavior: query goes to embedding
        as-is, structured filters come only from the request's own metadata/telemetry
        filters (handled entirely outside this package, unchanged)."""
        return cls(
            raw_query=raw_query, structured_constraints=(), semantic_residual=raw_query,
            residual_policy="full_query", diagnostics=ParserDiagnostics(parser_mode="none"),
        )


@dataclass(frozen=True)
class RelaxationPolicy:
    mode: RelaxationMode = "off"
    min_results: int | None = None
    max_relaxation_passes: int = 4
    relaxation_timeout_ms: float = 2000.0
    allow_semantic_only_fallback: bool = False

    def __post_init__(self) -> None:
        if self.max_relaxation_passes < 1:
            raise ValueError("max_relaxation_passes must be >= 1")
        if self.relaxation_timeout_ms <= 0:
            raise ValueError("relaxation_timeout_ms must be positive")
        if self.min_results is not None and self.min_results < 0:
            raise ValueError("min_results must be >= 0")


@dataclass(frozen=True)
class QueryExecutionPlan:
    """The thing engine.py actually consumes: a parsed query plus the relaxation policy
    that governs it, with hard/soft constraints already split out so the relaxation
    ladder (relaxation.py) never has to re-inspect provenance -- it just walks
    soft_constraints in confidence order."""

    parsed_query: ParsedQuery
    relaxation_policy: RelaxationPolicy
    hard_constraints: tuple[StructuredConstraint, ...]
    soft_constraints: tuple[StructuredConstraint, ...]

    @classmethod
    def from_parsed_query(cls, parsed: ParsedQuery, relaxation_policy: RelaxationPolicy) -> "QueryExecutionPlan":
        hard = tuple(c for c in parsed.structured_constraints if c.provenance.hard)
        soft = tuple(c for c in parsed.structured_constraints if not c.provenance.hard)
        return cls(parsed, relaxation_policy, hard, soft)

    def active_filter_dict(self, constraints: "tuple[StructuredConstraint, ...] | None" = None) -> dict[str, Any]:
        return constraints_to_filter_dict(self.hard_constraints + self.soft_constraints if constraints is None else constraints)


__all__ = [
    "ConstraintSource", "ConstraintOperator", "ParserMode", "ParserProvider",
    "RelaxationMode", "SemanticResidualPolicy",
    "ConstraintProvenance", "StructuredConstraint", "constraints_to_filter_dict",
    "ParserDiagnostics", "ParsedQuery", "RelaxationPolicy", "QueryExecutionPlan",
]
