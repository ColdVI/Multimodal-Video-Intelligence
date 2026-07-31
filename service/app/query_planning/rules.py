"""parser_mode="rules" provider: deterministic keyword matching against the ontology.

No model weights, no network, no lazy-imported heavy dependency -- this provider is
always available and is what parser_mode falls back to on an llm provider timeout/error
(see planner.py). Confidence is fixed and high (rules either match or they don't; there is
no probabilistic signal to report), which also means rules constraints are relaxable in
the ladder (plan Sec.7's Pass 3) but are trusted more than an LLM's own soft constraints
in practice, since rules-parser output is exact-match, not inferred.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass

from app.query_planning.models import ConstraintProvenance, ParserDiagnostics, ParsedQuery, StructuredConstraint
from app.query_planning.ontology import ONTOLOGY, OntologyConcept, ontology_widened_notes, resolve_specificity

RULES_CONFIDENCE = 0.9

_NUMERIC_PATTERN = re.compile(
    r"(?:en az|at least)\s+(\d+)\s+(\S+)", re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _synonym_pattern(concept: OntologyConcept) -> re.Pattern[str]:
    words = [re.escape(word) for word in (*concept.synonyms_tr, *concept.synonyms_en) if word]
    return re.compile(r"\b(?:" + "|".join(words) + r")\b", re.IGNORECASE) if words else re.compile(r"(?!)")


_CONCEPT_PATTERNS: dict[str, re.Pattern[str]] = {
    concept_id: _synonym_pattern(concept) for concept_id, concept in ONTOLOGY.items()
}


@dataclass(frozen=True)
class _NumericOverride:
    concept_id: str
    minimum: int


def _find_numeric_overrides(normalized_query: str) -> list[_NumericOverride]:
    overrides = []
    for match in _NUMERIC_PATTERN.finditer(normalized_query):
        count_str, trailing_word = match.group(1), match.group(2)
        for concept_id, concept in ONTOLOGY.items():
            for synonym in (*concept.synonyms_tr, *concept.synonyms_en):
                if synonym and synonym.lower() in trailing_word.lower():
                    overrides.append(_NumericOverride(concept_id, int(count_str)))
                    break
    return overrides


def parse(query: str) -> ParsedQuery:
    started = time.perf_counter()
    normalized = _normalize(query)
    matched_concepts = [
        concept for concept_id, concept in ONTOLOGY.items() if _CONCEPT_PATTERNS[concept_id].search(normalized)
    ]
    resolution = resolve_specificity(matched_concepts)
    overrides = {override.concept_id: override.minimum for override in _find_numeric_overrides(normalized)}

    constraints = []
    for concept in resolution.selected:
        minimum = overrides.get(concept.concept_id, 1)
        constraints.append(StructuredConstraint(
            field=concept.canonical_field, operator="gte", value=minimum,
            provenance=ConstraintProvenance(
                source="rules_parser", confidence=RULES_CONFIDENCE, hard=False, relaxable=True,
                data_source="ontology",
            ),
        ))

    diagnostics = ParserDiagnostics(
        parser_mode="rules", provider="rules", latency_ms=(time.perf_counter() - started) * 1000.0,
        confidence=RULES_CONFIDENCE if constraints else None,
        ontology_widened=ontology_widened_notes(resolution),
        notes=tuple(f"elided_lower_specificity:{c.concept_id}" for c in resolution.elided),
    )
    return ParsedQuery(
        raw_query=query, structured_constraints=tuple(constraints), semantic_residual=query,
        residual_policy="full_query", diagnostics=diagnostics,
    )


__all__ = ["parse", "RULES_CONFIDENCE"]
