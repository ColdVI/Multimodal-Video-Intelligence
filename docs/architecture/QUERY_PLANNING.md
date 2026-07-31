# Query planning architecture

Implements `docs/planning/ADVANCED_RETRIEVAL_FINAL_PLAN_v2.1.md` Sec.2-7. All code lives
under `service/app/query_planning/`.

## Why this exists

The research-plane parser (`search/parser.py`) produces `car_count`/`truck_count`, which
do not exist in the product schema (`app.search.filter_schema.CANONICAL_FILTER_FIELDS`
only has `vehicle_count`/`bus_count`). Wiring a parser straight to the product schema
without an ontology layer would 400 on the first "araba" query, because
`app.search.pushdown.normalize_filters()` rejects unknown fields by design. Ontology runs
*before* the parser can be considered correct at all.

## Data flow

```
query text
  -> app.query_planning.planner.plan_query(query, available_fields, parser_mode=...)
       -> rules.parse() | llm.parse_with_transformers_local() | llm.parse_with_vllm_openai_compatible()
       -> coverage.split_by_coverage()  # drops constraints for fields unavailable on this dataset/run
  -> QueryExecutionPlan(parsed_query, relaxation_policy, hard_constraints, soft_constraints)
  -> models.constraints_to_filter_dict(...)  # the ONLY bridge to app.search.pushdown.normalize_filters()
  -> merged with request.metadata_filters/telemetry_filters (explicit always wins on a field collision)
  -> service/app/search/engine.py's existing candidate-resolution/native-pushdown path (unchanged)
```

No parser/ontology/relaxation code ever builds a SQL fragment. `constraints_to_filter_dict()`
produces the same `{field: value | {"min":..,"max":..}}` shape `normalize_filters()` already
accepts; from there the request follows the exact same parametrized-query path every other
filter does.

## Modules

| Module | Responsibility |
|---|---|
| `models.py` | `ParsedQuery`, `StructuredConstraint`, `ConstraintProvenance`, `ParserDiagnostics`, `RelaxationPolicy`, `QueryExecutionPlan`, `constraints_to_filter_dict()` |
| `ontology.py` | `ONTOLOGY` concept catalog, `resolve_specificity()` (most-specific concept per family wins; "kamyon"/"truck" widens to `vehicle_count` with an explicit diagnostic since no `truck_count` column exists) |
| `coverage.py` | `available_field_names(dataset_id)` (mirrors `GET /datasets/{id}/filter-schema`'s field-availability logic), `split_by_coverage()` |
| `rules.py` | Deterministic keyword provider for `parser_mode="rules"` -- no model, no network, always available |
| `llm.py` | `parse_with_transformers_local()` / `parse_with_vllm_openai_compatible()` for `parser_mode="llm"`. Both share `_parse_llm_response()`, the actual safety boundary: field must be in `CANONICAL_FILTER_FIELDS`, operator in `{eq,gte,lte,range}`, value type-checked -- a hallucinated field/operator raises `LLMParserError` immediately. |
| `planner.py` | `plan_query()`, the single entry point; converts any `LLMParserError` into a safe `parser_mode="none"`-equivalent fallback with the reason recorded in `diagnostics.parser.fallback_reason` |
| `relaxation.py` | `run_relaxation_ladder()`, Pass 1-5 (see below) |

## Modes and flags

Env default -> request-level override (same pattern as the existing
`FILTER_EXECUTION_MODE`/`filter_execution_mode`):

| Env var | Default | Request field |
|---|---|---|
| `QUERY_PARSER_MODE` | `none` | `parser_mode: "none" \| "rules" \| "llm"` |
| `QUERY_PARSER_LLM_PROVIDER` | `transformers_local` | (config-only; no per-request override yet) |
| `FILTER_RELAXATION_MODE` | `off` | `filter_relaxation_mode: "off" \| "diagnose_only" \| "auto_soft"` |
| (n/a) | `top_k` | `min_results: int \| None` |

`parser_mode="none"` (the default) skips `plan_query()` entirely -- zero extra Postgres
round-trip, zero behavior change from before this work existed.

## Relaxation ladder (Sec.7)

Only ever relaxes `plan.soft_constraints` (parser-derived). Explicit request filters are
outside the ladder's candidate pool entirely -- "manual filter never removed" holds by
construction, not by a check inside `relaxation.py`.

1. Everything active (hard + soft)
2. Drop low-confidence `detector_derived` soft constraints (default threshold 0.7)
3. Drop low-confidence `rules_parser`/`llm_parser` soft constraints (default threshold 0.7)
4. Drop all remaining soft constraints (hard-only)
5. Drop trusted-telemetry hard constraints too (`source="telemetry_derived"`) -- **manual
   (`explicit_request`) constraints still survive even this pass**; only runs if
   `allow_semantic_only_fallback=true`

Only triggers when the primary search actually returned fewer than `min_results` rows.
`filter_relaxation_mode="diagnose_only"` runs the identical ladder and reports what
`auto_soft` would have done, without swapping the relaxed results into the response.

## Response contract additions

All additive; no existing field removed or renamed.

```json
{
  "diagnostics": {
    "parser": {"parser_mode": "rules", "provider": "rules", "fallback_triggered": false, "field_unavailable": [], "ontology_widened": []},
    "filter_relaxation": {"mode": "auto_soft", "triggered": true, "passes_executed": 3, "relaxed_constraints": [...], "exact_filter_match": false, "stopped_reason": "satisfied"}
  },
  "execution_policy": {"parser_mode": "rules", "filter_relaxation_mode": "auto_soft", "adaptive_exact_rerank": false, "vlm_rerank_mode": "off"}
}
```

## Verified evidence

- `service/tests/test_query_planning_*.py` (unit, pure logic) and
  `test_query_planning_engine_integration.py` /
  `test_query_planning_relaxation_integration.py` (live, against the real ClickHouse +
  Postgres containers -- "otobüs ve yaya" through `parser_mode=rules` correctly resolves
  to `bus_count>=1 AND person_count>=1`, and `auto_soft` relaxation rescues an
  underfilled result by dropping the lower-priority constraint first).
- `llm.py`'s two model-serving providers are not exercised end-to-end this session (no
  LLM available); their shared validator is fully unit-tested including an injection
  attempt (`test_query_planning_llm.py::test_field_outside_allow_list_is_rejected_not_silently_widened`).
