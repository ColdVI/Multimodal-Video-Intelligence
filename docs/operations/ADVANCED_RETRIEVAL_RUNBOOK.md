# Advanced retrieval operations runbook

All new capabilities from `docs/planning/ADVANCED_RETRIEVAL_FINAL_PLAN_v2.1.md` are
default-off / behavior-preserving. This runbook covers how to turn each on and verify it.

## Enabling the query parser

```bash
# Env default (affects every request unless overridden):
QUERY_PARSER_MODE=rules   # or "llm" (needs QUERY_PARSER_LLM_MODEL_ID / _BASE_URL configured)
```

Per-request override: `{"parser_mode": "rules", ...}` in the `POST /search` body.
Verify: `response.diagnostics.parser.parser_mode` and `response.execution_policy.parser_mode`
should reflect the resolved mode; `response.diagnostics.parser.field_unavailable` lists
any constraint dropped for lacking dataset/run coverage.

## Enabling filter relaxation

```bash
FILTER_RELAXATION_MODE=auto_soft   # or "diagnose_only"
```

Per-request: `{"filter_relaxation_mode": "auto_soft", "min_results": 5, ...}`.
`min_results` defaults to `top_k` if unset. Verify:
`response.diagnostics.filter_relaxation.triggered`/`.relaxed_constraints`/`.stopped_reason`.
`diagnose_only` reports the same finding without changing `response.results`.

## Enabling adaptive MRL exact rerank (experimental)

```bash
ADAPTIVE_MRL_EXACT_RERANK=true
```

Per-request: `{"adaptive_mrl": {"enabled": true, "exact_rerank": true, ...}, ...}`.
Only ClickHouse is wired; **known to fail the physical-read gate at 20K-row scale** (see
`docs/architecture/ADAPTIVE_MRL.md`) -- enable only for small corpora or with that
limitation accepted.

## Enabling diagnostics on a specific request (never on by default)

```json
{"diagnose": true, "explain": true, ...}
```

Forces `count()`/`EXPLAIN indexes=1` for that one request regardless of whether it's
underfilled. Never set these as an env default -- they exist specifically to stay off the
hot path otherwise.

## Detector enrichment (ingest-time only, not yet wired to a CLI flag this session)

Detector enrichment's DB write functions
(`app.db.postgres.write_run_detector_enrichment`, `app.db.clickhouse.write_run_detector_enrichment`)
and the aggregation/detector modules (`service/app/enrichment/`) are additive and tested,
but no ingest CLI entry point calls them yet this session -- that wiring
(`DETECTOR_ENRICHMENT_ENABLED=true` triggering a real detector pass during
`app.ingestion.ingest`) is a follow-up. Apply the schema migration first:

```bash
PYTHONPATH=service python -c "from app.db import postgres, clickhouse; postgres.init_schema(include_vectors=True); clickhouse.init_schema()"
```

(Additive `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`; safe to run against a live
deployment at any time, including with an active run present.)

## Verifying nothing changed with everything off

```bash
PYTHONPATH=service pytest service/tests/ -q
pytest tests/ -q
```

Expect the same pass/skip counts as `artifacts/baseline_contract/test_results.json`
(521 passed / 4 pre-existing unrelated failures / 16 skipped), plus every new test added
by this work, all passing. Live-backend-dependent new tests skip gracefully without
`CLICKHOUSE_HOST`/`POSTGRES_HOST`/etc. set; with them set (matching whatever real stack
is running), they execute for real -- see `docs/operations/ADVANCED_RETRIEVAL_ROLLBACK.md`
for how this session reached the live containers used to produce this work's evidence.

## Backend benchmark reproduction

```bash
# Requires live ClickHouse + Postgres + Qdrant reachable via the usual CLICKHOUSE_HOST/
# POSTGRES_HOST/QDRANT_URL env vars. Writes and cleans up its own dataset_id; never
# touches real institution data.
PYTHONPATH=service python <adapt from artifacts/advanced_retrieval/backends/ methodology>
```

No standalone script was committed for this ad hoc 5,000-row run (see
`docs/operations/KNOWN_LIMITATIONS.md`) -- promoting it to a committed, parametrized
benchmark script is part of the required 100K-1M-row follow-up.
