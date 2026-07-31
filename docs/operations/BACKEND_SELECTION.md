# Backend selection

Real, code-verified reachability audit: `artifacts/baseline_contract/backend_reachability.json`
(baseline) and `artifacts/advanced_retrieval/prefix_fixes/backend_reachability_after_fixes.json`
(post-Phase -1 delta). Real cross-backend correctness/latency data:
`artifacts/advanced_retrieval/backends/summary.json`.

## Current state (unchanged by this work)

```text
Production default:       clickhouse (ENABLED_VECTOR_BACKENDS=clickhouse)
Enableable alternatives:  qdrant, pgvector  (ENABLED_VECTOR_BACKENDS=clickhouse,qdrant,pgvector)
Correctness reference:    numpy_exact (never production; not in ENABLED_VECTOR_BACKENDS'
                           allow-list; reachable only via direct internal call)
Incomplete/experimental:  milvus (search-only adapter; no run-scoped collection, no
                           metadata/telemetry filter pushdown, no schema/ingest-write path;
                           absent from VECTOR_BACKENDS, so ENABLED_VECTOR_BACKENDS=milvus
                           fails Settings.validate() at startup)
```

## Evidence gathered this session

- All three enableable backends (ClickHouse, pgvector, Qdrant) verified live, correctness
  cross-checked against `numpy_exact` at 5,000 rows: 20/20 query agreement, all three.
- `milvus.py`/`common_exact.py` widened to accept the full backend contract
  (`run_id`/filters/`diagnose`/`explain`) so a direct internal call fails with a clear
  `ValueError` naming exactly what's unsupported, instead of crashing with `TypeError` or
  silently returning wrong data. This did not change either backend's enabled/production
  status.

## Decision

**No change.** `production_vector_backend` stays `clickhouse`.
`production_defaults_changed=false`. Per the plan's own rule (Sec.20): "Kanit yetersizse
production default ClickHouse olarak kalmalidir" -- a 5,000-row single-scale, exact-
strategy-only measurement does not clear that bar in either direction.

## What would change this decision

A benchmark at the plan's own 100K-1M-row scale, covering all four strategies
(`exact`/`ann`/`prefilter`/`postfilter` and their per-backend equivalents), filtered
recall, and 1/5/10-way concurrency -- matching the depth of this repo's own existing
ClickHouse-only Faz 2 study (`docs/agents/TASKS.md`). Not run this session; see
`artifacts/advanced_retrieval/backends/summary.json`'s `required_followup` for the exact
scope.

If such evidence ever does justify a change, per the plan's own requirement it must ship
with: a decision record, a migration, a rollback path, compatibility notes, and an
operator runbook update -- not a config default flip alone.
