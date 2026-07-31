# Advanced retrieval rollback

## Config-level rollback (covers everything except the schema migration)

Every new capability defaults off. To fully revert behavior without touching code or
schema, ensure these env vars are unset or at their defaults:

```text
QUERY_PARSER_MODE=none
FILTER_RELAXATION_MODE=off
ADAPTIVE_MRL_EXACT_RERANK=false
CAPTION_MODE=off
VLM_RERANK_MODE=off
```

And that no request sets `parser_mode`, `filter_relaxation_mode`,
`adaptive_mrl.exact_rerank`, `diagnose`, or `explain`. With all of the above, every
response is byte-for-byte equivalent (modulo the documented volatile fields: timing,
timestamps, UUIDs) to the pre-this-work baseline in
`artifacts/baseline_contract/api_contract.json`, **except** for the additive diagnostics
keys themselves (`filtered_corpus_count`, `candidate_input_count`,
`candidate_count_status`, `explain_status`, `stage1_*`/`stage2_*`/`final_returned_count`,
`parser`, `filter_relaxation`, `execution_policy`) -- these are new keys, never removals
or renames, so no existing client integration breaks, but a byte-diff will show them.

## Code-level rollback

All new code lives in newly-created files/packages
(`service/app/query_planning/`, `service/app/enrichment/`, `service/app/search/
exact_rerank.py`, `service/app/search/vlm_rerank.py`, `bench/language_parity.py`,
`bench/windowing_ablation.py`, `service/app/evaluation/`) plus additive edits to
existing files (`clickhouse.py`, `postgres.py`, `qdrant.py`, `milvus.py`,
`common_exact.py`, `engine.py`, `main.py`, `config.py`, `filter_schema.py`,
`filter_projection.py`, `router.py`). `git revert` of this branch's commits (or any
subset -- each commit is independently revertible per the commit discipline used
throughout) removes the feature entirely; no existing function signature was changed in
a way that breaks an existing caller (`_one_run()` gained required keyword-only
`metadata_filters`/`telemetry_filters` params, but it has exactly one caller, `search()`,
updated in the same commit).

## Schema rollback

**Only one schema change this session**: two nullable, additive columns
(`median_visible_vehicle_count`, `detection_persistence_ratio`) on
`segment_metadata`/`run_segment_metadata` (Postgres) and `seg_ch_{d}[_runs]`
(ClickHouse), added via `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`.

To roll back:

```sql
-- Postgres
ALTER TABLE segment_metadata DROP COLUMN IF EXISTS median_visible_vehicle_count;
ALTER TABLE segment_metadata DROP COLUMN IF EXISTS detection_persistence_ratio;
ALTER TABLE run_segment_metadata DROP COLUMN IF EXISTS median_visible_vehicle_count;
ALTER TABLE run_segment_metadata DROP COLUMN IF EXISTS detection_persistence_ratio;
```

```sql
-- ClickHouse, per enabled dimension d
ALTER TABLE seg_ch_{d} DROP COLUMN IF EXISTS median_visible_vehicle_count;
ALTER TABLE seg_ch_{d} DROP COLUMN IF EXISTS detection_persistence_ratio;
ALTER TABLE seg_ch_{d}_runs DROP COLUMN IF EXISTS median_visible_vehicle_count;
ALTER TABLE seg_ch_{d}_runs DROP COLUMN IF EXISTS detection_persistence_ratio;
```

Safe at any time: nothing reads these columns unless detector enrichment explicitly wrote
them (which nothing does automatically -- no ingest CLI wiring exists yet this session,
per `docs/operations/ADVANCED_RETRIEVAL_RUNBOOK.md`), and both columns are nullable with
no default, so no existing row or query is affected by adding or dropping them.
**No active-run or existing-data safety concern**: the migration was applied live against
this session's real running Postgres/ClickHouse containers with zero errors, and every
existing dataset (`auair`, `capera`) remained fully intact and queryable throughout (see
`artifacts/baseline_contract/` for before-state and this branch's commit history for the
applied statements).

## Verification after any rollback step

```bash
PYTHONPATH=service pytest service/tests/ -q
pytest tests/ -q
```
