# Advanced retrieval implementation report

Implements `docs/planning/ADVANCED_RETRIEVAL_FINAL_PLAN_v2.1.md` on branch
`feat/advanced-retrieval-evidence-gated`, starting from `main` at `c77a27f` (matching the
plan's own stated reference commit).

## Repository-reality check (plan Sec.0/§2 of the task)

The plan's own two-plane analysis (`search/`, `bench/`, `eval/`, `models/` = research;
`service/app/**` = product) was verified, not assumed: a targeted grep swept every
`from search`/`import bench`/etc. pattern under `service/app/**` and manually verified
every hit as a false positive (the product plane's own `app.search`/`app.bench`
sub-packages, unrelated `.search()` method calls, or the vendored external Qwen repo
path). The plane separation was **already_implemented**; this work preserves it and adds
`service/app/query_planning/`, `service/app/enrichment/`, `service/app/evaluation/`,
`service/app/search/exact_rerank.py`, `service/app/search/vlm_rerank.py`,
`bench/language_parity.py`, `bench/windowing_ablation.py` without crossing it.

## Classification of plan items against real code (Sec.0's required classes)

| Item | Class | Evidence |
|---|---|---|
| Adaptive `candidate_count` bug (Sec.2.1) | `confirmed_defect` | `clickhouse.py`'s pre-fix `count_sql` omitted `candidate_clause`; reproduced live before fixing |
| Double embed call (Sec.2.2) | `confirmed_defect` | `engine.py` called `embed_query()` twice; call-counted before/after |
| count()/EXPLAIN always-on (Sec.2.3) | `confirmed_defect` | reproduced live: 3 ClickHouse queries per default search, 6 for adaptive |
| ClickHouse client per call (Sec.2.4) | `confirmed_defect` | zero reuse existed (not even an unsafe singleton) |
| Adaptive request validation missing (Sec.2.5/4.4) | `partially_implemented` | `base_dim` was already `Literal[256,512]`-restricted; enabled-dimensions/pair-allow-list/`top_n>=top_k` were not checked |
| Research/product plane separation (Sec.0) | `already_implemented` | verified by exhaustive import grep, see above |
| SQL injection allow-list at the pushdown layer (Sec.6.3/11) | `already_implemented` | `normalize_filters()` already rejects unknown fields; this work's parser/ontology feed into it, never around it |
| `DatasetFilterSchema` (plan's proposed new contract) | `already_implemented` (reused, not rebuilt) | `app.search.filter_schema.CANONICAL_FILTER_FIELDS` + `GET /datasets/{id}/filter-schema` already existed; `coverage.available_field_names()` mirrors its logic instead of duplicating a new schema type |
| Windowing being the dominant recall axis (Sec.5) | `blocked_by_environment` (harness built, full sweep not run) | every grid point needs a fresh ingest pass; scoped out given session time budget |
| TR/EN Qwen text-tower quality (Sec.6.5, Ek A item 2) | `measured` (small sample) | real Qwen CPU text-tower run, 84% binding cross-lingual top-1 accuracy, n=19-22 |
| Milvus production-readiness (Sec.20) | `confirmed not production_supported` | widened its contract to fail clearly instead of crashing; did not add it to `VECTOR_BACKENDS` |

## What was built, phase by phase

**Phase 0** -- `artifacts/baseline_contract/`: real HEAD/test/config/schema/backend-
reachability snapshot before any change.

**Phase -1** -- 5 confirmed defects fixed with live evidence
(`artifacts/advanced_retrieval/prefix_fixes/summary.json`): adaptive diagnostics split,
single embed call, count/EXPLAIN off the hot path, thread-safe ClickHouse client reuse,
adaptive request validation with an explicit allow-list.

**Phase 1/1.5** -- Two real-executed notebooks (`notebooks/09_advanced_retrieval_smoke.ipynb`,
`10_clickhouse_adaptive_mrl_scale.ipynb`, run via `jupyter execute`, not just
nbformat-validated); `bench/language_parity.py` (real Qwen CPU measurement, 84% binding
cross-lingual accuracy) and `bench/windowing_ablation.py` (harness built and unit-tested;
full sweep deferred, exact command recorded).

**Phase 2-4** -- `service/app/query_planning/`: contracts, ontology (specificity
precedence, `truck`->`vehicle_count` widening), dataset coverage, rules parser (found and
fixed a real Turkish-plural/English-plural matching bug), LLM provider interfaces
(untested end-to-end, no LLM available; shared validator fully tested including an
injection attempt), planner (safe fallback on any LLM failure). Wired into `engine.py`
behind `parser_mode` (default `none`).

**Phase 5** -- `service/app/query_planning/relaxation.py`: the 5-pass ladder. Found and
fixed two real bugs while testing pass 5 (a default-budget bug that made pass 5
structurally unreachable; a semantics bug that would have dropped manual filters,
contradicting the plan's own unconditional rule). Wired into `engine.py`; live-tested
rescuing an underfilled result.

**Phase 6** -- `service/app/search/exact_rerank.py`: `rerank_candidates_exact()` +
physical-read gate. Measured, not assumed: **the gate failed** at 20K-row scale (full
partition scan regardless of candidate count). Feature stays experimental, default off,
ranking correctness proven.

**Phase 7-8** -- `service/app/enrichment/`: 2 canonical columns migrated live (additive,
zero errors, verified against real read-back), aggregation math (found and fixed a real
uneven-frame-list indexing bug), detector config resolution + fake-model-tested
invocation wrapper, strict/best_effort failure policy, separate provenance namespace.
Tracking deferred as an explicit stub (plan requires a use-case validation this
environment cannot provide).

**Phase 9** -- `service/app/evaluation/hard_negatives.py`: layered contract + Tier A
auto-derivation. Generated a real (if small, n=3) Tier A set from the live `auair`
dataset's actual `vehicle_count` distribution, read-only. Tier B not attempted; Tier C
cannot be fabricated by this agent.

**Phase 10-11** -- `service/app/enrichment/caption.py` (authoritative-field prohibition
enforced in code) and `service/app/search/vlm_rerank.py` (pure SLA/trigger-decision
logic, VRAM gate checked even in `force` mode). Both default off, neither exercised
against a real model (none available).

**Backend benchmark** -- Real 5,000-row, 3-backend, 20-query correctness/latency
comparison (caught and fixed a real methodology bug -- an unpopulated join table --
in the benchmark script itself before trusting its 0/20 pgvector result). Not at the
plan's 100K-1M-row bar; `production_vector_backend` unchanged.

## Test evidence

`artifacts/baseline_contract/test_results.json`: 521 passed / 4 pre-existing unrelated
failures / 16 skipped, before this work. Every commit on this branch added new passing
tests without modifying that baseline (the 4 pre-existing failures are untouched,
unrelated to this plan, and explicitly not this work's to fix). New live-backend-
dependent tests skip gracefully without a reachable ClickHouse/Postgres/Qdrant and run
for real against this session's actual containers when reachable -- see each phase's
commit message for the specific pass counts observed.

## Final GO/NO-GO

See `artifacts/advanced_retrieval/final_acceptance.json` for the machine-readable form.

```text
Accepted (default off, ready to enable):
  - Phase -1 defect fixes                     -- always active, no flag
  - query_planning (parser_mode=rules)        -- accepted
  - filter_relaxation                         -- accepted
  - ontology + dataset coverage                -- accepted

Experimental (built, correctness proven, gate/validation incomplete):
  - adaptive_mrl_exact_rerank                  -- physical-read gate failed at measured scale
  - query_planning (parser_mode=llm)           -- untested end-to-end, no LLM available
  - detector_enrichment                        -- migration + aggregation real; real model run not_run

Deferred (contract built, explicitly not implemented):
  - tracking                                   -- use-case validation not possible here
  - caption                                    -- no model available
  - vlm_rerank                                 -- no model available; VRAM gate not_run

Unchanged:
  - production_vector_backend = clickhouse
  - production_defaults_changed = false
```
