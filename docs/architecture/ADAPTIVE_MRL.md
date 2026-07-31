# Adaptive MRL: fixes, validation, and exact rerank

Implements `docs/planning/ADVANCED_RETRIEVAL_FINAL_PLAN_v2.1.md` Sec.2/4/6/14.

## Phase -1 fixes (already in production, always active)

1. **Diagnostics split** (`service/app/db/clickhouse.py`, `postgres.py`, `qdrant.py`).
   `search_vectors()` now returns `filtered_corpus_count`, `candidate_input_count`,
   `candidate_count_status`, `explain_status` alongside the existing `candidate_count`.
   For adaptive requests, `engine.py::_one_run()` additionally reports
   `stage1_requested_candidate_count`, `stage1_returned_candidate_count`,
   `stage2_input_candidate_count`, `stage2_returned_count`, `final_returned_count`.
   `candidate_shortage`/`ann_filter_loss` are judged from `stage1_returned_candidate_count`
   (always known for free), never from stage-2's own candidate-restricted count.
2. **Single embed call.** `app.embedding.router.embed_query_multi(text, dimensions)`
   computes the raw base embedding once and derives every requested dimension via
   `truncate_and_normalize()` -- `engine.py`'s adaptive branch calls this once instead of
   calling `embed_query()` twice.
3. **count()/EXPLAIN off the hot path.** Both only run when the result is actually
   underfilled or when the request sets `diagnose=true`/`explain=true` (new, additive,
   default-`false` `SearchRequest` fields).
4. **Thread-safe ClickHouse client reuse.** `app.db.clickhouse.client()` is thread-local
   (one `clickhouse_connect.Client` per worker thread, created once), with
   `_with_reconnect` evicting and retrying once on a transient connection error.
5. **Adaptive request validation.** `POST /search` rejects (400) any `adaptive_mrl`
   request whose `base_dim` is disabled, whose `base_dim >= dimension`, whose
   `(base_dim, dimension)` pair is outside `main.ADAPTIVE_MRL_ALLOWED_PAIRS =
   {(256,512),(256,1024),(256,2048),(512,1024),(512,2048)}`, or whose `top_n < top_k`.

Full before/after evidence: `artifacts/advanced_retrieval/prefix_fixes/summary.json`.

## Exact candidate rerank (Phase 6, experimental)

`service/app/search/exact_rerank.py::rerank_candidates_exact(dataset_id, dimension,
query_vector, candidate_ids, top_k, run_id=None, backend="clickhouse")` -- additive,
narrower than `search_vectors()`: no `strategy` parameter (always
`query_plan_try_use_vector_search=0`), no filters (stage-1 candidates already satisfy
them), always requests ClickHouse's `read_rows`/`read_bytes` physical-read counters.

Enable with `adaptive_mrl.exact_rerank=true` (request) or `ADAPTIVE_MRL_EXACT_RERANK=true`
(env). **Default is `false`.** Ranking correctness is verified live against
`app.search.common_exact` (the project's brute-force reference) -- identical top-k order
on the same candidate set.

### Physical-read gate: FAILED at measured scale

`app.search.exact_rerank.evaluate_physical_read_gate()` turns a real `rows_read`
measurement into pass/fail. Measured against a real 20,000-row `seg_ch_512` corpus:
`rows_read` was exactly 20,000 (the full corpus) at every tested candidate count
(25-500) -- `segment_id IN (...)` is not physically scoped to the candidate set, because
`segment_id` is not `seg_ch_512`'s sort key (`ORDER BY (dataset_id, video_id, t_start)`).
The plan's suggested `(video_id,t_start) IN (...)` alternative showed no improvement at
the same scale, but that scale (~3 granules total at ClickHouse's default
`index_granularity=8192`) is too small to fairly test whether sort-key-aligned lookups
help at all -- a genuinely open question, not a settled negative.

Full data and the required larger-scale follow-up:
`artifacts/advanced_retrieval/adaptive_mrl/physical_read_gate_summary.json`.

**Status: `experimental`.** The feature stays available behind its flag (correctness is
proven) but is not a production default, per the plan's own gate
("rows_read ve bytes_read candidate sayisiyla yaklasik olceklenmiyorsa exact candidate
rerank production'a kabul edilmez").

## Allow-list rationale

`base_dim` is syntactically restricted to `{256, 512}` by `AdaptiveMRL`'s Pydantic
`Literal` type; every pair satisfying `base_dim < dimension` with that constraint happens
to already be in `ADAPTIVE_MRL_ALLOWED_PAIRS` today. The allow-list is still the actual
source of truth (checked independently of the inequality) so a future widening of
`base_dim`'s type can't silently skip the "proven by benchmark evidence" gate the plan
requires for new pairs.
