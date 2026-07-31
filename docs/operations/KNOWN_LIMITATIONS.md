# Advanced retrieval: known limitations

Companion to `artifacts/baseline_contract/known_limitations.md` (Phase 0, environment-level).
This file tracks limitations introduced or discovered by
`docs/planning/ADVANCED_RETRIEVAL_FINAL_PLAN_v2.1.md`'s implementation.

## Environment blockers (not this codebase's fault)

- **No representative GPU.** The only GPU present (NVIDIA GT 1030, 4GB) is explicitly
  non-representative institution hardware. Blocks: real Qwen video-embedding throughput
  at scale, detector training/GPU inference at scale, VLM rerank end to end, VRAM
  co-residency measurement (plan Sec.11.1).
- **No human annotators.** Hard-negative Tier C (color/direction/fine-detail) cannot be
  produced by this agent -- `attribute_reachability_precheck()` implements only the
  structural half of the plan's Sec.9.2 gate.
- **No labeled multi-instance tracking benchmark.** Tracking (Phase 8) stays an
  unimplemented, explicitly-raising stub pending that validation.

## Real gaps found and left open (time-budget, not environment)

- **`llm.py`'s two providers are untested end-to-end.** No LLM (local or vLLM-served) was
  available. Their shared response validator has full unit coverage (including an
  injection attempt); the actual model call paths (`parse_with_transformers_local`,
  `parse_with_vllm_openai_compatible`) are code-reviewed, not live-tested.
- **Detector enrichment's real model path is untested end-to-end.** A real detection run
  needs real video frames through a real YOLO variant -- an ingest-pipeline integration
  exercise. `detector.py`'s contract is tested with an injected fake model.
- **Hard-negative Tier A covers 3 queries, not ~100.** Generated from the live `auair`
  dataset's real `vehicle_count` (read-only); `class_proximity` produced nothing because
  `auair`'s `person_count` is uniformly 0 in this dataset. Tier B (metadata-derivable:
  `camera_motion`/`brightness`) was not attempted -- needs those fields verified
  populated on a real dataset first.
- **Windowing ablation harness (`bench/windowing_ablation.py`) was not run at full
  scope.** Every grid point needs a fresh ingest pass (new segment boundaries); a
  multi-point sweep is a real, multi-CPU-minute-per-config cost this session did not
  spend. See `artifacts/advanced_retrieval/windowing/status.json` for the exact required
  command.
- **Backend benchmark ran at 5,000 rows, not the plan's 100K-1M bar.** Real, correct,
  but not sufficient evidence to change the production backend default (and did not --
  `production_vector_backend` stays `clickhouse`). See
  `artifacts/advanced_retrieval/backends/summary.json`.
- **Adaptive MRL exact rerank's physical-read gate failed at 20K-row measured scale.**
  `rows_read` did not scale with candidate count for `segment_id IN (...)`; the
  `(video_id,t_start)` alternative showed no improvement at that scale either, but the
  scale itself (~3 ClickHouse granules total) is too small to fairly test sort-key-based
  lookups. Feature stays `experimental`, default off.
- **Milvus and `numpy_exact` remain unreachable via the public HTTP contract by
  design** -- absent from `VECTOR_BACKENDS`/`BACKEND_REGISTRY`, so
  `ENABLED_VECTOR_BACKENDS` cannot include either and `POST /search` rejects them before
  `engine.search()` runs. `milvus.py`/`common_exact.py` were widened this session to
  accept the full backend kwarg contract (`run_id`, filters, `diagnose`/`explain`) and
  fail clearly instead of crashing if ever called directly with them, but neither
  became production-supported.
- **`enrichment/caption.py` and `search/vlm_rerank.py` have zero real-model exercise.**
  Both are pure contract/policy code with full unit coverage of their own logic; no
  captioning model or VLM was available to validate an actual generation/rerank call.

## Pre-existing, unrelated to this plan (recorded, not fixed)

- 4 failures in `tests/test_faz11_docs_and_notebook.py`
  (`FileNotFoundError: docs/USER_GUIDE.md`), caused by commit `af6617b`'s docs
  reorganization moving that file without updating this test's path reference. Present
  at the branch point before any of this work; left alone to avoid unrequested scope
  creep. See `artifacts/baseline_contract/test_results.json`.
