# Baseline known limitations (Phase 0)

Recorded before any advanced-retrieval code changes, at HEAD `c77a27f`.

## Environment

- No GPU representative of institution hardware. The only GPU present (NVIDIA GT 1030,
  4GB) is explicitly documented elsewhere in this repo as non-representative; installed
  Torch is CPU-only (`2.13.0+cpu`, `torch.cuda.is_available()==False`). Any phase requiring
  real GPU throughput/VRAM measurement (Qwen forward-pass timing at scale, detector
  training/accuracy at scale, VLM rerank, VRAM co-residency) cannot be executed here and
  will be marked `blocked_by_environment` with the exact command to run on real hardware.
- No human annotators available in this session. Hard-negative Tier C (color/direction/
  fine-detail, human-verified) cannot be produced by this agent; Tier A/B (auto-derivable
  from VisDrone/metadata) can be generated programmatically and will be.
- Docker daemon was stopped at the start of this session and was started mid-session.
  A pre-existing, already-running stack (`video-search-faz7-{pg,ch,qdrant,api,ui}`)
  was discovered already up (containers created ~21-28h prior). This stack holds real
  data (`auair`, `capera` datasets) that must not be modified. This session's live testing
  uses a dedicated, separate test `dataset_id` against the same backend containers, reached
  directly from the venv with current-HEAD code (not through the containers' own,
  possibly-stale images).
- `psycopg2-binary`, `qdrant-client`, and `pymilvus` were not installed in the dev venv at
  session start (previously documented in `docs/operations/BLOCKERS.md` as a psycopg2 gap).
  Installed this session, unpinned (latest compatible), to unblock live backend testing.
  `pymilvus` resolved to `3.0.1`, a different major version than `service/requirements.txt`'s
  pin of `2.4.9` -- Milvus-adapter behavior was not validated against the pinned version.

## Pre-existing, unrelated test failures (not caused by this work, not fixed by this work)

4 failures in `tests/test_faz11_docs_and_notebook.py`, all `FileNotFoundError` on
`docs/USER_GUIDE.md`. Root cause: commit `af6617b` ("docs: reorganize repository
documentation hierarchy") moved this file to `docs/archive/superseded/USER_GUIDE.md`
without updating this doc-consistency test's path reference. This is unrelated to
`ADVANCED_RETRIEVAL_FINAL_PLAN_v2.1.md` and out of scope for this work; left as-is to
avoid unrequested scope creep. See `test_results.json` for full detail.

## Plan-document assumptions not reachable/verifiable in this environment

- Live ClickHouse benchmarking at 1M-row scale: feasible in principle (Docker is up), but
  building and indexing a synthetic 1M-row table is a multi-minute-to-hour operation per
  the repo's own prior measurements (100K rows took ~35.6s insert + ~38.7s index build);
  attempted only if time budget allows after core correctness work, and reported honestly
  either way.
- Qwen3-VL-Embedding-2B GPU throughput: cannot be measured here (no representative GPU).
  Colab-based confirmation scripts are provided per plan Sec.5.2 but not executed.
- VRAM co-residency of a 112B model with Qwen3-VL-Embedding-2B (plan Sec.11.1): cannot be
  measured on this hardware at all; marked `blocked_by_environment` outright.
- Detector accuracy (mAP, per-class recall) on VisDrone/AU-AIR at production scale requires
  multi-hour CPU YOLO inference in this environment; existing repo evidence
  (`docs/operations/STATUS.md` Faz 3) already covers a 73-window bake-off. This work will
  reuse that evidence rather than re-running it, and will not fabricate new accuracy numbers.
