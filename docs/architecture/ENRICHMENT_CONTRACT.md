# Enrichment contract: detector, tracking, caption

Implements `docs/planning/ADVANCED_RETRIEVAL_FINAL_PLAN_v2.1.md` Sec.8/10/16/18. Code
under `service/app/enrichment/`.

## Detector enrichment (Phase 7)

**Not a post-hoc hook.** Run-versioning means canonical count columns live in
`seg_ch_{d}_runs`/`run_segment_metadata`, keyed by `run_id`; an active run cannot be
mutated (`_ensure_inactive()`). Enrichment writes must happen inside the same ingest run
that wrote the base rows, before that run is activated --
`app.db.postgres.write_run_detector_enrichment()` and
`app.db.clickhouse.write_run_detector_enrichment()` are additive `UPDATE`/`ALTER ...
UPDATE` calls scoped to one `run_id`, not a general post-hoc patch path.

**Exactly two new canonical, filterable columns** (plan Sec.8.1's explicit rejection of
the "5 new columns" v1 proposal): `median_visible_vehicle_count`,
`detection_persistence_ratio`. Added via `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` to
`segment_metadata`/`run_segment_metadata` (Postgres) and `seg_ch_{d}[_runs]`
(ClickHouse), registered in `app.search.filter_schema.CANONICAL_FILTER_FIELDS` and
`app.search.filter_projection.POSTGRES_RUN_COLUMNS` -- filterable through the exact same
`normalize_filters()` path as every other field. Everything else the plan's v1 proposed
(`max_visible_*`, `p90_visible_*`, `unique_vehicle_tracks`) is a **sidecar concept**
(`app.enrichment.contracts.SidecarDetection`), never a migrated column.

**Null vs zero is enforced in code, not just documented**:
`app.enrichment.aggregation.median_visible_count()` (and its siblings) return `None` for
an empty/failed frame series and a real `0.0` for an all-zero-but-successful one.
`build_canonical_and_sidecar()` raises `DetectorStrictFailure` under
`failure_policy="strict"` and returns `None`-valued fields under `"best_effort"` --
never coalesces a failure to `0`.

Config resolution (`app.enrichment.detector.resolve_variant()`) mirrors
`ingest/04_detect.py::_resolve_variant()`'s `config.yaml: detector.variants` structure
without importing that research-plane module -- a deliberate small duplication that keeps
the research/product plane import boundary clean, per the plan's explicit instruction not
to wire `search/`, `bench/`, `eval/`, `models/` into the production import path.

**Not executed end-to-end this session**: a real detection run needs real video frames
through a real YOLO variant, which is an ingest-pipeline integration exercise, not a unit
test. `detector.py`'s contract (config resolution, the `DetectionResult` shape
`aggregation.py` consumes) is tested with an injected fake model
(`test_enrichment_detector.py`); the DB write path is tested live against the real
Postgres/ClickHouse containers (`test_enrichment_db_write.py`).

Provenance stays in its own namespace (`app.enrichment.provenance.merge_provenance()`):
`{"embedding_provenance": {...}, "enrichment_provenance": {"detector": {...}}}` --
detector metadata never overwrites embedding provenance.

## Tracking (Phase 8) -- deferred

`app.enrichment.tracking` is a deliberate stub. The plan requires the unique-object-count
use case to be validated *before* tracking is implemented at all (Sec.16); no such
validation is possible in this environment (needs a labeled multi-instance video
benchmark). `TrackingConfig`/`TrackingResult` fix the contract shape (`unique_vehicle_tracks`
explicitly marked sidecar-only, never a canonical hard filter) for whenever that gate is
met. Calling `run_tracking_not_implemented()` raises clearly rather than no-op'ing.

## Caption (Phase 10) -- default off, contracts only

`CAPTION_MODE=off` by default. `app.enrichment.caption.CAPTION_AUTHORITATIVE_FIELDS_FORBIDDEN`
+ `assert_caption_not_authoritative()` enforce in code (not just prose) that caption text
never becomes an authoritative `velocity_mps`/`altitude_m`/count/`is_night` filter.
`run_caption()` never propagates a model failure into core search (`text=None` on
failure, best-effort only -- the plan does not ask for a caption strict mode). No
captioning model is available in this environment; not exercised end-to-end.

## Detector accuracy (Phase 8, referenced)

CapERA-style plumbing smoke is not an accuracy gate -- only decode/load/latency/coverage.
Real detector accuracy (mAP, per-class recall, count MAE) requires VisDrone/AU-AIR/a
labeled institution subset; this repo's own prior bake-off
(`docs/operations/STATUS.md` Faz 3, 73 windows, real VisDrone annotations) is the existing
evidence base and was not re-run this session -- reused, not fabricated.
