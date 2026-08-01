# Backend Profile and Hybrid Search Audit

Date: 2026-08-01

Branch: `feat/advanced-retrieval-evidence-gated`

Decision: **NO_GO**

This audit preserves the ClickHouse production default. It combines code inspection,
new regression tests, Compose configuration validation, and existing immutable benchmark
artifacts. Docker Desktop's daemon was unavailable in this session, so no live profile
startup, 100K rerun, or new database benchmark is claimed.

## Executive result

- The existing 100K ClickHouse evidence is consistent with prefilter doing filtered
  exact scoring on this schema: the saved `EXPLAIN indexes=1` interpretation says no
  vector index, and exact/prefilter have identical `rows_read` and `bytes_read` for both
  saved selectivities. This is a result for this schema and ClickHouse version, not a
  universal ClickHouse claim.
- Qdrant-only and pgvector-only paths are now expressible as standalone Compose files.
  Static Compose resolution and code contracts pass; live startup/ingest/search/resume
  remains unverified because the daemon is down.
- API defaults now come from validated `DEFAULT_VECTOR_BACKEND`; strategy is selected by
  backend (`prefilter`, `ann`, or `iterative_scan`). Production remains ClickHouse.
- Unsupported adaptive exact rerank is rejected instead of returning an empty success;
  unmeasured pushdown candidate counts remain `null/not_requested`; request relaxation
  controls are exposed; and detector enrichment cannot mutate an active run.
- Generic ingest is synchronous and serial across every enabled dimension/backend. Every
  enabled backend is activation-blocking. No partial asynchronous mirror framework was
  added.

## ClickHouse evidence

Source: `artifacts/scale_evidence_bench_scale_512.json` (100K rows, 512d, 50 repeats).

| Strategy | Selectivity | p50 / p95 ms | rows returned | rows / bytes read | Recall@10 |
|---|---:|---:|---:|---:|---:|
| exact | loose | 156.314 / 238.420 | 10 | 149,938 / 217,331,576 | reference |
| prefilter | loose | 143.776 / 175.076 | 10 | 149,938 / 217,331,576 | not separately saved |
| HNSW/auto | loose | 22.540 / 40.330 | 10 | 94,845 / 11,360,274 | 1.0 |
| postfilter+rescore | loose | 109.181 / 150.220 | 10 | 149,938 / 217,331,576 | not separately saved |
| exact | strict | 111.841 / 175.622 | 10 | 139,814 / 217,602,602 | reference |
| prefilter | strict | 104.188 / 125.219 | 10 | 139,814 / 217,602,602 | not separately saved |
| HNSW/auto | strict | 16.723 / 33.708 | 0 | 49,938 / 10,993,703 | 0 returned |
| postfilter+rescore | strict | 113.256 / 166.004 | 0 | 105,031 / 216,973,646 | 0 returned |

The smaller saved strategy matrix (`artifacts/strategy_matrix_report.json`) records
`vector_index_in_plan=false` for exact and prefilter and `true` for HNSW and
postfilter+rescore. The 100K artifact does not embed raw `EXPLAIN PIPELINE` output.
Therefore the strongest supported statement is: **prefilter did not use the vector index
in the measured schema/plan and behaved like a filtered scan**. Identical physical-read
counters and similar latency are empirically consistent with approximately `O(M*d)`
scoring, but one scale and two selectivities do not establish an asymptotic growth curve.

Loose HNSW is the measured latency winner and retained Recall@10=1.0. Under the strict
filter, prefilter is the fastest strategy that still returns all 10 requested results.
The strict HNSW path underfills because ANN candidate generation precedes filtering; none
of the fetched nearest candidates survived `bus_count>=1 AND person_count>=3`. Exact and
prefilter returning 10 proves this is candidate loss, not an empty filtered corpus.
Medium selectivity was not recorded and remains a required rerun.

## Backend profile data paths

### Qdrant-only

`docker-compose.qdrant.yml` contains only PostgreSQL, Qdrant, API, and UI. At startup the
API initializes PostgreSQL metadata with `include_vectors=false`, then initializes only
enabled registry adapters. With `ENABLED_VECTOR_BACKENDS=qdrant`, no ClickHouse adapter
is initialized or written.

PostgreSQL owns datasets, runs/chunks, videos/segments, metadata, telemetry, active-run
pointers, ground truth, and result hydration. Qdrant owns embeddings and filter payloads.
Collections are `segments_<dimension>` for enabled dimensions, with exact dimension size,
cosine distance, HNSW `m=16`, and `ef_construct=128`. The default Qdrant profile creates
`segments_512`; enabling `2048,1024,512,256` creates all four collections.

Payload includes `run_id`, `dataset_id`, `chunk_index`, `segment_id`, `video_id`,
`t_start`, `t_end`, canonical metadata, canonical telemetry, and the configured extra
values carried by vector rows. Index creation covers `dataset_id`, `run_id`,
`chunk_index`, `event_category`, `split`, `video_id`, `altitude_m`, `velocity_mps`,
`gimbal_pitch`, `person_count`, `vehicle_count`, `bus_count`, and `is_night`.

Search returns Qdrant `segment_id + score`; `engine.py` hydrates those IDs from the active
PostgreSQL run. Resume cleanup iterates only the run's enabled backends. Finalization
counts Qdrant rows for every enabled dimension before PostgreSQL atomically switches the
active pointer. These are passing code contracts, not a live Compose acceptance result.

### pgvector-only

`docker-compose.pgvector.yml` contains PostgreSQL/pgvector, API, and UI only. Registry
initialization calls PostgreSQL schema initialization with vectors enabled; metadata and
vectors remain in the same PostgreSQL control plane. No ClickHouse or Qdrant service is
present. Static configuration passes; live smoke remains not run.

### ClickHouse production and benchmark

The existing `docker-compose.yml` remains the ClickHouse production profile with
`DEFAULT_VECTOR_BACKEND=clickhouse`, `ENABLED_VECTOR_BACKENDS=clickhouse`, and 512d by
default. `docker-compose.benchmark.yml` adds Qdrant and defaults to all three backends and
all four dimensions, but its values are now explicitly configurable through
`BENCHMARK_VECTOR_BACKENDS`, `BENCHMARK_DEFAULT_VECTOR_BACKEND`, and
`BENCHMARK_DIMENSIONS`.

Validated launch forms are:

```powershell
docker compose --env-file .env -f docker-compose.yml up -d --build
docker compose --env-file .env -f docker-compose.qdrant.yml up -d --build
docker compose --env-file .env -f docker-compose.pgvector.yml up -d --build
docker compose --env-file .env -f docker-compose.yml -f docker-compose.benchmark.yml up -d --build
```

Use exactly one of the first three production profile commands. The benchmark form is an
override on the ClickHouse production file and intentionally starts all data stores.

## Ingest fan-out

`GenericIngestor._flush_pending()` writes PostgreSQL metadata first, then loops dimensions
and enabled backends serially. A backend exception fails the chunk, prevents finalization,
and preserves the previous active run. `RunCoordinator.finalize_run()` requires the exact
row count for every enabled backend/dimension. Consequently the slowest backend and every
preceding serial write contribute directly to activation latency.

The existing identical 5K-row exact-backend artifact measured standalone writes:
ClickHouse 0.9 s, Qdrant 2.2 s, pgvector 12.9 s. These imply standalone Qdrant and
pgvector writes were 2.44x and 14.33x the ClickHouse write time in that run. Adding the
standalone times gives only a serial-work estimate (3.1 s for ClickHouse+Qdrant; 16.0 s
for all three), not a measured generic-ingest wall clock. Metadata time, indexing time,
chunk commit time, cached fan-out rows/s, and real-Qwen end-to-end fan-out were not saved,
so no measured multi-backend slowdown multiplier is claimed.

Production recommendation remains synchronous PostgreSQL + ClickHouse/512d. Benchmark
recommendation is the benchmark Compose profile with all three backends and identical
cached embeddings. Optional asynchronous mirrors are deferred pending a design that has
an explicit mirror ledger, retry policy, consistency reporting, and no activation gate.

## Verification and status

- PASS: new targeted regression set, 39 passed / 2 live skips.
- PASS: `docker compose config --quiet` for ClickHouse, Qdrant-only, pgvector-only, and
  benchmark configurations.
- PASS_WITH_PREEXISTING_FAILURES: root `tests/`, 384 passed / 4 failures. All four are
  pre-existing documentation relocation failures for missing root-level docs and match
  the branch's recorded four-failure baseline; they are unrelated to these changes.
- NOT_RUN: full service suite; two attempts timed out while unavailable live-backend
  probes waited on closed ports. Targeted affected tests completed.
- BLOCKED: live startup/health/schema/ingest/search/resume/activation for Qdrant-only and
  pgvector-only; Docker daemon unavailable.
- BLOCKED: fresh 100K loose/medium/strict `EXPLAIN indexes=1` + `EXPLAIN PIPELINE` matrix;
  Docker daemon unavailable and medium was absent from the saved artifact.
- BLOCKED: cached multi-backend generic-ingest and real-Qwen end-to-end fan-out benchmark.

Accepted by code/test: validated defaults, backend-aware strategy default, exact-rerank
rejection, honest candidate diagnostics, explicit relaxation controls and filter
preservation, active-run enrichment guard. Experimental until live acceptance: Qdrant-only
and pgvector-only deployment profiles. Detector enrichment remains a contract/scaffold,
not generic-ingest end-to-end. Adaptive exact rerank remains experimental because its
saved physical-read gate failed. Async mirrors remain deferred.

## Merge decision

**NO_GO**. The known code defects are fixed locally, but the requested live profile smoke
matrix, medium-selectivity ClickHouse plan/benchmark, and multi-backend ingest timing are
release blockers. Re-run those gates on a host with Docker before merging. No production
backend default was changed and no remote push was performed.
