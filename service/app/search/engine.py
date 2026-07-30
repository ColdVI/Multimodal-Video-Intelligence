from __future__ import annotations

import statistics
import time
from typing import Any, Callable

import numpy as np

from app.config import settings
from app.db import clickhouse, milvus, postgres, qdrant
from app.embedding.router import embed_query
from app.search import common_exact
from app.search.strategies import validate_strategy


BACKENDS: dict[str, Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]]] = {
    "clickhouse": clickhouse.search_vectors,
    "qdrant": qdrant.search_vectors,
    "pgvector": postgres.search_vectors,
    "milvus": milvus.search_vectors,
    "numpy_exact": common_exact.search_vectors,
}

PATTERN_EXECUTION_IMPLEMENTED = False


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _merge_results(hits: list[dict[str, Any]], hydrated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["segment_id"]: row for row in hydrated}
    results = []
    for hit in hits:
        row = by_id.get(hit["segment_id"], {"segment_id": hit["segment_id"]})
        results.append({**row, "score": hit["score"]})
    return results


def _one_run(
    request: Any, active_snapshot: dict[str, Any] | None,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any], int, set[str]]:
    started = time.perf_counter()
    filter_started = time.perf_counter()
    run_id = None if active_snapshot is None else str(active_snapshot["run_id"])
    if run_id is None:
        candidate_ids = postgres.filter_segment_ids(
            request.dataset_id, request.metadata_filters, request.telemetry_filters,
        )
    else:
        candidate_ids = postgres.filter_run_segment_ids(
            request.dataset_id, run_id, request.metadata_filters, request.telemetry_filters,
        )
    filter_ms = (time.perf_counter() - filter_started) * 1000.0
    candidate_set = set(candidate_ids)
    if not candidate_ids:
        total_ms = (time.perf_counter() - started) * 1000.0
        timings = {"filter": filter_ms, "embed": 0.0, "vector_search": 0.0, "hydrate": 0.0, "total": total_ms}
        return timings, [], {"plan_used_vector_index": False, "indexed_vectors_count": None, "notes": ["filters matched zero candidates"]}, 0, candidate_set

    embed_started = time.perf_counter()
    query_vector = embed_query(request.query, request.dimension)
    embed_ms = (time.perf_counter() - embed_started) * 1000.0

    search_started = time.perf_counter()
    search_function = BACKENDS[request.backend]
    if request.adaptive_mrl.enabled:
        base_vector = embed_query(request.query, request.adaptive_mrl.base_dim)
        search_kwargs = {} if run_id is None else {"run_id": run_id}
        base_hits, _ = search_function(
            request.dataset_id,
            request.adaptive_mrl.base_dim,
            base_vector,
            request.adaptive_mrl.top_n,
            request.strategy,
            candidate_ids, **search_kwargs,
        )
        rerank_ids = [hit["segment_id"] for hit in base_hits]
        hits, diagnostics = search_function(
            request.dataset_id,
            request.dimension,
            query_vector,
            request.top_k,
            request.strategy,
            rerank_ids, **search_kwargs,
        )
        diagnostics.setdefault("notes", []).append(
            f"adaptive MRL {request.adaptive_mrl.base_dim}→{request.dimension}, top_n={request.adaptive_mrl.top_n}"
        )
    else:
        search_kwargs = {} if run_id is None else {"run_id": run_id}
        hits, diagnostics = search_function(
            request.dataset_id,
            request.dimension,
            query_vector,
            request.top_k,
            request.strategy,
            candidate_ids, **search_kwargs,
        )
    vector_search_ms = (time.perf_counter() - search_started) * 1000.0

    hydrate_started = time.perf_counter()
    hydrate_kwargs = {} if run_id is None else {"run_id": run_id}
    hydrated = postgres.hydrate([hit["segment_id"] for hit in hits], **hydrate_kwargs)
    results = _merge_results(hits, hydrated)
    hydrate_ms = (time.perf_counter() - hydrate_started) * 1000.0
    total_ms = (time.perf_counter() - started) * 1000.0
    timings = {
        "filter": filter_ms,
        "embed": embed_ms,
        "vector_search": vector_search_ms,
        "hydrate": hydrate_ms,
        "total": total_ms,
    }
    return timings, results, diagnostics, len(candidate_ids), candidate_set


def search(request: Any) -> dict[str, Any]:
    dataset = postgres.dataset_info(request.dataset_id)
    if dataset is None:
        raise ValueError(f"unknown dataset_id: {request.dataset_id}")
    if request.telemetry_filters and not dataset["has_telemetry"]:
        active = {key: value for key, value in request.telemetry_filters.items() if value}
        if active:
            raise ValueError(f"dataset {request.dataset_id} has no telemetry fields")
    validate_strategy(request.backend, request.strategy)
    if request.pattern == "C" and request.backend != "pgvector":
        raise ValueError("pattern C is the pgvector single-store path")
    active_snapshot = postgres.get_active_run_snapshot(request.dataset_id)
    totals: list[float] = []
    timing_rows: list[dict[str, float]] = []
    results: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    candidate_count = 0
    candidate_set: set[str] = set()
    for _ in range(request.repeats):
        timings, results, diagnostics, candidate_count, candidate_set = _one_run(request, active_snapshot)
        timing_rows.append(timings)
        totals.append(timings["total"])

    timing_medians = {
        stage: round(statistics.median(row[stage] for row in timing_rows), 3)
        for stage in ("filter", "embed", "vector_search", "hydrate", "total")
    }
    returned_ids = [row["segment_id"] for row in results]
    filter_correctness = all(segment_id in candidate_set for segment_id in returned_ids)
    underfilled = len(results) < request.top_k
    candidate_shortage = underfilled and candidate_count < request.top_k
    ann_filter_loss = underfilled and candidate_count >= request.top_k
    diagnostics = {
        "candidate_count": candidate_count,
        "returned_count": len(results),
        "underfilled": underfilled,
        "underfilled_reason": (
            "candidate_shortage" if candidate_shortage
            else "ann_filter_loss" if ann_filter_loss
            else None
        ),
        "candidate_shortage": candidate_shortage,
        "ann_filter_loss": ann_filter_loss,
        "underfilled_expected": candidate_shortage,
        "plan_used_vector_index": diagnostics.get("plan_used_vector_index"),
        "indexed_vectors_count": diagnostics.get("indexed_vectors_count"),
        "filter_correctness": filter_correctness,
        "notes": diagnostics.get("notes", []),
        "quality_vs_groundtruth": None,
        "r_at_1": None if settings.embedding_mode == "synthetic" else None,
        "ndcg": None if settings.embedding_mode == "synthetic" else None,
    }
    return {
        "embedding_mode": settings.embedding_mode,
        "dataset_id": request.dataset_id,
        "run_id": None if active_snapshot is None else active_snapshot["run_id"],
        "vector_provenance": (
            dataset["vector_provenance"] if active_snapshot is None else active_snapshot["vector_provenance"]
        ),
        "model_id": None if active_snapshot is None else active_snapshot.get("model_id"),
        "model_revision": None if active_snapshot is None else active_snapshot.get("model_revision"),
        "source_commit": None if active_snapshot is None else active_snapshot.get("source_commit"),
        "backend": request.backend,
        "strategy": request.strategy,
        "dimension": request.dimension,
        "pattern": request.pattern,
        "timings_ms": timing_medians,
        "timings_stats": {
            "p50": round(_percentile(totals, 50), 3),
            "p95": round(_percentile(totals, 95), 3),
            "n_repeats": request.repeats,
        },
        "diagnostics": diagnostics,
        "results": results,
    }
