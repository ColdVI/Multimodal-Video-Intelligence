from __future__ import annotations

import statistics
import time
from typing import Any, Callable

import numpy as np

from app.config import settings
from app.db import clickhouse, milvus, postgres, qdrant
from app.embedding.router import embed_query, embed_query_multi
from app.query_planning.models import QueryExecutionPlan, RelaxationPolicy, constraints_to_filter_dict
from app.query_planning.planner import plan_query
from app.query_planning.relaxation import RelaxationOutcome, run_relaxation_ladder
from app.search import common_exact
from app.search.strategies import validate_strategy
from app.search.pushdown import matches, normalize_filters


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
    request: Any, active_snapshot: dict[str, Any] | None, execution_mode: str,
    *, metadata_filters: dict[str, Any], telemetry_filters: dict[str, Any],
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any], int | None, set[str]]:
    started = time.perf_counter()
    filter_started = time.perf_counter()
    run_id = None if active_snapshot is None else str(active_snapshot["run_id"])
    native_pushdown = execution_mode == "pushdown"
    if native_pushdown:
        candidate_ids = None
    elif run_id is None:
        candidate_ids = postgres.filter_segment_ids(
            request.dataset_id, metadata_filters, telemetry_filters,
        )
    else:
        candidate_ids = postgres.filter_run_segment_ids(
            request.dataset_id, run_id, metadata_filters, telemetry_filters,
        )
    filter_ms = (time.perf_counter() - filter_started) * 1000.0
    candidate_set = set(candidate_ids or [])
    if not native_pushdown and len(candidate_set) > settings.legacy_candidate_limit:
        raise ValueError(
            f"legacy candidate limit exceeded: {len(candidate_set)} > {settings.legacy_candidate_limit}; use pushdown"
        )
    if candidate_ids == []:
        total_ms = (time.perf_counter() - started) * 1000.0
        timings = {"filter": filter_ms, "embed": 0.0, "vector_search": 0.0, "hydrate": 0.0, "total": total_ms}
        empty_diagnostics = {
            "plan_used_vector_index": None, "indexed_vectors_count": None,
            "notes": ["filters matched zero candidates"],
            "filtered_corpus_count": 0, "candidate_input_count": 0, "candidate_count": 0,
            "candidate_count_status": "computed", "explain_status": "not_requested",
        }
        return timings, [], empty_diagnostics, 0, candidate_set

    diagnose = bool(getattr(request, "diagnose", False))
    explain = bool(getattr(request, "explain", False))
    search_extra: dict[str, Any] = {}
    if diagnose:
        search_extra["diagnose"] = True
    if explain:
        search_extra["explain"] = True

    embed_started = time.perf_counter()
    search_function = BACKENDS[request.backend]
    if request.adaptive_mrl.enabled:
        vectors = embed_query_multi(request.query, (request.dimension, request.adaptive_mrl.base_dim))
        query_vector = vectors[request.dimension]
        base_vector = vectors[request.adaptive_mrl.base_dim]
    else:
        query_vector = embed_query(request.query, request.dimension)
    embed_ms = (time.perf_counter() - embed_started) * 1000.0

    search_started = time.perf_counter()
    if request.adaptive_mrl.enabled:
        search_kwargs = {} if run_id is None else {"run_id": run_id}
        if native_pushdown:
            search_kwargs.update(metadata_filters=metadata_filters, telemetry_filters=telemetry_filters)
        base_hits, base_diagnostics = search_function(
            request.dataset_id,
            request.adaptive_mrl.base_dim,
            base_vector,
            request.adaptive_mrl.top_n,
            request.strategy,
            candidate_ids, **search_kwargs, **search_extra,
        )
        rerank_ids = [hit["segment_id"] for hit in base_hits]
        exact_rerank = bool(getattr(request.adaptive_mrl, "exact_rerank", False)) or settings.adaptive_mrl_exact_rerank
        if rerank_ids and exact_rerank:
            # EXPERIMENTAL (plan Sec.4.5 physical-read gate failed at 20K-row measured
            # scale -- artifacts/advanced_retrieval/adaptive_mrl/physical_read_gate_summary.json):
            # stage-2 becomes a true exact rerank restricted to rerank_ids, never the
            # same ANN strategy as stage-1. Only ClickHouse is wired; other backends
            # report exact_rerank_unsupported and the caller should not have set this
            # flag for them (main.py does not currently reject it per-backend -- a
            # request-time validation gap tracked in KNOWN_LIMITATIONS.md).
            from app.search.exact_rerank import rerank_candidates_exact

            hits, stage2_diagnostics = rerank_candidates_exact(
                request.dataset_id, request.dimension, query_vector, rerank_ids, request.top_k,
                run_id=run_id, backend=request.backend,
            )
        elif rerank_ids:
            hits, stage2_diagnostics = search_function(
                request.dataset_id,
                request.dimension,
                query_vector,
                request.top_k,
                request.strategy,
                rerank_ids, **search_kwargs, **search_extra,
            )
        else:
            hits, stage2_diagnostics = [], dict(base_diagnostics)
        # Stage-1 (base_dim/top_n over the filtered corpus) and stage-2 (dimension/top_k
        # rerank of only the stage-1 candidate IDs) are surfaced as distinct diagnostics
        # rather than collapsing to whichever value stage-2 happened to report -- stage-2's
        # own candidate_count reflects the rerank_ids restriction, not the filtered corpus.
        # filtered_corpus_count is only non-None when stage-1 itself ran diagnose/underfilled
        # (it's a conditional count query); candidate_shortage/ann_filter_loss must instead
        # key off stage1_returned_candidate_count, which is always known for free (it's just
        # len(rerank_ids)) and is exactly the plan's "was there enough for Stage-2 to work
        # with" question -- unlike filtered_corpus_count, it can never silently be None.
        stage1_returned_candidate_count = len(rerank_ids)
        diagnostics = dict(stage2_diagnostics)
        diagnostics["filtered_corpus_count"] = base_diagnostics.get("filtered_corpus_count")
        diagnostics["stage1_requested_candidate_count"] = request.adaptive_mrl.top_n
        diagnostics["stage1_returned_candidate_count"] = stage1_returned_candidate_count
        diagnostics["stage2_input_candidate_count"] = stage1_returned_candidate_count
        diagnostics["stage2_returned_count"] = len(hits)
        diagnostics["final_returned_count"] = len(hits)
        diagnostics["candidate_count"] = stage1_returned_candidate_count
        diagnostics.setdefault("notes", []).append(
            f"adaptive MRL {request.adaptive_mrl.base_dim}→{request.dimension}, top_n={request.adaptive_mrl.top_n}"
        )
    else:
        search_kwargs = {} if run_id is None else {"run_id": run_id}
        if native_pushdown:
            search_kwargs.update(metadata_filters=metadata_filters, telemetry_filters=telemetry_filters)
        hits, diagnostics = search_function(
            request.dataset_id,
            request.dimension,
            query_vector,
            request.top_k,
            request.strategy,
            candidate_ids, **search_kwargs, **search_extra,
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
    raw_candidate_count = diagnostics.get("candidate_count")
    candidate_count = (
        raw_candidate_count
        if raw_candidate_count is not None
        else len(candidate_ids) if candidate_ids is not None
        else None
    )
    return timings, results, diagnostics, candidate_count, candidate_set


def _plan_query_for_request(request: Any, parser_mode: str) -> QueryExecutionPlan | None:
    """parser_mode="none" (the default) skips this entirely -- no extra Postgres round
    trip for available_field_names() on the hot path when nothing needs it."""
    if parser_mode == "none":
        return None
    from app.query_planning.coverage import available_field_names

    available_fields = available_field_names(request.dataset_id)
    relaxation_mode = getattr(request, "filter_relaxation_mode", None) or settings.filter_relaxation_mode
    requested_min_results = getattr(request, "min_results", None)
    min_results = request.top_k if requested_min_results is None else requested_min_results
    return plan_query(
        request.query, available_fields, parser_mode=parser_mode,
        llm_provider=settings.query_parser_llm_provider,
        llm_model_id=settings.query_parser_llm_model_id,
        llm_base_url=settings.query_parser_llm_base_url,
        relaxation_policy=RelaxationPolicy(
            mode=relaxation_mode,
            min_results=min_results,
            max_relaxation_passes=getattr(request, "max_relaxation_passes", 5),
            relaxation_timeout_ms=getattr(request, "relaxation_timeout_ms", 2000.0),
            allow_semantic_only_fallback=getattr(request, "allow_semantic_only_fallback", False),
        ),
    )


def _merge_planned_filters(request: Any, plan: QueryExecutionPlan | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parser-derived constraints are soft, additive, and never override an explicit
    request filter on the same field -- explicit values are applied last so they win."""
    if plan is None:
        return dict(request.metadata_filters), dict(request.telemetry_filters)
    merged = plan.active_filter_dict()
    merged.update(request.metadata_filters)
    merged.update(request.telemetry_filters)
    return merged, {}


def _run_relaxation(
    request: Any, active_snapshot: dict[str, Any] | None, execution_mode: str,
    plan: QueryExecutionPlan, base_telemetry_filters: dict[str, Any], known_pass1_count: int,
) -> RelaxationOutcome:
    """The ladder only ever relaxes plan.soft_constraints (parser-derived). Explicit
    request filters are never part of the candidate pool it walks -- they are re-applied
    on every probe via base_metadata_filters/request.metadata_filters, so "manual filter
    never removed" holds by construction, not by a check inside relaxation.py."""

    def search_fn(constraints: tuple) -> int:
        filters = constraints_to_filter_dict(constraints)
        filters.update(request.metadata_filters)
        _, probe_results, _, _, _ = _one_run(
            request, active_snapshot, execution_mode,
            metadata_filters=filters, telemetry_filters=base_telemetry_filters,
        )
        return len(probe_results)

    return run_relaxation_ladder(
        plan.hard_constraints, plan.soft_constraints, plan.relaxation_policy, search_fn,
        known_pass1_count=known_pass1_count,
    )


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
    requested_mode = getattr(request, "filter_execution_mode", None) or settings.filter_execution_mode
    execution_mode = requested_mode if active_snapshot is not None else "legacy_candidate_ids_compatibility"

    parser_mode = getattr(request, "parser_mode", None) or settings.query_parser_mode
    plan = _plan_query_for_request(request, parser_mode)
    merged_metadata_filters, merged_telemetry_filters = _merge_planned_filters(request, plan)

    totals: list[float] = []
    timing_rows: list[dict[str, float]] = []
    results: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    candidate_count: int | None = None
    candidate_set: set[str] = set()
    for _ in range(request.repeats):
        timings, results, diagnostics, candidate_count, candidate_set = _one_run(
            request, active_snapshot, execution_mode,
            metadata_filters=merged_metadata_filters, telemetry_filters=merged_telemetry_filters,
        )
        timing_rows.append(timings)
        totals.append(timings["total"])

    timing_medians = {
        stage: round(statistics.median(row[stage] for row in timing_rows), 3)
        for stage in ("filter", "embed", "vector_search", "hydrate", "total")
    }

    relaxation_outcome = None
    if plan is not None:
        relaxation_outcome = _run_relaxation(
            request, active_snapshot, execution_mode, plan, merged_telemetry_filters, len(results),
        )
        if plan.relaxation_policy.mode == "auto_soft" and relaxation_outcome.selected_pass != 1:
            selected = relaxation_outcome.passes[relaxation_outcome.selected_pass - 1]
            merged_metadata_filters = constraints_to_filter_dict(selected.active_constraints)
            merged_metadata_filters.update(request.metadata_filters)
            merged_telemetry_filters = dict(request.telemetry_filters)
            timings, results, diagnostics, candidate_count, candidate_set = _one_run(
                request, active_snapshot, execution_mode,
                metadata_filters=merged_metadata_filters, telemetry_filters=merged_telemetry_filters,
            )
            timing_medians = {stage: round(timings[stage], 3) for stage in timing_medians}

    returned_ids = [row["segment_id"] for row in results]
    if execution_mode == "pushdown":
        predicates = normalize_filters(merged_metadata_filters, merged_telemetry_filters)
        filter_correctness = all(matches(row, predicates) for row in results)
    else:
        filter_correctness = all(segment_id in candidate_set for segment_id in returned_ids)
    underfilled = len(results) < request.top_k
    candidate_shortage = None if candidate_count is None else underfilled and candidate_count < request.top_k
    ann_filter_loss = None if candidate_count is None else underfilled and candidate_count >= request.top_k
    diagnostics = {
        "candidate_count": candidate_count,
        "returned_count": len(results),
        "underfilled": underfilled,
        "underfilled_reason": (
            "not_measured" if underfilled and candidate_count is None
            else "candidate_shortage" if candidate_shortage
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
        # Additive Phase -1 diagnostics: filtered_corpus_count/candidate_input_count/
        # candidate_count_status/explain_status are always present; only non-None when
        # actually measured (diagnose=True or the search was underfilled). stage1_*/
        # stage2_*/final_returned_count are only populated in adaptive MRL mode.
        "filtered_corpus_count": diagnostics.get("filtered_corpus_count"),
        "candidate_input_count": diagnostics.get("candidate_input_count"),
        "candidate_count_status": diagnostics.get("candidate_count_status"),
        "explain_status": diagnostics.get("explain_status"),
        "stage1_requested_candidate_count": diagnostics.get("stage1_requested_candidate_count"),
        "stage1_returned_candidate_count": diagnostics.get("stage1_returned_candidate_count"),
        "stage2_input_candidate_count": diagnostics.get("stage2_input_candidate_count"),
        "stage2_returned_count": diagnostics.get("stage2_returned_count"),
        "final_returned_count": diagnostics.get("final_returned_count", len(results)),
        "filter_relaxation": relaxation_outcome.as_diagnostics() if relaxation_outcome is not None else None,
        "quality_vs_groundtruth": None,
        "r_at_1": None if settings.embedding_mode == "synthetic" else None,
        "ndcg": None if settings.embedding_mode == "synthetic" else None,
        "run_id": None if active_snapshot is None else active_snapshot["run_id"],
        "filter_execution_mode": execution_mode,
        "candidate_count_source": "backend_pushdown" if execution_mode == "pushdown" else "postgres_candidate_ids",
        "filter_pushdown_backend": request.backend if execution_mode == "pushdown" else None,
        "model_id": None if active_snapshot is None else active_snapshot.get("model_id"),
        "model_revision": None if active_snapshot is None else active_snapshot.get("model_revision"),
        "vector_provenance": (
            dataset["vector_provenance"] if active_snapshot is None else active_snapshot["vector_provenance"]
        ),
        "parser": plan.parsed_query.diagnostics.as_dict() if plan is not None else None,
    }
    return {
        "embedding_mode": settings.embedding_mode,
        "dataset_id": request.dataset_id,
        "run_id": None if active_snapshot is None else active_snapshot["run_id"],
        "dataset_version": None if active_snapshot is None else active_snapshot.get("dataset_version"),
        "vector_provenance": (
            dataset["vector_provenance"] if active_snapshot is None else active_snapshot["vector_provenance"]
        ),
        "model_id": None if active_snapshot is None else active_snapshot.get("model_id"),
        "model_revision": None if active_snapshot is None else active_snapshot.get("model_revision"),
        "source_commit": None if active_snapshot is None else active_snapshot.get("source_commit"),
        "filter_execution_mode": execution_mode,
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
        "execution_policy": {
            "parser_mode": parser_mode,
            "filter_relaxation_mode": getattr(request, "filter_relaxation_mode", None) or settings.filter_relaxation_mode,
            "adaptive_exact_rerank": bool(
                getattr(request, "adaptive_mrl", None) and getattr(request.adaptive_mrl, "exact_rerank", False)
            ) or settings.adaptive_mrl_exact_rerank,
            "vlm_rerank_mode": settings.vlm_rerank_mode,
        },
        "results": results,
    }
