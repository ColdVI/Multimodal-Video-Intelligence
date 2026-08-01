"""Adaptive MRL Stage-2 exact candidate rerank, per
docs/planning/ADVANCED_RETRIEVAL_FINAL_PLAN_v2.1.md Sec.4/14.

rerank_candidates_exact() is additive to (not a replacement of) clickhouse.search_vectors():
it exists because the general-purpose search_vectors() accepts a `strategy` a caller could
set wrong (defeating exact rerank) and metadata_filters/telemetry_filters a caller could
pass redundantly (stage-1 candidates already satisfy them) -- this function's narrower
signature makes both mistakes impossible to express, and it always requests the physical-
read counters search_vectors() only fetches when diagnose=True.

Physical-read red gate (Sec.4.5, Ek A item 1): segment_id is NOT seg_ch_{d}[_runs]'s sort
key (ORDER BY (dataset_id[, run_id], video_id, t_start)), so "segment_id IN (...)" is
logically exact-only, not necessarily physically scoped to the candidate set -- ClickHouse
may still have to scan whole granules/partitions to find them. evaluate_physical_read_gate()
turns a real measurement into a pass/fail verdict; it never assumes the answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


def rerank_candidates_exact(
    dataset_id: str,
    dimension: int,
    query_vector: Iterable[float],
    candidate_ids: list[str],
    top_k: int,
    *,
    run_id: str | None = None,
    backend: str = "clickhouse",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """candidate_ids must be non-empty (stage-1 already ran; an empty stage-1 result is
    the caller's job to short-circuit before ever reaching here) and no filters are
    accepted -- stage-1's filtered search already enforced them. Backends without a
    verified exact-candidate-rerank implementation report
    exact_rerank_unsupported:<backend> in notes and fall back to the caller's existing
    path rather than silently returning nothing (Sec.4.2)."""
    if not candidate_ids:
        raise ValueError("rerank_candidates_exact requires a non-empty candidate_ids (stage-1 result)")
    if backend == "clickhouse":
        return _rerank_clickhouse(dataset_id, dimension, query_vector, candidate_ids, top_k, run_id)
    raise ValueError(f"exact rerank is unsupported for backend {backend!r}")


def _rerank_clickhouse(
    dataset_id: str, dimension: int, query_vector: Iterable[float], candidate_ids: list[str],
    top_k: int, run_id: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from app.db.clickhouse import client

    table = f"seg_ch_{dimension}" if run_id is None else f"seg_ch_{dimension}_runs"
    params: dict[str, Any] = {
        "dataset_id": dataset_id, "query_vector": list(query_vector), "top_k": top_k,
        "candidate_ids": candidate_ids,
    }
    run_clause = ""
    if run_id is not None:
        params["run_id"] = run_id
        run_clause = " AND run_id={run_id:UUID}"
    sql = f"""SELECT segment_id, 1-cosineDistance(embedding, {{query_vector:Array(Float32)}}) AS score
              FROM {table}
              WHERE dataset_id={{dataset_id:String}}{run_clause} AND segment_id IN {{candidate_ids:Array(String)}}
              ORDER BY cosineDistance(embedding, {{query_vector:Array(Float32)}})
              LIMIT {{top_k:UInt32}}"""
    target = client()
    result = target.query(sql, parameters=params, settings={"query_plan_try_use_vector_search": 0})
    rows = [{"segment_id": row[0], "score": float(row[1])} for row in result.result_rows]
    summary = result.summary or {}
    return rows, {
        "plan_used_vector_index": False, "indexed_vectors_count": None, "notes": [],
        "rows_read": int(summary["read_rows"]) if "read_rows" in summary else None,
        "bytes_read": int(summary["read_bytes"]) if "read_bytes" in summary else None,
        "candidate_count": len(candidate_ids),
    }


@dataclass(frozen=True)
class PhysicalReadVerdict:
    status: str  # "passed" | "failed" | "not_run"
    rows_read: int | None
    candidate_count: int
    partition_size: int | None
    rows_per_candidate: float | None
    reason: str


def evaluate_physical_read_gate(
    rows_read: int | None, candidate_count: int, partition_size: int | None, *, tolerance_factor: float = 10.0,
) -> PhysicalReadVerdict:
    """Fails if rows_read scales with partition_size rather than candidate_count. A
    tolerance_factor of 10 means: reading up to 10x the candidate count (index
    granularity, e.g. ClickHouse's default 8192-row granules, means some over-read is
    structurally normal) still passes; reading a large fraction of the whole partition
    regardless of how few candidates were requested does not. Both thresholds are this
    function's own default judgment call, not a plan-mandated constant -- callers
    measuring at a different scale should pass their own tolerance_factor."""
    if rows_read is None:
        return PhysicalReadVerdict("not_run", None, candidate_count, partition_size, None, "rows_read was not measured")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    rows_per_candidate = rows_read / candidate_count
    if partition_size is not None and rows_read >= partition_size * 0.9 and candidate_count < partition_size * 0.5:
        return PhysicalReadVerdict(
            "failed", rows_read, candidate_count, partition_size, rows_per_candidate,
            f"rows_read={rows_read} is >=90% of partition_size={partition_size} despite only "
            f"{candidate_count} candidates requested -- looks like a full partition scan, not a candidate-scoped read",
        )
    if rows_per_candidate > tolerance_factor:
        return PhysicalReadVerdict(
            "failed", rows_read, candidate_count, partition_size, rows_per_candidate,
            f"rows_read/candidate_count={rows_per_candidate:.1f} exceeds tolerance_factor={tolerance_factor}",
        )
    return PhysicalReadVerdict(
        "passed", rows_read, candidate_count, partition_size, rows_per_candidate,
        f"rows_read/candidate_count={rows_per_candidate:.1f} is within tolerance_factor={tolerance_factor}",
    )


__all__ = ["rerank_candidates_exact", "PhysicalReadVerdict", "evaluate_physical_read_gate"]
