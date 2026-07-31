from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from app.db import clickhouse


def stable_top_k(
    embeddings: np.ndarray,
    query: np.ndarray,
    ids: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    matrix = np.asarray(embeddings, dtype=np.float32)
    vector = np.asarray(query, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(ids) or matrix.shape[1] != vector.shape[0]:
        raise ValueError("numpy_exact matrix/query/id shape mismatch")
    scores = matrix @ vector
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [{"segment_id": ids[int(i)], "score": float(scores[int(i)])} for i in order]


def _parse_vector(value: Any) -> np.ndarray:
    if isinstance(value, str):
        return np.fromstring(value.strip("[]"), dtype=np.float32, sep=",")
    return np.asarray(value, dtype=np.float32)


def search_vectors(
    dataset_id: str,
    dimension: int,
    query_vector: Iterable[float],
    top_k: int,
    strategy: str,
    candidate_ids: list[str] | None,
    *,
    run_id: str | None = None,
    metadata_filters: dict[str, Any] | None = None,
    telemetry_filters: dict[str, Any] | None = None,
    diagnose: bool = False,
    explain: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Benchmark correctness reference: brute-force float32 cosine over the run-scoped or
    base table. Has no native metadata/telemetry filter pushdown -- callers must resolve
    filters to candidate_ids first (legacy_candidate_ids mode); passing filters without
    candidate_ids raises rather than silently returning unfiltered results."""
    if candidate_ids is None and (metadata_filters or telemetry_filters):
        raise ValueError(
            "numpy_exact has no native filter pushdown; resolve metadata_filters/"
            "telemetry_filters to candidate_ids before calling, or omit them."
        )
    notes = ["stable float32 reference"]
    table = f"seg_ch_{dimension}_runs" if run_id is not None else f"seg_ch_{dimension}"
    params: dict[str, Any] = {"dataset_id": dataset_id}
    run_clause = ""
    if run_id is not None:
        params["run_id"] = run_id
        run_clause = " AND run_id={run_id:UUID}"
    clause = ""
    if candidate_ids is not None:
        if not candidate_ids:
            return [], {
                "plan_used_vector_index": None, "indexed_vectors_count": None, "notes": notes,
                "filtered_corpus_count": 0, "candidate_input_count": 0, "candidate_count": 0,
                "candidate_count_status": "computed", "explain_status": "not_applicable_for_backend",
            }
        clause = " AND segment_id IN {candidate_ids:Array(String)}"
        params["candidate_ids"] = candidate_ids
    result = clickhouse.client().query(
        f"SELECT segment_id,embedding FROM {table} "
        f"WHERE dataset_id={{dataset_id:String}}{run_clause}{clause} ORDER BY segment_id",
        parameters=params,
    )
    rows = result.result_rows
    ids = [row[0] for row in rows]
    matrix = np.stack([_parse_vector(row[1]) for row in rows]) if rows else np.empty((0, dimension), dtype=np.float32)
    top = stable_top_k(matrix, np.asarray(query_vector, dtype=np.float32), ids, top_k) if rows else []
    return top, {
        "plan_used_vector_index": None, "indexed_vectors_count": None, "notes": notes,
        "filtered_corpus_count": len(ids), "candidate_input_count": len(candidate_ids) if candidate_ids is not None else None,
        "candidate_count": len(ids), "candidate_count_status": "computed",
        "explain_status": "not_applicable_for_backend",
    }
