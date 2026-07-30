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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    params: dict[str, Any] = {"dataset_id": dataset_id}
    clause = ""
    if candidate_ids is not None:
        if not candidate_ids:
            return [], {"plan_used_vector_index": False, "indexed_vectors_count": None, "notes": ["stable float32 reference"]}
        clause = " AND segment_id IN {candidate_ids:Array(String)}"
        params["candidate_ids"] = candidate_ids
    result = clickhouse.client().query(
        f"SELECT segment_id,embedding FROM seg_ch_{dimension} "
        f"WHERE dataset_id={{dataset_id:String}}{clause} ORDER BY segment_id",
        parameters=params,
    )
    rows = result.result_rows
    ids = [row[0] for row in rows]
    matrix = np.stack([_parse_vector(row[1]) for row in rows]) if rows else np.empty((0, dimension), dtype=np.float32)
    result = stable_top_k(matrix, np.asarray(query_vector, dtype=np.float32), ids, top_k) if rows else []
    return result, {"plan_used_vector_index": False, "indexed_vectors_count": None, "notes": ["stable float32 reference"]}
