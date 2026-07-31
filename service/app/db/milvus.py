from __future__ import annotations

from typing import Any, Iterable

from app.config import settings


def _client():
    from pymilvus import MilvusClient

    return MilvusClient(uri=settings.milvus_uri)


def health() -> bool:
    try:
        _client().list_collections()
        return True
    except Exception:
        return False


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
    """Incomplete/experimental adapter: no run-scoped collection and no metadata/
    telemetry filter pushdown exist for Milvus in this codebase. Raises rather than
    silently searching the wrong (non-run or unfiltered) collection -- see
    docs/operations/BACKEND_SELECTION.md and artifacts/baseline_contract/backend_reachability.json."""
    if run_id is not None:
        raise ValueError("milvus adapter has no run-scoped collection; run-scoped search is not supported")
    if metadata_filters or telemetry_filters:
        raise ValueError("milvus adapter has no metadata/telemetry filter pushdown; only candidate_ids restriction is supported")
    name = f"segments_{dimension}"
    filters = [f'dataset_id == "{dataset_id}"']
    if candidate_ids is not None:
        if not candidate_ids:
            return [], {
                "plan_used_vector_index": None, "indexed_vectors_count": None, "notes": [],
                "filtered_corpus_count": 0, "candidate_input_count": 0, "candidate_count": 0,
                "candidate_count_status": "computed", "explain_status": "not_applicable_for_backend",
            }
        quoted = ",".join(repr(value) for value in candidate_ids)
        filters.append(f"segment_id in [{quoted}]")
    result = _client().search(
        collection_name=name,
        data=[list(query_vector)],
        filter=" and ".join(filters),
        limit=top_k,
        output_fields=["segment_id"],
        search_params={"metric_type": "COSINE", "params": {"nprobe": 32}},
    )[0]
    rows = [{"segment_id": hit["entity"]["segment_id"], "score": float(hit["distance"])} for hit in result]
    candidate_input_count = len(candidate_ids) if candidate_ids is not None else None
    return rows, {
        "plan_used_vector_index": True, "indexed_vectors_count": None, "notes": [],
        "filtered_corpus_count": None, "candidate_input_count": candidate_input_count,
        "candidate_count": candidate_input_count, "candidate_count_status": "not_requested",
        "explain_status": "not_applicable_for_backend",
    }

