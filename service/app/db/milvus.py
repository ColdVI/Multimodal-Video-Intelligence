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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    name = f"segments_{dimension}"
    filters = [f'dataset_id == "{dataset_id}"']
    if candidate_ids is not None:
        if not candidate_ids:
            return [], {"plan_used_vector_index": False, "indexed_vectors_count": None, "notes": []}
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
    return rows, {"plan_used_vector_index": True, "indexed_vectors_count": None, "notes": []}

