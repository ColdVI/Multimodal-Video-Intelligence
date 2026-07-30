from __future__ import annotations

import uuid
from typing import Any, Iterable

from app.config import DIMENSIONS, settings
from app.search.strategies import qdrant_search_params


_NAMESPACE = uuid.UUID("17c52443-1998-443e-8c70-00544a9f6ee9")


def point_id(segment_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, segment_id))


def client():
    from qdrant_client import QdrantClient

    return QdrantClient(url=settings.qdrant_url, timeout=30)


def collection_name(dimension: int) -> str:
    return f"segments_{dimension}"


def health() -> bool:
    try:
        client().get_collections()
        return True
    except Exception:
        return False


def init_schema(dimensions: tuple[int, ...] = DIMENSIONS) -> None:
    from qdrant_client import models

    target = client()
    indexes = {
        "dataset_id": models.PayloadSchemaType.KEYWORD,
        "video_id": models.PayloadSchemaType.KEYWORD,
        "altitude_m": models.PayloadSchemaType.FLOAT,
        "velocity_mps": models.PayloadSchemaType.FLOAT,
        "gimbal_pitch": models.PayloadSchemaType.FLOAT,
        "person_count": models.PayloadSchemaType.INTEGER,
    }
    for dimension in dimensions:
        name = collection_name(dimension)
        if not target.collection_exists(name):
            target.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=100),
            )
        for field_name, field_schema in indexes.items():
            target.create_payload_index(name, field_name=field_name, field_schema=field_schema, wait=True)


def replace_vectors(dataset_id: str, dimension: int, rows: list[dict[str, Any]]) -> None:
    from qdrant_client import models

    target = client()
    name = collection_name(dimension)
    target.delete(
        collection_name=name,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=[models.FieldCondition(key="dataset_id", match=models.MatchValue(value=dataset_id))])
        ),
        wait=True,
    )
    batch: list[Any] = []
    for row in rows:
        payload = {key: value for key, value in row.items() if key != "embedding" and value is not None}
        batch.append(models.PointStruct(id=point_id(row["segment_id"]), vector=row["embedding"], payload=payload))
        if len(batch) == 256:
            target.upsert(name, points=batch, wait=True)
            batch.clear()
    if batch:
        target.upsert(name, points=batch, wait=True)


def _filter(dataset_id: str, candidate_ids: list[str] | None):
    from qdrant_client import models

    must: list[Any] = [models.FieldCondition(key="dataset_id", match=models.MatchValue(value=dataset_id))]
    if candidate_ids is not None:
        must.append(models.HasIdCondition(has_id=[point_id(value) for value in candidate_ids]))
    return models.Filter(must=must)


def search_vectors(
    dataset_id: str,
    dimension: int,
    query_vector: Iterable[float],
    top_k: int,
    strategy: str,
    candidate_ids: list[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from qdrant_client import models

    if candidate_ids is not None and not candidate_ids:
        return [], diagnostics(dimension)
    params = models.SearchParams(**qdrant_search_params(strategy))
    target = client()
    result = target.query_points(
        collection_name=collection_name(dimension),
        query=list(query_vector),
        query_filter=_filter(dataset_id, candidate_ids),
        search_params=params,
        limit=top_k,
        with_payload=True,
    )
    rows = [
        {"segment_id": point.payload["segment_id"], "score": float(point.score)}
        for point in result.points
    ]
    return rows, diagnostics(dimension)


def diagnostics(dimension: int) -> dict[str, Any]:
    info = client().get_collection(collection_name(dimension))
    indexed = getattr(info, "indexed_vectors_count", None)
    notes = []
    if indexed == 0:
        notes.append("indexed_vectors_count=0; Qdrant latency brute-force")
    return {"plan_used_vector_index": bool(indexed), "indexed_vectors_count": indexed, "notes": notes}


def table_count(dataset_id: str, dimension: int) -> int:
    from qdrant_client import models

    result = client().count(
        collection_name=collection_name(dimension),
        count_filter=models.Filter(
            must=[models.FieldCondition(key="dataset_id", match=models.MatchValue(value=dataset_id))]
        ),
        exact=True,
    )
    return int(result.count)
