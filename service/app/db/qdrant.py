from __future__ import annotations

import hashlib
from typing import Any

from qdrant_client import QdrantClient, models

from app.config import settings

DIMS = (2048, 1024, 512, 256)
PAYLOAD_FIELDS = {
    "dataset_id": models.PayloadSchemaType.KEYWORD,
    "video_id": models.PayloadSchemaType.KEYWORD,
    "segment_id": models.PayloadSchemaType.KEYWORD,
    "altitude_m": models.PayloadSchemaType.FLOAT,
    "velocity_mps": models.PayloadSchemaType.FLOAT,
    "gimbal_pitch": models.PayloadSchemaType.FLOAT,
    "person_count": models.PayloadSchemaType.INTEGER,
}


def client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, timeout=30)


def point_id(segment_id: str) -> int:
    return int(hashlib.sha256(segment_id.encode()).hexdigest()[:15], 16)


def collection(dim: int) -> str:
    return f"segments_{dim}"


def init_schema() -> None:
    db = client()
    existing = {item.name for item in db.get_collections().collections}
    for dim in DIMS:
        name = collection(dim)
        if name not in existing:
            db.create_collection(name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE), hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128, full_scan_threshold=1))
        # Payload indexes must exist before points are inserted so filtered HNSW is built correctly.
        for field, schema in PAYLOAD_FIELDS.items():
            db.create_payload_index(name, field_name=field, field_schema=schema, wait=True)


def healthy() -> bool:
    try:
        client().get_collections(); return True
    except Exception:
        return False


def upsert(rows: list[dict[str, Any]], vectors: dict[int, list[tuple[str, Any]]]) -> None:
    by_id = {r["segment_id"]: r for r in rows}
    db = client()
    for dim, entries in vectors.items():
        points = []
        for segment_id, vector in entries:
            row = by_id[segment_id]
            payload = {k:row.get(k) for k in PAYLOAD_FIELDS if row.get(k) is not None}
            points.append(models.PointStruct(id=point_id(segment_id), vector=vector.tolist(), payload=payload))
        for start in range(0, len(points), 256):
            db.upsert(collection(dim), points[start:start+256], wait=True)


def _range(value: list | tuple) -> models.Range:
    return models.Range(gte=value[0] if value[0] is not None else None, lte=value[1] if value[1] is not None else None)


def vector_search(dataset_id: str, vector: Any, dim: int, top_k: int, strategy: str,
                  candidate_ids: list[str] | None, telemetry: dict | None) -> tuple[list[tuple[str,float]], dict]:
    must: list[Any] = [models.FieldCondition(key="dataset_id", match=models.MatchValue(value=dataset_id))]
    for key in ("altitude_m", "velocity_mps", "gimbal_pitch"):
        if (telemetry or {}).get(key): must.append(models.FieldCondition(key=key, range=_range(telemetry[key])))
    if candidate_ids is not None:
        if not candidate_ids: return [], {"indexed_vectors_count": _indexed(dim)}
        must.append(models.HasIdCondition(has_id=[point_id(x) for x in candidate_ids]))
    exact = strategy == "exact"
    if strategy not in {"exact","ann","ann_high_ef","prefilter"}: raise ValueError(f"unsupported Qdrant strategy: {strategy}")
    ef = 512 if strategy == "ann_high_ef" else 128
    result = client().query_points(collection_name=collection(dim), query=vector.tolist(), query_filter=models.Filter(must=must), limit=top_k, search_params=models.SearchParams(hnsw_ef=ef, exact=exact), with_payload=True).points
    return [(str(p.payload["segment_id"]), float(p.score)) for p in result], {"indexed_vectors_count": _indexed(dim)}


def _indexed(dim: int) -> int:
    info = client().get_collection(collection(dim))
    return int(info.indexed_vectors_count or 0)


def count(dataset_id: str, dim: int = 512) -> int:
    result = client().count(collection(dim), count_filter=models.Filter(must=[models.FieldCondition(key="dataset_id",match=models.MatchValue(value=dataset_id))]), exact=True)
    return int(result.count)
