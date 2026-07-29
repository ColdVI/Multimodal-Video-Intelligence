"""Qdrant ince sarmalayici (spec SS4.5). Install/start/health/stop/cleanup
scripts/install_qdrant_colab.sh'ya delege eder. Payload index'leri INGEST'TEN
ONCE olusturulmali (spec SS0 bulgu #6) - create_collection_and_indexes()
bu sirayi ZORUNLU kilar, cagiran taraf sirayi bozamaz."""
from ._runner import run_action

SCRIPT = "install_qdrant_colab.sh"
HOT_FILTER_FIELDS = [
    ("altitude_m", "float"), ("velocity_mps", "float"),
    ("person_count", "integer"), ("vehicle_count", "integer"),
    ("is_night", "integer"),
]


def install() -> dict:
    return run_action(SCRIPT, "install", timeout=300)


def start() -> dict:
    return run_action(SCRIPT, "start", timeout=60)


def health_check(timeout: int = 35) -> dict:
    result = run_action(SCRIPT, "health", timeout=timeout)
    return {"healthy": result["ok"], **result}


def stop() -> dict:
    return run_action(SCRIPT, "stop", timeout=30)


def cleanup() -> dict:
    return run_action(SCRIPT, "cleanup", timeout=60)


def get_client(host: str = "127.0.0.1", port: int = 6333):
    from qdrant_client import QdrantClient
    return QdrantClient(host=host, port=port)


def create_collection_and_indexes(client, collection_name: str, dimension: int,
                                  m: int = 16, ef_construct: int = 128,
                                  full_scan_threshold: int = 10000) -> None:
    """SIRA KRITIK: koleksiyon -> payload index'leri -> (cagiran taraf) ingest.
    Ters sirada filtreli HNSW'nin filtre-farkinda kenarlari KURULMAZ."""
    from qdrant_client.models import Distance, HnswConfigDiff, VectorParams

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=m, ef_construct=ef_construct, full_scan_threshold=full_scan_threshold),
    )
    for field, schema in HOT_FILTER_FIELDS:
        client.create_payload_index(collection_name, field, schema)


def indexed_vectors_count(client, collection_name: str) -> dict:
    """spec SS4.5: indexed_vectors_count==0 -> HNSW HIC KURULMAMIS, o kosunun
    tum latency sayilari brute-force'tur, boyle RAPORLANMALI."""
    info = client.get_collection(collection_name)
    return {
        "points_count": info.points_count,
        "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
        "status": str(info.status),
        "hnsw_actually_built": bool(getattr(info, "indexed_vectors_count", 0)),
    }


def query(client, collection_name: str, qvec, top_k: int, filter_conditions: dict = None,
         exact: bool = False, hnsw_ef: int = None):
    """filter_conditions: {"altitude_m": {"lt": 20.0}} gibi basit
    alan->kosul sozlugu. exact=True -> spec SS7.3 strategy-2 (params exact)."""
    from qdrant_client.models import FieldCondition, Filter, Range, SearchParams

    must = []
    if filter_conditions:
        for field, cond in filter_conditions.items():
            must.append(FieldCondition(key=field, range=Range(**cond)))
    query_filter = Filter(must=must) if must else None
    params = SearchParams(exact=exact, hnsw_ef=hnsw_ef) if (exact or hnsw_ef) else None
    return client.search(collection_name=collection_name, query_vector=qvec, limit=top_k,
                         query_filter=query_filter, search_params=params)


__all__ = ["install", "start", "health_check", "stop", "cleanup", "get_client",
          "create_collection_and_indexes", "indexed_vectors_count", "HOT_FILTER_FIELDS",
          "query"]
