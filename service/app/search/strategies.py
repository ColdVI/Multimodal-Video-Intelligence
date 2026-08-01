from __future__ import annotations


SUPPORTED_STRATEGIES = {
    "clickhouse": ("exact", "ann", "prefilter", "postfilter"),
    "qdrant": ("exact", "ann", "ann_high_ef", "prefilter"),
    "pgvector": ("exact", "ann", "iterative_scan", "iterative_strict"),
    "milvus": ("ann",),
    "numpy_exact": ("exact",),
}

DEFAULT_STRATEGIES = {
    "clickhouse": "prefilter",
    "qdrant": "ann",
    "pgvector": "iterative_scan",
}


def default_strategy(backend: str) -> str:
    try:
        return DEFAULT_STRATEGIES[backend]
    except KeyError as exc:
        raise ValueError(f"no default strategy for backend: {backend}") from exc


def validate_strategy(backend: str, strategy: str) -> None:
    if backend not in SUPPORTED_STRATEGIES:
        raise ValueError(f"unknown backend: {backend}")
    if strategy not in SUPPORTED_STRATEGIES[backend]:
        raise ValueError(f"strategy {strategy!r} is not supported by {backend}")


def clickhouse_settings(strategy: str, top_k: int) -> tuple[dict[str, object], list[str]]:
    validate_strategy("clickhouse", strategy)
    values: dict[str, object] = {}
    notes: list[str] = []
    if strategy == "exact":
        values["query_plan_try_use_vector_search"] = 0
    elif strategy == "prefilter":
        values["vector_search_filter_strategy"] = "prefilter"
    elif strategy == "postfilter":
        values.update(
            vector_search_filter_strategy="auto",
            vector_search_index_fetch_multiplier=3.0,
            vector_search_with_rescoring=1,
        )
    else:
        values["vector_search_filter_strategy"] = "auto"
    if top_k > 100:
        values["max_limit_for_vector_search_queries"] = top_k
        notes.append(f"top_k={top_k}; max_limit_for_vector_search_queries query-scoped yükseltildi")
    else:
        notes.append("top_k<=100, max_limit ayarı değişmedi")
    return values, notes


def qdrant_search_params(strategy: str) -> dict[str, object]:
    validate_strategy("qdrant", strategy)
    if strategy == "exact":
        return {"exact": True}
    return {"hnsw_ef": 512 if strategy == "ann_high_ef" else 128}


def pgvector_session_settings(strategy: str) -> tuple[str, ...]:
    validate_strategy("pgvector", strategy)
    if strategy == "exact":
        return ("SET LOCAL enable_indexscan=off", "SET LOCAL enable_bitmapscan=off")
    if strategy == "ann":
        return ("SET LOCAL hnsw.ef_search=40", "SET LOCAL hnsw.iterative_scan='off'")
    if strategy == "iterative_scan":
        return ("SET LOCAL hnsw.ef_search=100", "SET LOCAL hnsw.iterative_scan='relaxed_order'")
    return ("SET LOCAL hnsw.ef_search=200", "SET LOCAL hnsw.iterative_scan='strict_order'")
