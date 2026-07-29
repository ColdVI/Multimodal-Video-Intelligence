STRATEGIES = {
    "clickhouse": ("exact", "ann", "prefilter", "postfilter"),
    "qdrant": ("exact", "ann", "ann_high_ef", "prefilter"),
    "pgvector": ("exact", "ann", "iterative_scan", "iterative_strict"),
    "numpy_exact": ("exact",),
    "milvus": ("ann",),
}


def validate(backend: str, strategy: str) -> None:
    if backend not in STRATEGIES:
        raise ValueError(f"unsupported backend: {backend}")
    if strategy not in STRATEGIES[backend]:
        raise ValueError(f"unsupported strategy {strategy!r} for {backend}")


def clickhouse_limit_settings(top_k: int) -> tuple[int, str]:
    if top_k > 100:
        return top_k, f"max_limit_for_vector_search_queries={top_k}"
    return 100, ""
