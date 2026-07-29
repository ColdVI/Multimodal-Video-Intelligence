"""ClickHouse ince sarmalayici (spec SS4.5). Install/start/health/stop/
cleanup scripts/install_clickhouse_colab.sh'ya delege eder - burada
ABSTRACTION KATMANI yazilmiyor (spec SS1.2: 'backend abstraction framework
YAPILMAYACAK'), sadece DDL + ingest + query icin ince yardimcilar var."""
import pathlib

from ._runner import run_action

SCRIPT = "install_clickhouse_colab.sh"


def install() -> dict:
    return run_action(SCRIPT, "install", timeout=900)


def start() -> dict:
    return run_action(SCRIPT, "start", timeout=60)


def health_check(host: str = "127.0.0.1", port: int = 8123, timeout: int = 35) -> dict:
    result = run_action(SCRIPT, "health", timeout=timeout)
    return {"healthy": result["ok"], **result}


def stop() -> dict:
    return run_action(SCRIPT, "stop", timeout=30)


def cleanup() -> dict:
    return run_action(SCRIPT, "cleanup", timeout=60)


def table_ddl(dimension: int, table_name: str = None) -> str:
    """spec SS4.5 - CODEC(NONE) resmi oneri (dense vektorler sikismiyor)."""
    table_name = table_name or f"seg_{dimension}"
    return f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    segment_id    String,
    dataset_id    LowCardinality(String),
    video_id      String,
    t_start       Float32,
    t_end         Float32,
    altitude_m    Float32,
    velocity_mps  Float32,
    person_count  UInt16,
    vehicle_count UInt16,
    is_night      UInt8,
    embedding     Array(Float32) CODEC(NONE),
    INDEX idx_vec embedding TYPE vector_similarity('hnsw', 'cosineDistance', {dimension}) GRANULARITY 100000000,
    INDEX idx_alt altitude_m TYPE minmax GRANULARITY 4,
    INDEX idx_vel velocity_mps TYPE minmax GRANULARITY 4,
    INDEX idx_per person_count TYPE minmax GRANULARITY 4
) ENGINE = MergeTree ORDER BY (dataset_id, video_id, t_start)
""".strip()


def get_client(host: str = "127.0.0.1", port: int = 8123):
    import clickhouse_connect
    return clickhouse_connect.get_client(host=host, port=port)


def _fmt_vector(vec) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


_STRATEGY_SETTINGS = {
    "auto": "",
    "exact": "SETTINGS query_plan_try_use_vector_search = 0",
    "prefilter": "SETTINGS vector_search_filter_strategy = 'prefilter'",
}


def build_query_sql(table: str, qvec, top_k: int, where: str = "1", strategy: str = "auto") -> str:
    """search/query.py::_build_sql ile AYNI desen (Faz 2 strateji matrisi) -
    burada AU-AIR seg_<dim> semasina genellenmis."""
    if strategy not in _STRATEGY_SETTINGS:
        raise ValueError(f"bilinmeyen strategy: {strategy!r}. Gecerli: {sorted(_STRATEGY_SETTINGS)}")
    return f"""
        SELECT segment_id, video_id,
               cosineDistance(embedding, {_fmt_vector(qvec)}) AS dist
        FROM {table}
        WHERE {where}
        ORDER BY dist ASC
        LIMIT {top_k}
        {_STRATEGY_SETTINGS[strategy]}
    """.strip()


__all__ = ["install", "start", "health_check", "stop", "cleanup", "table_ddl", "get_client",
          "build_query_sql"]
