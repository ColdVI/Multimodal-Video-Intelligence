"""pgvector ince sarmalayici (spec SS4.5). Install/start/health/stop/cleanup
scripts/install_pgvector_colab.sh'ya delege eder. Boyuta gore tip secimi
ZORUNLU (spec SS0 bulgu #3): vector HNSW limiti 2000 boyut - 2048d
halfvec(fp16) olmak ZORUNDA."""
from ._runner import run_action

SCRIPT = "install_pgvector_colab.sh"
VECTOR_HNSW_DIM_LIMIT = 2000


def install() -> dict:
    return run_action(SCRIPT, "install", timeout=900)


def start() -> dict:
    return run_action(SCRIPT, "start", timeout=60)


def health_check(timeout: int = 35) -> dict:
    result = run_action(SCRIPT, "health", timeout=timeout)
    return {"healthy": result["ok"], **result}


def stop() -> dict:
    return run_action(SCRIPT, "stop", timeout=30)


def cleanup() -> dict:
    return run_action(SCRIPT, "cleanup", timeout=60)


def get_connection(host: str = "127.0.0.1", port: int = 5432,
                   dbname: str = "phase6_vector_bench", user: str = "postgres"):
    import psycopg
    return psycopg.connect(host=host, port=port, dbname=dbname, user=user, autocommit=True)


def vector_type_for_dimension(dimension: int) -> str:
    """>2000 boyutta vector() HNSW kuramaz (spec SS0 bulgu #3) -> halfvec
    ZORUNLU. <=2000'de vector() varsayilan, ama 1024d icin AYRICA halfvec de
    kurulur (precision confound izolasyonu, spec SS7.3)."""
    if dimension > VECTOR_HNSW_DIM_LIMIT:
        return "halfvec"
    return "vector"


def table_ddl(dimension: int, storage_type: str, table_name: str = None) -> str:
    if storage_type not in ("vector", "halfvec"):
        raise ValueError(f"storage_type 'vector' veya 'halfvec' olmali, geldi: {storage_type}")
    if storage_type == "vector" and dimension > VECTOR_HNSW_DIM_LIMIT:
        raise ValueError(f"vector() {VECTOR_HNSW_DIM_LIMIT} boyutu asamaz (istenen {dimension}) - halfvec kullanin")
    suffix = "v" if storage_type == "vector" else "h"
    table_name = table_name or f"seg_{dimension}_{suffix}"
    ops_class = "vector_cosine_ops" if storage_type == "vector" else "halfvec_cosine_ops"
    return (
        f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
        f"    segment_id text PRIMARY KEY,\n"
        f"    dataset_id text NOT NULL,\n"
        f"    video_id text NOT NULL,\n"
        f"    altitude_m real,\n"
        f"    velocity_mps real,\n"
        f"    person_count integer,\n"
        f"    vehicle_count integer,\n"
        f"    v {storage_type}({dimension})\n"
        f");\n"
        f"CREATE INDEX IF NOT EXISTS {table_name}_hnsw ON {table_name} "
        f"USING hnsw (v {ops_class}) WITH (m = 16, ef_construction = 128);"
    )


def _fmt_vector(vec) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def build_query_sql(table: str, qvec, top_k: int, where: str = "TRUE") -> str:
    """<=> = pgvector cosine distance operatoru (vector_cosine_ops/
    halfvec_cosine_ops index'i bunu kullanir)."""
    return (
        f"SELECT segment_id, video_id FROM {table} "
        f"WHERE {where} "
        f"ORDER BY v <=> '{_fmt_vector(qvec)}' LIMIT {top_k}"
    )


__all__ = ["install", "start", "health_check", "stop", "cleanup", "get_connection",
          "vector_type_for_dimension", "table_ddl", "VECTOR_HNSW_DIM_LIMIT", "build_query_sql"]
