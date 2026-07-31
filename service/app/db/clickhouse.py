from __future__ import annotations

import functools
import math
import threading
from typing import Any, Iterable

from app.config import DIMENSIONS, settings
from app.search.strategies import clickhouse_settings
from app.search.pushdown import normalize_filters


_local = threading.local()


def _new_client():
    import clickhouse_connect

    return clickhouse_connect.get_client(
        host=settings.ch_host,
        port=settings.ch_port,
        username=settings.ch_user,
        password=settings.ch_password,
        database=settings.ch_db,
        connect_timeout=5,
    )


def client():
    """One clickhouse_connect Client per thread, created once and reused across calls on
    that thread. FastAPI's sync `def` handlers (including POST /search) run in Starlette's
    threadpool, so this avoids paying connection-setup cost on every request without
    sharing a single Client instance across threads -- clickhouse_connect's per-instance
    thread-safety under concurrent .query() calls is not a contract this codebase relies
    on. See _reset_client()/_with_reconnect for transient-failure recovery."""
    existing = getattr(_local, "client", None)
    if existing is None:
        existing = _new_client()
        _local.client = existing
    return existing


def _reset_client() -> None:
    _local.client = None


def _transient_error_types() -> tuple[type[Exception], ...]:
    from clickhouse_connect.driver.exceptions import InterfaceError, OperationalError

    return (OperationalError, InterfaceError, ConnectionError, OSError, TimeoutError)


def _with_reconnect(fn):
    """Run fn once; on a transient connection-level error, evict this thread's cached
    client and retry exactly once with a freshly created one. Real programming/data
    errors (ProgrammingError, DataError, IntegrityError, ValueError, ...) are not in
    _transient_error_types() and propagate immediately without a pointless retry."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except _transient_error_types():
            _reset_client()
            return fn(*args, **kwargs)

    return wrapper


@_with_reconnect
def _health_probe() -> bool:
    return client().command("SELECT 1") == 1


def health() -> bool:
    try:
        return _health_probe()
    except Exception:
        _reset_client()
        return False


@_with_reconnect
def init_schema(dimensions: tuple[int, ...] = DIMENSIONS) -> None:
    root = client()
    root.command(f"CREATE DATABASE IF NOT EXISTS {settings.ch_db}")
    root.command(
        """CREATE TABLE IF NOT EXISTS telemetry_raw (
               dataset_id LowCardinality(String), video_id String, frame_index UInt32,
               timestamp DateTime64(6), latitude Nullable(Float64), longitude Nullable(Float64),
               altitude_m Nullable(Float32), velocity_mps Nullable(Float32),
               gimbal_pitch Nullable(Float32), compass_heading Nullable(Float32),
               gimbal_heading Nullable(Float32), extra Map(String, Float64)
             ) ENGINE=MergeTree ORDER BY (dataset_id,video_id,frame_index)"""
    )
    for dimension in dimensions:
        root.command(
            f"""CREATE TABLE IF NOT EXISTS seg_ch_{dimension} (
                   segment_id String, dataset_id LowCardinality(String), video_id String,
                   t_start Float32, t_end Float32,
                   altitude_m Float32, velocity_mps Float32, gimbal_pitch Float32,
                   person_count UInt16, vehicle_count UInt16, is_night UInt8,
                   embedding Array(Float32) CODEC(NONE),
                   INDEX idx_vec embedding TYPE vector_similarity('hnsw','cosineDistance',{dimension}) GRANULARITY 100000000,
                   INDEX idx_alt altitude_m TYPE minmax GRANULARITY 4,
                   INDEX idx_vel velocity_mps TYPE minmax GRANULARITY 4,
                   INDEX idx_per person_count TYPE minmax GRANULARITY 4
                 ) ENGINE=MergeTree ORDER BY (dataset_id,video_id,t_start)""",
            settings={"allow_experimental_vector_similarity_index": 1},
        )
        root.command(
            f"""CREATE TABLE IF NOT EXISTS seg_ch_{dimension}_runs (
                   run_id UUID, chunk_index UInt32, segment_id String,
                   dataset_id LowCardinality(String), video_id String,
                   t_start Float32, t_end Float32,
                   event_category Nullable(String), split Nullable(String),
                   latitude Nullable(Float64), longitude Nullable(Float64),
                   altitude_m Nullable(Float32), velocity_mps Nullable(Float32),
                   roll Nullable(Float32), pitch Nullable(Float32), yaw Nullable(Float32),
                   yaw_rate Nullable(Float32), gimbal_pitch Nullable(Float32),
                   gimbal_heading Nullable(Float32), compass_heading Nullable(Float32),
                   person_count Nullable(UInt16), vehicle_count Nullable(UInt16),
                   bus_count Nullable(UInt16), is_night Nullable(UInt8),
                   embedding Array(Float32) CODEC(NONE),
                   INDEX idx_vec embedding TYPE vector_similarity('hnsw','cosineDistance',{dimension}) GRANULARITY 100000000,
                   INDEX idx_alt altitude_m TYPE minmax GRANULARITY 4,
                   INDEX idx_vel velocity_mps TYPE minmax GRANULARITY 4,
                   INDEX idx_per person_count TYPE minmax GRANULARITY 4
                 ) ENGINE=MergeTree ORDER BY (dataset_id,run_id,video_id,t_start)""",
            settings={"allow_experimental_vector_similarity_index": 1},
        )


@_with_reconnect
def replace_vectors(dataset_id: str, dimension: int, rows: list[dict[str, Any]]) -> None:
    target = client()
    escaped = dataset_id.replace("'", "''")
    target.command(f"ALTER TABLE seg_ch_{dimension} DELETE WHERE dataset_id='{escaped}'", settings={"mutations_sync": 2})
    columns = [
        "segment_id", "dataset_id", "video_id", "t_start", "t_end", "altitude_m",
        "velocity_mps", "gimbal_pitch", "person_count", "vehicle_count", "is_night", "embedding",
    ]
    data = [[row[column] for column in columns] for row in rows]
    if data:
        target.insert(f"seg_ch_{dimension}", data, column_names=columns)


def _empty_diagnostics(notes: list[str]) -> dict[str, Any]:
    return {
        "plan_used_vector_index": None, "indexed_vectors_count": None, "notes": notes,
        "filtered_corpus_count": 0, "candidate_input_count": 0, "candidate_count": 0,
        "candidate_count_status": "computed", "explain_status": "not_requested",
    }


def _corpus_count(target: Any, table: str, run_clause: str, predicate_clause: str, params: dict[str, Any]) -> int:
    sql = f"SELECT count() FROM {table} WHERE dataset_id={{dataset_id:String}}{run_clause}{predicate_clause}"
    return int(target.query(sql, parameters=params).result_rows[0][0])


@_with_reconnect
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
    """Search seg_ch_{dimension}[_runs]. count()/EXPLAIN never run unconditionally on the
    hot path: count() only runs when the result is actually underfilled (needed to tell
    candidate_shortage apart from ann_filter_loss) or when diagnose=True is requested;
    EXPLAIN only runs when explain=True is requested."""
    values, notes = clickhouse_settings(strategy, top_k)
    values["allow_experimental_vector_similarity_index"] = 1
    params: dict[str, Any] = {"dataset_id": dataset_id, "query_vector": list(query_vector), "top_k": top_k}
    table = f"seg_ch_{dimension}"
    run_clause = ""
    if run_id is not None:
        table = f"seg_ch_{dimension}_runs"
        params["run_id"] = run_id
        run_clause = " AND run_id={run_id:UUID}"
    predicate_clauses = []
    predicates = normalize_filters(metadata_filters, telemetry_filters)
    for index, predicate in enumerate(predicates):
        field = predicate.field
        if predicate.kind == "equals":
            type_name = "String" if isinstance(predicate.value, str) else "UInt8"
            key = f"filter_{index}"
            params[key] = predicate.value
            predicate_clauses.append(f"{field}={{{key}:{type_name}}}")
        elif predicate.wrap and predicate.minimum is not None and predicate.maximum is not None:
            low, high = f"filter_{index}_lo", f"filter_{index}_hi"
            params.update({low: predicate.minimum, high: predicate.maximum})
            predicate_clauses.append(f"({field}>={{{low}:Float64}} OR {field}<={{{high}:Float64}})")
        else:
            if predicate.minimum is not None:
                key = f"filter_{index}_lo"
                params[key] = predicate.minimum
                predicate_clauses.append(f"{field}>={{{key}:Float64}}")
            if predicate.maximum is not None:
                key = f"filter_{index}_hi"
                params[key] = predicate.maximum
                predicate_clauses.append(f"{field}<={{{key}:Float64}}")
    predicate_clause = "" if not predicate_clauses else " AND " + " AND ".join(predicate_clauses)
    candidate_clause = ""
    if candidate_ids is not None:
        if not candidate_ids:
            return [], _empty_diagnostics(notes)
        params["candidate_ids"] = candidate_ids
        candidate_clause = " AND segment_id IN {candidate_ids:Array(String)}"
    sql = f"""SELECT segment_id, 1-cosineDistance(embedding, {{query_vector:Array(Float32)}}) AS score
              FROM {table}
              WHERE dataset_id={{dataset_id:String}}{run_clause}{predicate_clause}{candidate_clause}
              ORDER BY cosineDistance(embedding, {{query_vector:Array(Float32)}})
              LIMIT {{top_k:UInt32}}"""
    target = client()
    result = target.query(sql, parameters=params, settings=values)
    rows = [{"segment_id": row[0], "score": float(row[1])} for row in result.result_rows]
    underfilled = len(rows) < top_k
    candidate_input_count = len(candidate_ids) if candidate_ids is not None else None
    filtered_corpus_count: int | None = None
    candidate_count: int | None = None
    candidate_count_status = "not_requested"
    if diagnose or underfilled:
        filtered_corpus_count = _corpus_count(target, table, run_clause, predicate_clause, params)
        if candidate_clause:
            candidate_count = _corpus_count(target, table, run_clause, predicate_clause + candidate_clause, params)
        else:
            candidate_count = filtered_corpus_count
        candidate_count_status = "computed"
    plan_used: bool | None = None
    explain_status = "not_requested"
    if explain:
        explain_status = "computed"
        try:
            explain_result = target.query(f"EXPLAIN indexes = 1 {sql}", parameters=params, settings=values)
            plan_used = "vector_similarity" in "\n".join(str(row[0]) for row in explain_result.result_rows)
        except Exception as exc:
            notes.append(f"EXPLAIN unavailable: {type(exc).__name__}")
            explain_status = "unavailable"
    return rows, {
        "plan_used_vector_index": plan_used, "indexed_vectors_count": None,
        "candidate_count": candidate_count, "filtered_corpus_count": filtered_corpus_count,
        "candidate_input_count": candidate_input_count, "candidate_count_status": candidate_count_status,
        "explain_status": explain_status, "notes": notes,
    }


@_with_reconnect
def table_count(dataset_id: str, dimension: int) -> int:
    result = client().query(
        f"SELECT count() FROM seg_ch_{dimension} WHERE dataset_id={{dataset_id:String}}",
        parameters={"dataset_id": dataset_id},
    )
    return int(result.result_rows[0][0])


@_with_reconnect
def storage_mb(dimension: int) -> float:
    result = client().query(
        "SELECT total_bytes FROM system.tables WHERE database=currentDatabase() AND name={name:String}",
        parameters={"name": f"seg_ch_{dimension}"},
    )
    return float(result.result_rows[0][0] or 0) / 1024**2 if result.result_rows else 0.0


@_with_reconnect
def write_run_chunk(
    run_id: str, dataset_id: str, dimension: int, chunk_index: int, rows: list[dict[str, Any]],
) -> int:
    columns = [
        "run_id", "chunk_index", "segment_id", "dataset_id", "video_id", "t_start", "t_end",
        "event_category", "split", "latitude", "longitude", "altitude_m", "velocity_mps",
        "roll", "pitch", "yaw", "yaw_rate", "gimbal_pitch", "gimbal_heading", "compass_heading",
        "person_count", "vehicle_count", "bus_count", "is_night", "embedding",
    ]
    data = [[run_id, chunk_index, *[row[column] for column in columns[2:]]] for row in rows]
    if data:
        client().insert(f"seg_ch_{dimension}_runs", data, column_names=columns)
    return len(data)


def _ensure_inactive(dataset_id: str, run_id: str) -> None:
    from app.db import postgres

    snapshot = postgres.get_active_run_snapshot(dataset_id)
    if snapshot and snapshot["run_id"] == run_id:
        raise ValueError("destructive cleanup of an active run is forbidden")


@_with_reconnect
def delete_inactive_chunk(run_id: str, dataset_id: str, chunk_index: int, dimension: int) -> int:
    _ensure_inactive(dataset_id, run_id)
    before = count_run(dataset_id, run_id, dimension)
    target = client()
    target.command(
        f"ALTER TABLE seg_ch_{dimension}_runs DELETE WHERE dataset_id={{dataset_id:String}} "
        "AND run_id={run_id:UUID} AND chunk_index={chunk_index:UInt32}",
        parameters={"dataset_id": dataset_id, "run_id": run_id, "chunk_index": chunk_index},
        settings={"mutations_sync": 2},
    )
    return max(0, before - count_run(dataset_id, run_id, dimension))


@_with_reconnect
def count_run(dataset_id: str, run_id: str, dimension: int) -> int:
    result = client().query(
        f"SELECT count() FROM seg_ch_{dimension}_runs WHERE dataset_id={{dataset_id:String}} AND run_id={{run_id:UUID}}",
        parameters={"dataset_id": dataset_id, "run_id": run_id},
    )
    return int(result.result_rows[0][0])


@_with_reconnect
def delete_run(dataset_id: str, run_id: str, dimension: int) -> int:
    _ensure_inactive(dataset_id, run_id)
    before = count_run(dataset_id, run_id, dimension)
    client().command(
        f"ALTER TABLE seg_ch_{dimension}_runs DELETE WHERE dataset_id={{dataset_id:String}} AND run_id={{run_id:UUID}}",
        parameters={"dataset_id": dataset_id, "run_id": run_id}, settings={"mutations_sync": 2},
    )
    return before
