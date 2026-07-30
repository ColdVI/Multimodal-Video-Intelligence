from __future__ import annotations

import math
from typing import Any, Iterable

from app.config import DIMENSIONS, settings
from app.search.strategies import clickhouse_settings


def client():
    import clickhouse_connect

    return clickhouse_connect.get_client(
        host=settings.ch_host,
        port=settings.ch_port,
        username=settings.ch_user,
        password=settings.ch_password,
        database=settings.ch_db,
        connect_timeout=5,
    )


def health() -> bool:
    try:
        return client().command("SELECT 1") == 1
    except Exception:
        return False


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
                   altitude_m Nullable(Float32), velocity_mps Nullable(Float32),
                   gimbal_pitch Nullable(Float32), person_count UInt16,
                   vehicle_count UInt16, is_night UInt8,
                   embedding Array(Float32) CODEC(NONE),
                   INDEX idx_vec embedding TYPE vector_similarity('hnsw','cosineDistance',{dimension}) GRANULARITY 100000000,
                   INDEX idx_alt altitude_m TYPE minmax GRANULARITY 4,
                   INDEX idx_vel velocity_mps TYPE minmax GRANULARITY 4,
                   INDEX idx_per person_count TYPE minmax GRANULARITY 4
                 ) ENGINE=MergeTree ORDER BY (dataset_id,run_id,video_id,t_start)""",
            settings={"allow_experimental_vector_similarity_index": 1},
        )


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


def search_vectors(
    dataset_id: str,
    dimension: int,
    query_vector: Iterable[float],
    top_k: int,
    strategy: str,
    candidate_ids: list[str] | None,
    *,
    run_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values, notes = clickhouse_settings(strategy, top_k)
    values["allow_experimental_vector_similarity_index"] = 1
    params: dict[str, Any] = {"dataset_id": dataset_id, "query_vector": list(query_vector), "top_k": top_k}
    table = f"seg_ch_{dimension}"
    run_clause = ""
    if run_id is not None:
        table = f"seg_ch_{dimension}_runs"
        params["run_id"] = run_id
        run_clause = " AND run_id={run_id:UUID}"
    candidate_clause = ""
    if candidate_ids is not None:
        if not candidate_ids:
            return [], {"plan_used_vector_index": False, "indexed_vectors_count": None, "notes": notes}
        params["candidate_ids"] = candidate_ids
        candidate_clause = " AND segment_id IN {candidate_ids:Array(String)}"
    sql = f"""SELECT segment_id, 1-cosineDistance(embedding, {{query_vector:Array(Float32)}}) AS score
              FROM {table}
              WHERE dataset_id={{dataset_id:String}}{run_clause}{candidate_clause}
              ORDER BY cosineDistance(embedding, {{query_vector:Array(Float32)}})
              LIMIT {{top_k:UInt32}}"""
    target = client()
    result = target.query(sql, parameters=params, settings=values)
    rows = [{"segment_id": row[0], "score": float(row[1])} for row in result.result_rows]
    plan_used = False
    try:
        explain = target.query(f"EXPLAIN indexes = 1 {sql}", parameters=params, settings=values)
        plan_used = "vector_similarity" in "\n".join(str(row[0]) for row in explain.result_rows)
    except Exception as exc:
        notes.append(f"EXPLAIN unavailable: {type(exc).__name__}")
    return rows, {"plan_used_vector_index": plan_used, "indexed_vectors_count": None, "notes": notes}


def table_count(dataset_id: str, dimension: int) -> int:
    result = client().query(
        f"SELECT count() FROM seg_ch_{dimension} WHERE dataset_id={{dataset_id:String}}",
        parameters={"dataset_id": dataset_id},
    )
    return int(result.result_rows[0][0])


def storage_mb(dimension: int) -> float:
    result = client().query(
        "SELECT total_bytes FROM system.tables WHERE database=currentDatabase() AND name={name:String}",
        parameters={"name": f"seg_ch_{dimension}"},
    )
    return float(result.result_rows[0][0] or 0) / 1024**2 if result.result_rows else 0.0


def write_run_chunk(
    run_id: str, dataset_id: str, dimension: int, chunk_index: int, rows: list[dict[str, Any]],
) -> int:
    columns = [
        "run_id", "chunk_index", "segment_id", "dataset_id", "video_id", "t_start", "t_end",
        "altitude_m", "velocity_mps", "gimbal_pitch", "person_count", "vehicle_count", "is_night", "embedding",
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


def count_run(dataset_id: str, run_id: str, dimension: int) -> int:
    result = client().query(
        f"SELECT count() FROM seg_ch_{dimension}_runs WHERE dataset_id={{dataset_id:String}} AND run_id={{run_id:UUID}}",
        parameters={"dataset_id": dataset_id, "run_id": run_id},
    )
    return int(result.result_rows[0][0])


def delete_run(dataset_id: str, run_id: str, dimension: int) -> int:
    _ensure_inactive(dataset_id, run_id)
    before = count_run(dataset_id, run_id, dimension)
    client().command(
        f"ALTER TABLE seg_ch_{dimension}_runs DELETE WHERE dataset_id={{dataset_id:String}} AND run_id={{run_id:UUID}}",
        parameters={"dataset_id": dataset_id, "run_id": run_id}, settings={"mutations_sync": 2},
    )
    return before
