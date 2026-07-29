from __future__ import annotations

from typing import Any

import clickhouse_connect

from app.config import settings
from app.search.strategies import clickhouse_limit_settings

DIMS = (2048, 1024, 512, 256)


def client():
    return clickhouse_connect.get_client(host=settings.ch_host, port=settings.ch_port, username=settings.ch_user, password=settings.ch_password, database=settings.ch_database)


def init_schema() -> None:
    root = clickhouse_connect.get_client(host=settings.ch_host, port=settings.ch_port, username=settings.ch_user, password=settings.ch_password)
    root.command(f"CREATE DATABASE IF NOT EXISTS {settings.ch_database}")
    db = client()
    db.command("""CREATE TABLE IF NOT EXISTS telemetry_raw (
        dataset_id LowCardinality(String), video_id String, segment_id String, frame_index UInt32,
        ts Nullable(DateTime64(6)), latitude Nullable(Float64), longitude Nullable(Float64), altitude_m Nullable(Float32),
        velocity_mps Nullable(Float32), gimbal_pitch Nullable(Float32), compass_heading Nullable(Float32), extra Map(String, Float64)
      ) ENGINE=MergeTree ORDER BY (dataset_id,video_id,frame_index)""")
    for dim in DIMS:
        db.command(f"""CREATE TABLE IF NOT EXISTS seg_ch_{dim} (
          segment_id String, dataset_id LowCardinality(String), video_id String,
          t_start Float32, t_end Float32, altitude_m Nullable(Float32), velocity_mps Nullable(Float32),
          gimbal_pitch Nullable(Float32), person_count Nullable(UInt16), vehicle_count Nullable(UInt16), is_night UInt8,
          embedding Array(Float32) CODEC(NONE),
          INDEX idx_vec embedding TYPE vector_similarity('hnsw','cosineDistance',{dim}) GRANULARITY 100000000,
          INDEX idx_alt altitude_m TYPE minmax GRANULARITY 4,
          INDEX idx_vel velocity_mps TYPE minmax GRANULARITY 4,
          INDEX idx_per person_count TYPE minmax GRANULARITY 4
        ) ENGINE=ReplacingMergeTree ORDER BY (dataset_id,video_id,t_start,segment_id)""")


def healthy() -> bool:
    try:
        return client().query("SELECT 1").result_rows[0][0] == 1
    except Exception:
        return False


def insert_segments(rows: list[dict[str, Any]], vectors: dict[int, list[tuple[str, Any]]]) -> None:
    if not rows: return
    by_id = {r["segment_id"]: r for r in rows}
    db = client()
    columns = ["segment_id","dataset_id","video_id","t_start","t_end","altitude_m","velocity_mps","gimbal_pitch","person_count","vehicle_count","is_night","embedding"]
    for dim, entries in vectors.items():
        db.command(f"ALTER TABLE seg_ch_{dim} DELETE WHERE dataset_id={{dataset_id:String}} SETTINGS mutations_sync=1", parameters={"dataset_id": rows[0]["dataset_id"]})
        data = []
        for segment_id, vector in entries:
            row = by_id[segment_id]
            data.append([segment_id,row["dataset_id"],row["video_id"],row["t_start"],row["t_end"],row.get("altitude_m"),row.get("velocity_mps"),row.get("gimbal_pitch"),row.get("person_count"),row.get("vehicle_count"),int(bool(row.get("is_night"))),vector.tolist()])
        db.insert(f"seg_ch_{dim}", data, column_names=columns)


def vector_search(dataset_id: str, vector: Any, dim: int, top_k: int, strategy: str,
                  candidate_ids: list[str] | None, telemetry: dict | None) -> tuple[list[tuple[str,float]], dict]:
    clauses = ["dataset_id={dataset_id:String}"]
    params: dict[str, Any] = {"dataset_id": dataset_id, "query_vector": vector.tolist(), "limit": top_k}
    if candidate_ids is not None:
        if not candidate_ids: return [], {"plan_used_vector_index": False}
        clauses.append("segment_id IN {candidate_ids:Array(String)}"); params["candidate_ids"] = candidate_ids
    mapping = {"altitude_m":"altitude_m", "velocity_mps":"velocity_mps", "gimbal_pitch":"gimbal_pitch"}
    for idx, (key, col) in enumerate(mapping.items()):
        value = (telemetry or {}).get(key)
        if value:
            if value[0] is not None: clauses.append(f"{col}>={{lo{idx}:Float64}}"); params[f"lo{idx}"] = value[0]
            if value[1] is not None: clauses.append(f"{col}<={{hi{idx}:Float64}}"); params[f"hi{idx}"] = value[1]
    settings_sql = {
        "exact": "query_plan_try_use_vector_search=0",
        "ann": "vector_search_filter_strategy='auto'",
        "prefilter": "vector_search_filter_strategy='prefilter'",
        "postfilter": "vector_search_filter_strategy='auto', vector_search_index_fetch_multiplier=3.0, vector_search_with_rescoring=1",
    }.get(strategy)
    if settings_sql is None: raise ValueError(f"unsupported ClickHouse strategy: {strategy}")
    max_limit, limit_setting = clickhouse_limit_settings(top_k)
    if limit_setting:
        settings_sql += f", {limit_setting}"
    query = f"SELECT segment_id, 1-cosineDistance(embedding, {{query_vector:Array(Float32)}}) score FROM seg_ch_{dim} WHERE {' AND '.join(clauses)} ORDER BY score DESC LIMIT {{limit:UInt32}} SETTINGS {settings_sql}"
    db = client()
    result = db.query(query, parameters=params).result_rows
    plan_used = strategy != "exact"
    try:
        explain = db.query("EXPLAIN indexes=1 " + query, parameters=params).result_rows
        plan_used = "vector_similarity" in "\n".join(str(x) for x in explain)
    except Exception:
        pass
    return [(x, float(score)) for x, score in result], {"plan_used_vector_index": plan_used, "max_limit_setting": max_limit}


def count(dataset_id: str, dim: int = 512) -> int:
    return int(client().query(f"SELECT uniqExact(segment_id) FROM seg_ch_{dim} WHERE dataset_id={{dataset_id:String}}", parameters={"dataset_id":dataset_id}).result_rows[0][0])
