from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable

import psycopg
from psycopg.rows import dict_row

from app.config import settings

DIMS = (2048, 1024, 512, 256)
TABLES = {d: f"emb_pg_{d}" for d in DIMS}

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS datasets (
  dataset_id text PRIMARY KEY, dataset_version text, source_hash text,
  license text, has_telemetry boolean NOT NULL, has_captions boolean NOT NULL
);
CREATE TABLE IF NOT EXISTS videos (
  dataset_id text, video_id text, source_uri text, split text,
  duration_s double precision, event_category text,
  PRIMARY KEY (dataset_id, video_id)
);
CREATE TABLE IF NOT EXISTS segments (
  segment_id text PRIMARY KEY, dataset_id text NOT NULL, video_id text NOT NULL,
  t_start double precision NOT NULL, t_end double precision NOT NULL,
  caption text, file_path text
);
CREATE TABLE IF NOT EXISTS segment_metadata (
  segment_id text PRIMARY KEY, person_count int, vehicle_count int,
  bus_count int, object_classes text[], brightness real, camera_motion real
);
CREATE TABLE IF NOT EXISTS segment_telemetry (
  segment_id text PRIMARY KEY, timestamp_start timestamptz, timestamp_end timestamptz,
  latitude double precision, longitude double precision,
  altitude_m real, velocity_mps real, roll real, pitch real, yaw real,
  yaw_rate real, gimbal_pitch real, gimbal_heading real,
  compass_heading real, imu_summary jsonb
);
CREATE TABLE IF NOT EXISTS emb_pg_2048 (segment_id text PRIMARY KEY, dataset_id text, v halfvec(2048));
CREATE TABLE IF NOT EXISTS emb_pg_1024 (segment_id text PRIMARY KEY, dataset_id text, v vector(1024));
CREATE TABLE IF NOT EXISTS emb_pg_1024h(segment_id text PRIMARY KEY, dataset_id text, v halfvec(1024));
CREATE TABLE IF NOT EXISTS emb_pg_512  (segment_id text PRIMARY KEY, dataset_id text, v vector(512));
CREATE TABLE IF NOT EXISTS emb_pg_256  (segment_id text PRIMARY KEY, dataset_id text, v vector(256));
CREATE INDEX IF NOT EXISTS ix_seg_ds ON segments(dataset_id);
CREATE INDEX IF NOT EXISTS ix_vid_split ON videos(dataset_id, split);
CREATE INDEX IF NOT EXISTS ix_tel_alt ON segment_telemetry(altitude_m);
CREATE INDEX IF NOT EXISTS ix_tel_vel ON segment_telemetry(velocity_mps);
CREATE INDEX IF NOT EXISTS ix_tel_gp ON segment_telemetry(gimbal_pitch);
CREATE INDEX IF NOT EXISTS ix_meta_p ON segment_metadata(person_count);
"""


@contextmanager
def connection():
    with psycopg.connect(settings.pg_dsn, row_factory=dict_row) as conn:
        yield conn


def init_schema() -> None:
    with connection() as conn:
        conn.execute(SCHEMA_SQL)


def healthy() -> bool:
    try:
        with connection() as conn:
            return conn.execute("SELECT 1").fetchone()["?column?"] == 1
    except Exception:
        return False


def vector_literal(vector: Iterable[float]) -> str:
    return "[" + ",".join(f"{float(x):.9g}" for x in vector) + "]"


def upsert_dataset(dataset: dict[str, Any]) -> None:
    with connection() as conn:
        conn.execute(
            """INSERT INTO datasets VALUES (%(dataset_id)s,%(dataset_version)s,%(source_hash)s,%(license)s,%(has_telemetry)s,%(has_captions)s)
            ON CONFLICT (dataset_id) DO UPDATE SET dataset_version=EXCLUDED.dataset_version, source_hash=EXCLUDED.source_hash,
            license=EXCLUDED.license, has_telemetry=EXCLUDED.has_telemetry, has_captions=EXCLUDED.has_captions""", dataset)


def upsert_segments(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with connection() as conn:
        with conn.cursor() as cur:
            for row in rows:
                cur.execute("""INSERT INTO videos(dataset_id,video_id,source_uri,split,duration_s,event_category)
                    VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (dataset_id,video_id) DO UPDATE SET
                    source_uri=EXCLUDED.source_uri,split=EXCLUDED.split,duration_s=EXCLUDED.duration_s,event_category=EXCLUDED.event_category""",
                    (row["dataset_id"], row["video_id"], row.get("file_path"), row.get("split"), row["t_end"], row.get("event_category")))
                cur.execute("""INSERT INTO segments(segment_id,dataset_id,video_id,t_start,t_end,caption,file_path)
                    VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (segment_id) DO UPDATE SET
                    caption=EXCLUDED.caption,file_path=EXCLUDED.file_path""",
                    (row["segment_id"], row["dataset_id"], row["video_id"], row["t_start"], row["t_end"], row.get("caption"), row.get("file_path")))
                cur.execute("""INSERT INTO segment_metadata(segment_id,person_count,vehicle_count,bus_count,object_classes,brightness,camera_motion)
                    VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (segment_id) DO UPDATE SET
                    person_count=EXCLUDED.person_count,vehicle_count=EXCLUDED.vehicle_count,bus_count=EXCLUDED.bus_count""",
                    (row["segment_id"], row.get("person_count"), row.get("vehicle_count"), row.get("bus_count"), row.get("object_classes"), row.get("brightness"), row.get("camera_motion")))
                if row.get("has_telemetry"):
                    cur.execute("""INSERT INTO segment_telemetry(segment_id,altitude_m,velocity_mps,roll,pitch,yaw,yaw_rate,gimbal_pitch,gimbal_heading,compass_heading)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (segment_id) DO UPDATE SET
                        altitude_m=EXCLUDED.altitude_m,velocity_mps=EXCLUDED.velocity_mps,roll=EXCLUDED.roll,pitch=EXCLUDED.pitch,
                        yaw=EXCLUDED.yaw,yaw_rate=EXCLUDED.yaw_rate,gimbal_pitch=EXCLUDED.gimbal_pitch""",
                        (row["segment_id"], row.get("altitude_m"), row.get("velocity_mps"), row.get("roll"), row.get("pitch"), row.get("yaw"), row.get("yaw_rate"), row.get("gimbal_pitch"), row.get("gimbal_heading"), row.get("compass_heading")))


def upsert_embeddings(dataset_id: str, vectors: dict[int, list[tuple[str, Any]]]) -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            for dim, entries in vectors.items():
                table = TABLES[dim]
                cast = "halfvec" if dim == 2048 else "vector"
                for segment_id, vector in entries:
                    literal = vector_literal(vector)
                    cur.execute(f"INSERT INTO {table}(segment_id,dataset_id,v) VALUES (%s,%s,%s::{cast}) ON CONFLICT (segment_id) DO UPDATE SET v=EXCLUDED.v", (segment_id, dataset_id, literal))
                    if dim == 1024:
                        cur.execute("INSERT INTO emb_pg_1024h(segment_id,dataset_id,v) VALUES (%s,%s,%s::halfvec) ON CONFLICT (segment_id) DO UPDATE SET v=EXCLUDED.v", (segment_id, dataset_id, literal))


def ensure_hnsw_indexes() -> None:
    statements = [
        "CREATE INDEX IF NOT EXISTS ix_pg_2048_hnsw ON emb_pg_2048 USING hnsw (v halfvec_cosine_ops) WITH (m=16, ef_construction=128)",
        "CREATE INDEX IF NOT EXISTS ix_pg_1024_hnsw ON emb_pg_1024 USING hnsw (v vector_cosine_ops) WITH (m=16, ef_construction=128)",
        "CREATE INDEX IF NOT EXISTS ix_pg_1024h_hnsw ON emb_pg_1024h USING hnsw (v halfvec_cosine_ops) WITH (m=16, ef_construction=128)",
        "CREATE INDEX IF NOT EXISTS ix_pg_512_hnsw ON emb_pg_512 USING hnsw (v vector_cosine_ops) WITH (m=16, ef_construction=128)",
        "CREATE INDEX IF NOT EXISTS ix_pg_256_hnsw ON emb_pg_256 USING hnsw (v vector_cosine_ops) WITH (m=16, ef_construction=128)",
    ]
    with connection() as conn:
        for sql in statements:
            conn.execute(sql)


def filter_ids(dataset_id: str, metadata: dict | None, telemetry: dict | None) -> list[str] | None:
    metadata, telemetry = metadata or {}, telemetry or {}
    clauses, params = ["s.dataset_id=%s"], [dataset_id]
    mapping = {"event_category": "v.event_category", "split": "v.split", "video_id": "s.video_id"}
    for key, column in mapping.items():
        if metadata.get(key) not in (None, ""):
            clauses.append(f"{column}=%s"); params.append(metadata[key])
    telmap = {"altitude_m": "t.altitude_m", "velocity_mps": "t.velocity_mps", "gimbal_pitch": "t.gimbal_pitch"}
    for key, column in telmap.items():
        value = telemetry.get(key)
        if value:
            if value[0] is not None: clauses.append(f"{column}>=%s"); params.append(value[0])
            if value[1] is not None: clauses.append(f"{column}<=%s"); params.append(value[1])
    if len(clauses) == 1:
        return None
    sql = f"""SELECT s.segment_id FROM segments s JOIN videos v USING(dataset_id,video_id)
        LEFT JOIN segment_telemetry t USING(segment_id) WHERE {' AND '.join(clauses)} ORDER BY s.segment_id"""
    with connection() as conn:
        return [row["segment_id"] for row in conn.execute(sql, params).fetchall()]


def vector_search(dataset_id: str, vector: Any, dim: int, top_k: int, strategy: str, candidate_ids: list[str] | None) -> list[tuple[str, float]]:
    table, cast = TABLES[dim], "halfvec" if dim == 2048 else "vector"
    where, params = ["dataset_id=%s"], [dataset_id]
    if candidate_ids is not None:
        if not candidate_ids: return []
        where.append("segment_id = ANY(%s)"); params.append(candidate_ids)
    literal = vector_literal(vector)
    settings_sql = ""
    if strategy == "exact": settings_sql = "SET LOCAL enable_indexscan=off; SET LOCAL enable_bitmapscan=off;"
    elif strategy == "ann": settings_sql = "SET LOCAL hnsw.ef_search=40; SET LOCAL hnsw.iterative_scan='off';"
    elif strategy == "iterative_scan": settings_sql = "SET LOCAL hnsw.iterative_scan='relaxed_order';"
    elif strategy == "iterative_strict": settings_sql = "SET LOCAL hnsw.ef_search=200; SET LOCAL hnsw.iterative_scan='strict_order';"
    with connection() as conn:
        if settings_sql: conn.execute(settings_sql)
        rows = conn.execute(f"SELECT segment_id, 1-(v <=> %s::{cast}) AS score FROM {table} WHERE {' AND '.join(where)} ORDER BY v <=> %s::{cast} LIMIT %s", [literal, *params, literal, top_k]).fetchall()
    return [(r["segment_id"], float(r["score"])) for r in rows]


def hydrate(ids: list[str]) -> list[dict]:
    if not ids: return []
    with connection() as conn:
        rows = conn.execute("""SELECT s.segment_id,s.video_id,s.t_start,s.t_end,s.caption,s.file_path,
            t.altitude_m,t.velocity_mps,t.gimbal_pitch FROM segments s LEFT JOIN segment_telemetry t USING(segment_id)
            WHERE s.segment_id=ANY(%s)""", (ids,)).fetchall()
    lookup = {r["segment_id"]: dict(r) for r in rows}
    return [lookup[x] for x in ids if x in lookup]


def stats() -> list[dict]:
    with connection() as conn:
        rows = conn.execute("""SELECT d.dataset_id,d.has_telemetry,d.has_captions,count(s.segment_id)::int AS segments
            FROM datasets d LEFT JOIN segments s USING(dataset_id) GROUP BY 1,2,3 ORDER BY 1""").fetchall()
    return [dict(r) for r in rows]


def facets(dataset_id: str) -> dict:
    with connection() as conn:
        meta = conn.execute("""SELECT array_remove(array_agg(DISTINCT v.event_category),NULL) event_categories,
            array_remove(array_agg(DISTINCT v.split),NULL) splits,
            array_remove(array_agg(DISTINCT s.video_id),NULL) video_ids FROM segments s JOIN videos v USING(dataset_id,video_id) WHERE s.dataset_id=%s""", (dataset_id,)).fetchone()
        telemetry = conn.execute("""SELECT min(t.altitude_m) altitude_min,max(t.altitude_m) altitude_max,
            min(t.velocity_mps) velocity_min,max(t.velocity_mps) velocity_max,
            min(t.gimbal_pitch) gimbal_pitch_min,max(t.gimbal_pitch) gimbal_pitch_max
            FROM segments s JOIN segment_telemetry t USING(segment_id) WHERE s.dataset_id=%s""", (dataset_id,)).fetchone()
    return {**dict(meta), "telemetry": dict(telemetry)}


def count(dataset_id: str) -> int:
    with connection() as conn:
        return int(conn.execute("SELECT count(*) n FROM segments WHERE dataset_id=%s", (dataset_id,)).fetchone()["n"])


def all_vectors(dataset_id: str, dim: int, candidate_ids: list[str] | None = None) -> list[tuple[str, list[float]]]:
    table = TABLES[dim]
    clauses, params = ["dataset_id=%s"], [dataset_id]
    if candidate_ids is not None:
        if not candidate_ids: return []
        clauses.append("segment_id=ANY(%s)"); params.append(candidate_ids)
    with connection() as conn:
        rows = conn.execute(f"SELECT segment_id,v::text AS vector FROM {table} WHERE {' AND '.join(clauses)} ORDER BY segment_id", params).fetchall()
    return [(r["segment_id"],[float(x) for x in r["vector"].strip("[]").split(",")]) for r in rows]
