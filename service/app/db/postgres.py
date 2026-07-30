from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable

from app.config import DIMENSIONS, settings
from app.search.strategies import pgvector_session_settings
from app.search.filter_projection import POSTGRES_RUN_COLUMNS
from app.search.pushdown import normalize_filters, sql_predicates


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS datasets (
  dataset_id text PRIMARY KEY, dataset_version text, source_hash text,
  license text, has_telemetry boolean NOT NULL, has_captions boolean NOT NULL,
  vector_provenance text NOT NULL DEFAULT 'synthetic'
);
ALTER TABLE datasets ADD COLUMN IF NOT EXISTS vector_provenance text NOT NULL DEFAULT 'synthetic';
CREATE TABLE IF NOT EXISTS videos (
  dataset_id text, video_id text, source_uri text, split text,
  duration_s double precision, event_category text,
  PRIMARY KEY (dataset_id, video_id)
);
CREATE TABLE IF NOT EXISTS segments (
  segment_id text PRIMARY KEY, dataset_id text NOT NULL, video_id text NOT NULL,
  t_start double precision NOT NULL, t_end double precision NOT NULL, caption text
);
CREATE TABLE IF NOT EXISTS segment_metadata (
  segment_id text PRIMARY KEY, person_count int, vehicle_count int,
  bus_count int, object_classes text[], brightness real, camera_motion real
);
CREATE TABLE IF NOT EXISTS segment_telemetry (
  segment_id text PRIMARY KEY, timestamp_start timestamptz, timestamp_end timestamptz,
  latitude double precision, longitude double precision,
  altitude_m real, velocity_mps real,
  roll real, pitch real, yaw real, yaw_rate real,
  gimbal_pitch real, gimbal_heading real, compass_heading real,
  imu_summary jsonb
);
CREATE TABLE IF NOT EXISTS retrieval_groundtruth (
  dataset_id text NOT NULL, query_id text NOT NULL, query_text text NOT NULL,
  relevant_segment_id text NOT NULL, relevant_video_id text NOT NULL,
  relevance_rank int NOT NULL, caption_index int, caption_source text NOT NULL,
  PRIMARY KEY (dataset_id,query_id,relevant_segment_id)
);
CREATE INDEX IF NOT EXISTS ix_seg_ds ON segments(dataset_id);
CREATE INDEX IF NOT EXISTS ix_seg_video ON segments(dataset_id, video_id);
CREATE INDEX IF NOT EXISTS ix_tel_alt ON segment_telemetry(altitude_m);
CREATE INDEX IF NOT EXISTS ix_tel_vel ON segment_telemetry(velocity_mps);
CREATE INDEX IF NOT EXISTS ix_tel_gp  ON segment_telemetry(gimbal_pitch);
CREATE INDEX IF NOT EXISTS ix_meta_p  ON segment_metadata(person_count);
CREATE INDEX IF NOT EXISTS ix_gt_dataset ON retrieval_groundtruth(dataset_id);
CREATE INDEX IF NOT EXISTS ix_gt_video ON retrieval_groundtruth(dataset_id,relevant_video_id);
CREATE TABLE IF NOT EXISTS schema_versions (
  component text PRIMARY KEY, version int NOT NULL, applied_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS ingest_runs (
  run_id uuid PRIMARY KEY, dataset_id text NOT NULL, dataset_version text,
  status text NOT NULL CHECK (status IN ('created','preflight_passed','ingesting','validating','completed','failed','aborted')),
  vector_provenance text NOT NULL, model_id text, model_revision text, source_commit text,
  enabled_backends jsonb NOT NULL, enabled_dimensions jsonb NOT NULL,
  manifest_hash text NOT NULL, started_at timestamptz NOT NULL, finished_at timestamptz,
  expected_segments bigint, rows_per_backend jsonb, error_summary jsonb
);
CREATE TABLE IF NOT EXISTS dataset_active_runs (
  dataset_id text PRIMARY KEY, active_run_id uuid NOT NULL REFERENCES ingest_runs(run_id),
  activated_at timestamptz NOT NULL
);
CREATE TABLE IF NOT EXISTS ingest_chunks (
  run_id uuid NOT NULL REFERENCES ingest_runs(run_id), dataset_id text NOT NULL,
  video_id text NOT NULL, video_path text NOT NULL, chunk_index int NOT NULL,
  chunk_start_s double precision NOT NULL, chunk_end_s double precision NOT NULL,
  expected_segments int, status text NOT NULL CHECK (status IN ('pending','writing','committed','failed')),
  backend_status jsonb NOT NULL DEFAULT '{}', chunk_hash text, updated_at timestamptz NOT NULL,
  PRIMARY KEY (run_id,video_id,chunk_index)
);
CREATE TABLE IF NOT EXISTS run_videos (
  run_id uuid NOT NULL REFERENCES ingest_runs(run_id), dataset_id text NOT NULL,
  video_id text NOT NULL, source_uri text, split text, duration_s double precision,
  event_category text, PRIMARY KEY(run_id,dataset_id,video_id)
);
CREATE TABLE IF NOT EXISTS run_segments (
  run_id uuid NOT NULL REFERENCES ingest_runs(run_id), segment_id text NOT NULL,
  dataset_id text NOT NULL, video_id text NOT NULL, t_start double precision NOT NULL,
  t_end double precision NOT NULL, caption text, chunk_index int NOT NULL,
  PRIMARY KEY(run_id,segment_id)
);
CREATE TABLE IF NOT EXISTS run_segment_metadata (
  run_id uuid NOT NULL, segment_id text NOT NULL, person_count int, vehicle_count int,
  bus_count int, object_classes text[], brightness real, camera_motion real,
  PRIMARY KEY(run_id,segment_id)
);
CREATE TABLE IF NOT EXISTS run_segment_telemetry (
  run_id uuid NOT NULL, segment_id text NOT NULL, timestamp_start timestamptz,
  timestamp_end timestamptz, latitude double precision, longitude double precision,
  altitude_m real, velocity_mps real, roll real, pitch real, yaw real, yaw_rate real,
  gimbal_pitch real, gimbal_heading real, compass_heading real, imu_summary jsonb,
  extra jsonb NOT NULL DEFAULT '{}', PRIMARY KEY(run_id,segment_id)
);
CREATE TABLE IF NOT EXISTS run_retrieval_groundtruth (
  run_id uuid NOT NULL, dataset_id text NOT NULL, query_id text NOT NULL,
  query_text text NOT NULL, relevant_segment_id text NOT NULL, relevant_video_id text NOT NULL,
  relevance_rank int NOT NULL, caption_index int, caption_source text NOT NULL,
  PRIMARY KEY(run_id,dataset_id,query_id,relevant_segment_id)
);
CREATE TABLE IF NOT EXISTS telemetry_field_registry (
  run_id uuid NOT NULL, dataset_id text NOT NULL, field_name text NOT NULL,
  source_name text NOT NULL, field_type text NOT NULL, unit text, semantics jsonb NOT NULL DEFAULT '{}',
  PRIMARY KEY(dataset_id,run_id,field_name)
);
CREATE INDEX IF NOT EXISTS ix_run_segments_dataset ON run_segments(run_id,dataset_id);
CREATE INDEX IF NOT EXISTS ix_run_segments_chunk ON run_segments(run_id,video_id,chunk_index);
CREATE INDEX IF NOT EXISTS ix_ingest_runs_dataset_status ON ingest_runs(dataset_id,status);
"""

VECTOR_TABLES = {
    2048: ("emb_pg_2048", "halfvec"),
    1024: ("emb_pg_1024", "vector"),
    512: ("emb_pg_512", "vector"),
    256: ("emb_pg_256", "vector"),
}


def _driver():
    import psycopg2
    import psycopg2.extras

    return psycopg2, psycopg2.extras


@contextmanager
def connection():
    psycopg2, _ = _driver()
    conn = psycopg2.connect(
        host=settings.pg_host,
        port=settings.pg_port,
        dbname=settings.pg_db,
        user=settings.pg_user,
        password=settings.pg_password,
        connect_timeout=5,
    )
    try:
        yield conn
    finally:
        conn.close()


def health() -> bool:
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            return cur.fetchone()[0] == 1
    except Exception:
        return False


def schema_status(
    *, dimensions: tuple[int, ...] = DIMENSIONS, include_vectors: bool = True,
) -> dict[str, bool]:
    required = {
        "datasets", "videos", "segments", "segment_metadata", "segment_telemetry",
        "retrieval_groundtruth",
    }
    if include_vectors:
        required.update(VECTOR_TABLES[dimension][0] for dimension in dimensions)
        if 1024 in dimensions:
            required.add("emb_pg_1024h")
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname='public'")
            present = {row[0] for row in cur.fetchall()}
        return {table: table in present for table in sorted(required)}
    except Exception:
        return {table: False for table in sorted(required)}


def dataset_info(dataset_id: str) -> dict[str, Any] | None:
    with connection() as conn:
        _, extras = _driver()
        with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT dataset_id,has_telemetry,has_captions,vector_provenance FROM datasets WHERE dataset_id=%s",
                (dataset_id,),
            )
            row = cur.fetchone()
            return None if row is None else dict(row)


def groundtruth_stats(dataset_id: str) -> dict[str, Any]:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT count(*),count(DISTINCT query_id),count(DISTINCT relevant_video_id),
                      count(*) FILTER (WHERE caption_source='unknown'),
                      count(*) FILTER (WHERE relevant_video_id NOT LIKE 'test__%%')
               FROM retrieval_groundtruth WHERE dataset_id=%s""",
            (dataset_id,),
        )
        count, queries, videos, unknown, non_test = cur.fetchone()
    return {
        "rows": int(count), "unique_queries": int(queries), "videos": int(videos),
        "caption_source_unknown": int(unknown), "non_test_video_ids": int(non_test),
    }


def init_schema(
    *, dimensions: tuple[int, ...] = DIMENSIONS, include_vectors: bool = True,
) -> None:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
        if include_vectors:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            for dimension in dimensions:
                table, vector_type = VECTOR_TABLES[dimension]
                cur.execute(
                    f"CREATE TABLE IF NOT EXISTS {table} ("
                    f"segment_id text PRIMARY KEY, dataset_id text, v {vector_type}({dimension}))"
                )
                cur.execute(
                    f"CREATE TABLE IF NOT EXISTS {table}_runs ("
                    f"run_id uuid NOT NULL, segment_id text NOT NULL, dataset_id text NOT NULL, "
                    f"chunk_index int NOT NULL, v {vector_type}({dimension}), PRIMARY KEY(run_id,segment_id))"
                )
                if dimension == 1024:
                    cur.execute(
                        "CREATE TABLE IF NOT EXISTS emb_pg_1024h ("
                        "segment_id text PRIMARY KEY, dataset_id text, v halfvec(1024))"
                    )
        conn.commit()


def _execute_values(cur, sql: str, rows: list[tuple[Any, ...]]) -> None:
    if rows:
        _, extras = _driver()
        extras.execute_values(cur, sql, rows, page_size=500)


def upsert_dataset_bundle(
    dataset: tuple[Any, ...],
    videos: list[tuple[Any, ...]],
    segments: list[tuple[Any, ...]],
    metadata: list[tuple[Any, ...]],
    telemetry: list[tuple[Any, ...]],
    groundtruth: list[tuple[Any, ...]] | None = None,
) -> None:
    with connection() as conn, conn.cursor() as cur:
        dataset_id = str(dataset[0])
        # Bundle, dataset'in tam snapshot'idir. Bu temizlik onceki train+test
        # CapERA kalintilarinin test-only kalite kapsaminda kalmasini engeller.
        cur.execute("DELETE FROM retrieval_groundtruth WHERE dataset_id=%s", (dataset_id,))
        cur.execute(
            "DELETE FROM segment_metadata WHERE segment_id IN "
            "(SELECT segment_id FROM segments WHERE dataset_id=%s)", (dataset_id,)
        )
        cur.execute(
            "DELETE FROM segment_telemetry WHERE segment_id IN "
            "(SELECT segment_id FROM segments WHERE dataset_id=%s)", (dataset_id,)
        )
        cur.execute("DELETE FROM segments WHERE dataset_id=%s", (dataset_id,))
        cur.execute("DELETE FROM videos WHERE dataset_id=%s", (dataset_id,))
        cur.execute(
            """INSERT INTO datasets VALUES (%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (dataset_id) DO UPDATE SET
               dataset_version=EXCLUDED.dataset_version, source_hash=EXCLUDED.source_hash,
               license=EXCLUDED.license, has_telemetry=EXCLUDED.has_telemetry,
               has_captions=EXCLUDED.has_captions, vector_provenance=EXCLUDED.vector_provenance""",
            dataset,
        )
        _execute_values(
            cur,
            """INSERT INTO videos(dataset_id,video_id,source_uri,split,duration_s,event_category) VALUES %s
               ON CONFLICT(dataset_id,video_id) DO UPDATE SET source_uri=EXCLUDED.source_uri,
               split=EXCLUDED.split,duration_s=EXCLUDED.duration_s,event_category=EXCLUDED.event_category""",
            videos,
        )
        _execute_values(
            cur,
            """INSERT INTO segments(segment_id,dataset_id,video_id,t_start,t_end,caption) VALUES %s
               ON CONFLICT(segment_id) DO UPDATE SET caption=EXCLUDED.caption""",
            segments,
        )
        _execute_values(
            cur,
            """INSERT INTO segment_metadata(segment_id,person_count,vehicle_count,bus_count,object_classes,brightness,camera_motion) VALUES %s
               ON CONFLICT(segment_id) DO UPDATE SET person_count=EXCLUDED.person_count,
               vehicle_count=EXCLUDED.vehicle_count,bus_count=EXCLUDED.bus_count,
               object_classes=EXCLUDED.object_classes,brightness=EXCLUDED.brightness,camera_motion=EXCLUDED.camera_motion""",
            metadata,
        )
        _execute_values(
            cur,
            """INSERT INTO segment_telemetry(segment_id,timestamp_start,timestamp_end,latitude,longitude,
               altitude_m,velocity_mps,roll,pitch,yaw,yaw_rate,gimbal_pitch,gimbal_heading,compass_heading,imu_summary)
               VALUES %s ON CONFLICT(segment_id) DO UPDATE SET altitude_m=EXCLUDED.altitude_m,
               velocity_mps=EXCLUDED.velocity_mps,roll=EXCLUDED.roll,pitch=EXCLUDED.pitch,
               yaw=EXCLUDED.yaw,yaw_rate=EXCLUDED.yaw_rate,gimbal_pitch=EXCLUDED.gimbal_pitch,
               gimbal_heading=EXCLUDED.gimbal_heading,compass_heading=EXCLUDED.compass_heading""",
            telemetry,
        )
        _execute_values(
            cur,
            """INSERT INTO retrieval_groundtruth(
                 dataset_id,query_id,query_text,relevant_segment_id,relevant_video_id,
                 relevance_rank,caption_index,caption_source
               ) VALUES %s
               ON CONFLICT(dataset_id,query_id,relevant_segment_id) DO UPDATE SET
                 query_text=EXCLUDED.query_text,relevant_video_id=EXCLUDED.relevant_video_id,
                 relevance_rank=EXCLUDED.relevance_rank,caption_index=EXCLUDED.caption_index,
                 caption_source=EXCLUDED.caption_source""",
            groundtruth or [],
        )
        conn.commit()


def upsert_vectors(dimension: int, rows: Iterable[tuple[str, str, list[float]]], *, half_1024: bool = False) -> None:
    table = "emb_pg_1024h" if dimension == 1024 and half_1024 else VECTOR_TABLES[dimension][0]
    values = [(segment_id, dataset_id, _vector_literal(vector)) for segment_id, dataset_id, vector in rows]
    with connection() as conn, conn.cursor() as cur:
        for dataset_id in sorted({row[1] for row in values}):
            cur.execute(f"DELETE FROM {table} WHERE dataset_id=%s", (dataset_id,))
        _execute_values(
            cur,
            f"INSERT INTO {table}(segment_id,dataset_id,v) VALUES %s ON CONFLICT(segment_id) DO UPDATE SET v=EXCLUDED.v",
            values,
        )
        conn.commit()


def create_vector_indexes(dimensions: tuple[int, ...] = DIMENSIONS) -> None:
    statements_by_dimension = {
        2048: ("CREATE INDEX IF NOT EXISTS ix_pg_2048_hnsw ON emb_pg_2048 USING hnsw (v halfvec_cosine_ops) WITH (m=16, ef_construction=128)",),
        1024: (
            "CREATE INDEX IF NOT EXISTS ix_pg_1024_hnsw ON emb_pg_1024 USING hnsw (v vector_cosine_ops) WITH (m=16, ef_construction=128)",
            "CREATE INDEX IF NOT EXISTS ix_pg_1024h_hnsw ON emb_pg_1024h USING hnsw (v halfvec_cosine_ops) WITH (m=16, ef_construction=128)",
        ),
        512: ("CREATE INDEX IF NOT EXISTS ix_pg_512_hnsw ON emb_pg_512 USING hnsw (v vector_cosine_ops) WITH (m=16, ef_construction=128)",),
        256: ("CREATE INDEX IF NOT EXISTS ix_pg_256_hnsw ON emb_pg_256 USING hnsw (v vector_cosine_ops) WITH (m=16, ef_construction=128)",),
    }
    with connection() as conn, conn.cursor() as cur:
        for dimension in dimensions:
            for statement in statements_by_dimension[dimension]:
                cur.execute(statement)
        conn.commit()


def _vector_literal(vector: Iterable[float]) -> str:
    return "[" + ",".join(f"{float(value):.9g}" for value in vector) + "]"


def filter_segment_ids(dataset_id: str, metadata_filters: dict[str, Any], telemetry_filters: dict[str, Any]) -> list[str]:
    clauses = ["s.dataset_id=%s"]
    params: list[Any] = [dataset_id]
    joins = ["JOIN videos v ON v.dataset_id=s.dataset_id AND v.video_id=s.video_id"]
    allowed_meta = {"event_category": "v.event_category", "split": "v.split", "video_id": "s.video_id"}
    for key, column in allowed_meta.items():
        value = (metadata_filters or {}).get(key)
        if value not in (None, "", []):
            clauses.append(f"{column}=%s")
            params.append(value)
    if telemetry_filters:
        joins.append("JOIN segment_telemetry t ON t.segment_id=s.segment_id")
        allowed_telemetry = {
            "altitude_m": "t.altitude_m",
            "velocity_mps": "t.velocity_mps",
            "gimbal_pitch": "t.gimbal_pitch",
        }
        for key, column in allowed_telemetry.items():
            bounds = telemetry_filters.get(key)
            if not bounds:
                continue
            lo, hi = bounds
            if lo is not None:
                clauses.append(f"{column}>=%s")
                params.append(float(lo))
            if hi is not None:
                clauses.append(f"{column}<=%s")
                params.append(float(hi))
    sql = f"SELECT s.segment_id FROM segments s {' '.join(joins)} WHERE {' AND '.join(clauses)} ORDER BY s.segment_id"
    with connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return [row[0] for row in cur.fetchall()]


def get_active_run_snapshot(dataset_id: str) -> dict[str, Any] | None:
    """Read the request-wide active run and provenance in one statement."""
    with connection() as conn:
        _, extras = _driver()
        with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT r.run_id::text,r.dataset_id,r.vector_provenance,r.model_id,
                          r.model_revision,r.source_commit,r.manifest_hash
                   FROM dataset_active_runs a JOIN ingest_runs r ON r.run_id=a.active_run_id
                   WHERE a.dataset_id=%s AND r.status='completed'""",
                (dataset_id,),
            )
            row = cur.fetchone()
            return None if row is None else dict(row)


def filter_run_segment_ids(
    dataset_id: str, run_id: str, metadata_filters: dict[str, Any], telemetry_filters: dict[str, Any],
) -> list[str]:
    clauses = ["s.dataset_id=%s", "s.run_id=%s"]
    params: list[Any] = [dataset_id, run_id]
    joins = [
        "JOIN run_videos v ON v.run_id=s.run_id AND v.dataset_id=s.dataset_id AND v.video_id=s.video_id",
        "LEFT JOIN run_segment_telemetry t ON t.run_id=s.run_id AND t.segment_id=s.segment_id",
        "LEFT JOIN run_segment_metadata m ON m.run_id=s.run_id AND m.segment_id=s.segment_id",
    ]
    predicate_sql, predicate_params = sql_predicates(
        normalize_filters(metadata_filters, telemetry_filters), POSTGRES_RUN_COLUMNS,
    )
    if predicate_sql:
        clauses.append(predicate_sql)
        params.extend(predicate_params)
    sql = f"SELECT s.segment_id FROM run_segments s {' '.join(joins)} WHERE {' AND '.join(clauses)} ORDER BY s.segment_id"
    with connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return [row[0] for row in cur.fetchall()]


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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    table, vector_type = VECTOR_TABLES[dimension]
    if run_id is not None:
        table = f"{table}_runs"
    literal = _vector_literal(query_vector)
    candidate_clause = ""
    params: list[Any] = [literal, dataset_id]
    run_clause = ""
    if run_id is not None:
        run_clause = " AND e.run_id=%s"
        params.append(run_id)
        joins = (
            f"{table} e JOIN run_segments s ON s.run_id=e.run_id AND s.segment_id=e.segment_id "
            "JOIN run_videos v ON v.run_id=s.run_id AND v.dataset_id=s.dataset_id AND v.video_id=s.video_id "
            "LEFT JOIN run_segment_telemetry t ON t.run_id=s.run_id AND t.segment_id=s.segment_id "
            "LEFT JOIN run_segment_metadata m ON m.run_id=s.run_id AND m.segment_id=s.segment_id"
        )
    else:
        joins = (
            f"{table} e JOIN segments s ON s.segment_id=e.segment_id "
            "JOIN videos v ON v.dataset_id=s.dataset_id AND v.video_id=s.video_id "
            "LEFT JOIN segment_telemetry t ON t.segment_id=s.segment_id "
            "LEFT JOIN segment_metadata m ON m.segment_id=s.segment_id"
        )
    predicate_sql, predicate_params = sql_predicates(
        normalize_filters(metadata_filters, telemetry_filters), POSTGRES_RUN_COLUMNS,
    )
    predicate_clause = "" if not predicate_sql else f" AND {predicate_sql}"
    params.extend(predicate_params)
    if candidate_ids is not None:
        if not candidate_ids:
            return [], {"plan_used_vector_index": False, "indexed_vectors_count": None, "notes": []}
        candidate_clause = " AND e.segment_id=ANY(%s)"
        params.append(candidate_ids)
    params.extend([literal, top_k])
    sql = f"""SELECT e.segment_id, 1-(e.v <=> %s::{vector_type}({dimension})) AS score
              FROM {joins} WHERE e.dataset_id=%s{run_clause}{predicate_clause}{candidate_clause}
              ORDER BY e.v <=> %s::{vector_type}({dimension}) LIMIT %s"""
    with connection() as conn, conn.cursor() as cur:
        for statement in pgvector_session_settings(strategy):
            cur.execute(statement)
        count_params: list[Any] = [dataset_id]
        if run_id is not None:
            count_params.append(run_id)
        count_params.extend(predicate_params)
        cur.execute(
            f"SELECT count(*) FROM {joins} WHERE e.dataset_id=%s{run_clause}{predicate_clause}",
            count_params,
        )
        candidate_count = int(cur.fetchone()[0])
        cur.execute(sql, params)
        rows = [{"segment_id": row[0], "score": float(row[1])} for row in cur.fetchall()]
        plan_used = strategy != "exact"
    return rows, {
        "plan_used_vector_index": plan_used, "indexed_vectors_count": None,
        "candidate_count": candidate_count, "notes": [],
    }


def hydrate(segment_ids: list[str], *, run_id: str | None = None) -> list[dict[str, Any]]:
    if not segment_ids:
        return []
    if run_id is not None:
        with connection() as conn:
            _, extras = _driver()
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    """SELECT s.segment_id,s.video_id,s.t_start,s.t_end,s.caption,v.source_uri AS file_path,
                              t.latitude,t.longitude,t.altitude_m,t.velocity_mps,t.roll,t.pitch,t.yaw,
                              t.yaw_rate,t.gimbal_pitch,t.gimbal_heading,t.compass_heading,
                              v.event_category,v.split,m.person_count,m.vehicle_count,m.bus_count,
                              coalesce((t.extra->>'is_night')::boolean,false) AS is_night,t.extra
                       FROM run_segments s
                       JOIN run_videos v ON v.run_id=s.run_id AND v.dataset_id=s.dataset_id AND v.video_id=s.video_id
                       LEFT JOIN run_segment_telemetry t ON t.run_id=s.run_id AND t.segment_id=s.segment_id
                       LEFT JOIN run_segment_metadata m ON m.run_id=s.run_id AND m.segment_id=s.segment_id
                       WHERE s.run_id=%s AND s.segment_id=ANY(%s)""",
                    (run_id, segment_ids),
                )
                return [dict(row) for row in cur.fetchall()]
    with connection() as conn:
        _, extras = _driver()
        with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT s.segment_id,s.video_id,s.t_start,s.t_end,s.caption,v.source_uri AS file_path,
                          t.latitude,t.longitude,t.altitude_m,t.velocity_mps,t.roll,t.pitch,t.yaw,
                          t.yaw_rate,t.gimbal_pitch,t.gimbal_heading,t.compass_heading,
                          v.event_category,v.split,
                          m.person_count,m.vehicle_count,m.bus_count,false AS is_night,'{}'::jsonb AS extra
                   FROM segments s
                   JOIN videos v ON v.dataset_id=s.dataset_id AND v.video_id=s.video_id
                   LEFT JOIN segment_telemetry t ON t.segment_id=s.segment_id
                   LEFT JOIN segment_metadata m ON m.segment_id=s.segment_id
                   WHERE s.segment_id=ANY(%s)""",
                (segment_ids,),
            )
            return [dict(row) for row in cur.fetchall()]


def stats(
    *, dimensions: tuple[int, ...] = DIMENSIONS, include_pgvector: bool = True,
) -> list[dict[str, Any]]:
    with connection() as conn:
        _, extras = _driver()
        with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT d.dataset_id,coalesce(r.dataset_version,d.dataset_version) AS dataset_version,
                          d.license,d.has_telemetry,d.has_captions,
                          coalesce(r.vector_provenance,d.vector_provenance) AS vector_provenance,
                          (CASE WHEN r.run_id IS NOT NULL
                            THEN coalesce(r.expected_segments,count(DISTINCT rs.segment_id))
                            ELSE count(DISTINCT s.segment_id) END)::int AS segments,
                          r.run_id::text AS active_run_id,r.status AS run_status,
                          r.model_id,r.model_revision,r.source_commit,r.enabled_backends,
                          r.enabled_dimensions,r.finished_at AS last_ingest_at
                   FROM datasets d LEFT JOIN segments s ON s.dataset_id=d.dataset_id
                   LEFT JOIN dataset_active_runs a ON a.dataset_id=d.dataset_id
                   LEFT JOIN ingest_runs r ON r.run_id=a.active_run_id
                   LEFT JOIN run_segments rs ON rs.run_id=r.run_id AND rs.dataset_id=d.dataset_id
                   GROUP BY d.dataset_id,r.run_id ORDER BY d.dataset_id"""
            )
            datasets = [dict(row) for row in cur.fetchall()]
            for item in datasets:
                if include_pgvector:
                    counts = {}
                    for dimension in dimensions:
                        table, _ = VECTOR_TABLES[dimension]
                        if item.get("active_run_id"):
                            cur.execute(
                                f"SELECT count(*)::int AS n FROM {table}_runs WHERE dataset_id=%s AND run_id=%s",
                                (item["dataset_id"], item["active_run_id"]),
                            )
                        else:
                            cur.execute(f"SELECT count(*)::int AS n FROM {table} WHERE dataset_id=%s", (item["dataset_id"],))
                        counts[str(dimension)] = cur.fetchone()["n"]
                    item["pgvector"] = counts
                cur.execute(
                    "SELECT count(*)::int AS n FROM retrieval_groundtruth WHERE dataset_id=%s",
                    (item["dataset_id"],),
                )
                item["groundtruth"] = cur.fetchone()["n"]
            return datasets


def table_count(dataset_id: str, dimension: int) -> int:
    table, _ = VECTOR_TABLES[dimension]
    with connection() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT count(*)::int FROM {table} WHERE dataset_id=%s", (dataset_id,))
        return int(cur.fetchone()[0])


def write_run_vectors(
    run_id: str, dataset_id: str, dimension: int, chunk_index: int,
    rows: Iterable[tuple[str, list[float]]],
) -> int:
    table, _ = VECTOR_TABLES[dimension]
    values = [(run_id, segment_id, dataset_id, chunk_index, _vector_literal(vector)) for segment_id, vector in rows]
    with connection() as conn, conn.cursor() as cur:
        _execute_values(
            cur,
            f"INSERT INTO {table}_runs(run_id,segment_id,dataset_id,chunk_index,v) VALUES %s "
            "ON CONFLICT(run_id,segment_id) DO UPDATE SET chunk_index=EXCLUDED.chunk_index,v=EXCLUDED.v",
            values,
        )
        conn.commit()
    return len(values)


def delete_inactive_chunk(run_id: str, dataset_id: str, chunk_index: int, dimension: int) -> int:
    table, _ = VECTOR_TABLES[dimension]
    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM dataset_active_runs WHERE active_run_id=%s", (run_id,))
        if cur.fetchone():
            raise ValueError("destructive cleanup of an active run is forbidden")
        cur.execute(
            f"DELETE FROM {table}_runs WHERE run_id=%s AND dataset_id=%s AND chunk_index=%s",
            (run_id, dataset_id, chunk_index),
        )
        deleted = cur.rowcount
        conn.commit()
        return int(deleted)


def count_run(dataset_id: str, run_id: str, dimension: int) -> int:
    table, _ = VECTOR_TABLES[dimension]
    with connection() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {table}_runs WHERE dataset_id=%s AND run_id=%s", (dataset_id, run_id))
        return int(cur.fetchone()[0])


def delete_run(dataset_id: str, run_id: str, dimension: int) -> int:
    table, _ = VECTOR_TABLES[dimension]
    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM dataset_active_runs WHERE active_run_id=%s", (run_id,))
        if cur.fetchone():
            raise ValueError("active run cannot be deleted")
        cur.execute(f"DELETE FROM {table}_runs WHERE dataset_id=%s AND run_id=%s", (dataset_id, run_id))
        count = cur.rowcount
        conn.commit()
        return int(count)


def write_run_metadata_chunk(
    run_id: str,
    dataset_id: str,
    chunk_index: int,
    videos: list[tuple[Any, ...]],
    segments: list[tuple[Any, ...]],
    metadata: list[tuple[Any, ...]],
    telemetry: list[tuple[Any, ...]],
) -> int:
    """Write one run chunk; tuple payloads omit run_id and chunk_index."""
    with connection() as conn, conn.cursor() as cur:
        _execute_values(
            cur,
            """INSERT INTO run_videos(run_id,dataset_id,video_id,source_uri,split,duration_s,event_category)
               VALUES %s ON CONFLICT(run_id,dataset_id,video_id) DO UPDATE SET
               source_uri=EXCLUDED.source_uri,split=EXCLUDED.split,duration_s=EXCLUDED.duration_s,
               event_category=EXCLUDED.event_category""",
            [(run_id, *row) for row in videos],
        )
        _execute_values(
            cur,
            """INSERT INTO run_segments(run_id,segment_id,dataset_id,video_id,t_start,t_end,caption,chunk_index)
               VALUES %s ON CONFLICT(run_id,segment_id) DO UPDATE SET caption=EXCLUDED.caption,
               chunk_index=EXCLUDED.chunk_index""",
            [(run_id, *row, chunk_index) for row in segments],
        )
        _execute_values(
            cur,
            """INSERT INTO run_segment_metadata(
                 run_id,segment_id,person_count,vehicle_count,bus_count,object_classes,brightness,camera_motion
               ) VALUES %s ON CONFLICT(run_id,segment_id) DO UPDATE SET
                 person_count=EXCLUDED.person_count,vehicle_count=EXCLUDED.vehicle_count,
                 bus_count=EXCLUDED.bus_count,object_classes=EXCLUDED.object_classes,
                 brightness=EXCLUDED.brightness,camera_motion=EXCLUDED.camera_motion""",
            [(run_id, *row) for row in metadata],
        )
        _execute_values(
            cur,
            """INSERT INTO run_segment_telemetry(
                 run_id,segment_id,timestamp_start,timestamp_end,latitude,longitude,altitude_m,
                 velocity_mps,roll,pitch,yaw,yaw_rate,gimbal_pitch,gimbal_heading,compass_heading,imu_summary,extra
               ) VALUES %s ON CONFLICT(run_id,segment_id) DO UPDATE SET
                 altitude_m=EXCLUDED.altitude_m,velocity_mps=EXCLUDED.velocity_mps,
                 gimbal_pitch=EXCLUDED.gimbal_pitch,compass_heading=EXCLUDED.compass_heading,extra=EXCLUDED.extra""",
            [(run_id, *row) for row in telemetry],
        )
        conn.commit()
    return len(segments)


def delete_inactive_metadata_chunk(run_id: str, dataset_id: str, video_id: str, chunk_index: int) -> int:
    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM dataset_active_runs WHERE active_run_id=%s", (run_id,))
        if cur.fetchone():
            raise ValueError("destructive cleanup of an active run is forbidden")
        cur.execute(
            "SELECT segment_id FROM run_segments WHERE run_id=%s AND dataset_id=%s AND video_id=%s AND chunk_index=%s",
            (run_id, dataset_id, video_id, chunk_index),
        )
        ids = [row[0] for row in cur.fetchall()]
        for table in ("run_segment_metadata", "run_segment_telemetry"):
            cur.execute(f"DELETE FROM {table} WHERE run_id=%s AND segment_id=ANY(%s)", (run_id, ids))
        cur.execute(
            "DELETE FROM run_segments WHERE run_id=%s AND dataset_id=%s AND video_id=%s AND chunk_index=%s",
            (run_id, dataset_id, video_id, chunk_index),
        )
        count = cur.rowcount
        conn.commit()
        return int(count)


def facets(dataset_id: str) -> dict[str, Any]:
    snapshot = get_active_run_snapshot(dataset_id)
    if snapshot is not None:
        return _run_facets(dataset_id, str(snapshot["run_id"]))
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT array_remove(array_agg(DISTINCT v.event_category),NULL),
                      array_remove(array_agg(DISTINCT v.split),NULL),
                      array_agg(DISTINCT v.video_id)
               FROM videos v WHERE v.dataset_id=%s""",
            (dataset_id,),
        )
        event_categories, splits, video_ids = cur.fetchone()
        cur.execute(
            """SELECT min(t.altitude_m),max(t.altitude_m),min(t.velocity_mps),max(t.velocity_mps),
                      min(t.gimbal_pitch),max(t.gimbal_pitch)
               FROM segment_telemetry t JOIN segments s ON s.segment_id=t.segment_id WHERE s.dataset_id=%s""",
            (dataset_id,),
        )
        values = cur.fetchone()
        cur.execute(
            """SELECT min(m.person_count),max(m.person_count),min(m.vehicle_count),max(m.vehicle_count),
                      min(m.bus_count),max(m.bus_count)
               FROM segment_metadata m JOIN segments s ON s.segment_id=m.segment_id WHERE s.dataset_id=%s""",
            (dataset_id,),
        )
        counts_values = cur.fetchone()
    def bounds(lo, hi):
        return None if lo is None or hi is None else [float(lo), float(hi)]
    return {
        "event_categories": sorted(event_categories or []),
        "splits": sorted(splits or []),
        "video_ids": sorted(video_ids or []),
        "telemetry": {
            "altitude_m": bounds(values[0], values[1]),
            "velocity_mps": bounds(values[2], values[3]),
            "gimbal_pitch": bounds(values[4], values[5]),
        },
        "counts": {
            "person_count": bounds(counts_values[0], counts_values[1]),
            "vehicle_count": bounds(counts_values[2], counts_values[3]),
            "bus_count": bounds(counts_values[4], counts_values[5]),
        },
    }


def _run_facets(dataset_id: str, run_id: str) -> dict[str, Any]:
    telemetry_names = (
        "latitude", "longitude", "altitude_m", "velocity_mps", "roll", "pitch", "yaw",
        "yaw_rate", "gimbal_pitch", "gimbal_heading", "compass_heading",
    )
    aggregate = ",".join(f"min(t.{name}),max(t.{name})" for name in telemetry_names)
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT array_remove(array_agg(DISTINCT event_category),NULL),
                      array_remove(array_agg(DISTINCT split),NULL),array_agg(DISTINCT video_id)
               FROM run_videos WHERE dataset_id=%s AND run_id=%s""",
            (dataset_id, run_id),
        )
        event_categories, splits, video_ids = cur.fetchone()
        cur.execute(
            f"""SELECT {aggregate} FROM run_segment_telemetry t
                 JOIN run_segments s ON s.run_id=t.run_id AND s.segment_id=t.segment_id
                 WHERE s.dataset_id=%s AND s.run_id=%s""",
            (dataset_id, run_id),
        )
        values = cur.fetchone()
        cur.execute(
            """SELECT min(person_count),max(person_count),min(vehicle_count),max(vehicle_count),
                      min(bus_count),max(bus_count)
               FROM run_segment_metadata m JOIN run_segments s ON s.run_id=m.run_id AND s.segment_id=m.segment_id
               WHERE s.dataset_id=%s AND s.run_id=%s""",
            (dataset_id, run_id),
        )
        counts = cur.fetchone()

    def bounds(lo, hi):
        return None if lo is None or hi is None else [float(lo), float(hi)]

    telemetry = {
        name: bounds(values[index * 2], values[index * 2 + 1])
        for index, name in enumerate(telemetry_names)
    }
    return {
        "run_id": run_id, "event_categories": sorted(event_categories or []),
        "splits": sorted(splits or []), "video_ids": sorted(video_ids or []),
        "telemetry": telemetry,
        "counts": {
            "person_count": bounds(counts[0], counts[1]),
            "vehicle_count": bounds(counts[2], counts[3]),
            "bus_count": bounds(counts[4], counts[5]),
        },
        "booleans": {"is_night": [False, True]},
    }


def list_datasets() -> list[dict[str, Any]]:
    return stats(dimensions=(), include_pgvector=False)


def list_runs(dataset_id: str) -> list[dict[str, Any]]:
    with connection() as conn:
        _, extras = _driver()
        with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT r.*, (a.active_run_id=r.run_id) AS is_active
                   FROM ingest_runs r LEFT JOIN dataset_active_runs a ON a.dataset_id=r.dataset_id
                   WHERE r.dataset_id=%s ORDER BY r.started_at DESC""",
                (dataset_id,),
            )
            return [dict(row) for row in cur.fetchall()]


def run_info(run_id: str) -> dict[str, Any] | None:
    with connection() as conn:
        _, extras = _driver()
        with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM ingest_runs WHERE run_id=%s", (run_id,))
            run = cur.fetchone()
            if run is None:
                return None
            cur.execute(
                """SELECT status,count(*)::int AS chunks,sum(coalesce(expected_segments,0))::int AS expected_segments
                   FROM ingest_chunks WHERE run_id=%s GROUP BY status ORDER BY status""", (run_id,),
            )
            return {**dict(run), "chunk_summary": [dict(row) for row in cur.fetchall()]}


def resolve_media_segment(segment_id: str, run_id: str | None = None) -> dict[str, Any] | None:
    if run_id is not None:
        sql = """SELECT s.segment_id,s.run_id::text,s.dataset_id,s.video_id,s.t_start,s.t_end,v.source_uri
                 FROM run_segments s JOIN run_videos v ON v.run_id=s.run_id AND v.dataset_id=s.dataset_id AND v.video_id=s.video_id
                 WHERE s.run_id=%s AND s.segment_id=%s"""
        params = (run_id, segment_id)
    else:
        sql = """SELECT s.segment_id,s.run_id::text,s.dataset_id,s.video_id,s.t_start,s.t_end,v.source_uri
                 FROM run_segments s
                 JOIN dataset_active_runs a ON a.active_run_id=s.run_id AND a.dataset_id=s.dataset_id
                 JOIN run_videos v ON v.run_id=s.run_id AND v.dataset_id=s.dataset_id AND v.video_id=s.video_id
                 WHERE s.segment_id=%s
                 UNION ALL
                 SELECT s.segment_id,NULL::text AS run_id,s.dataset_id,s.video_id,s.t_start,s.t_end,v.source_uri
                 FROM segments s JOIN videos v ON v.dataset_id=s.dataset_id AND v.video_id=s.video_id
                 WHERE s.segment_id=%s AND NOT EXISTS (
                   SELECT 1 FROM run_segments rs JOIN dataset_active_runs a ON a.active_run_id=rs.run_id
                   WHERE rs.segment_id=%s
                 ) LIMIT 1"""
        params = (segment_id, segment_id, segment_id)
    with connection() as conn:
        _, extras = _driver()
        with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return None if row is None else dict(row)
