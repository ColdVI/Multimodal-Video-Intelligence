from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.db import clickhouse, postgres
from app.db.ingest_runs import PostgresRunStore, RunSpec, legacy_run_id
from app.db.registry import BACKEND_REGISTRY


FAZ11_SCHEMA_VERSION = 11


@dataclass(frozen=True)
class MigrationPlan:
    schema_version_before: int | None
    schema_version_after: int
    datasets: tuple[dict[str, Any], ...]
    enabled_backends: tuple[str, ...]
    enabled_dimensions: tuple[int, ...]
    destructive_steps: tuple[str, ...] = ()
    qdrant_requires_reingest: bool = False


def detect_schema_version() -> int | None:
    with postgres.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.schema_versions')")
        if cur.fetchone()[0] is None:
            return None
        cur.execute("SELECT version FROM schema_versions WHERE component='faz11'")
        row = cur.fetchone()
        return None if row is None else int(row[0])


def plan_migration() -> MigrationPlan:
    with postgres.connection() as conn:
        _, extras = postgres._driver()
        with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT d.dataset_id,d.dataset_version,d.source_hash,d.vector_provenance,
                          count(s.segment_id)::bigint AS segments
                   FROM datasets d LEFT JOIN segments s ON s.dataset_id=d.dataset_id
                   GROUP BY d.dataset_id,d.dataset_version,d.source_hash,d.vector_provenance
                   ORDER BY d.dataset_id"""
            )
            datasets = tuple(dict(row) for row in cur.fetchall())
    return MigrationPlan(
        schema_version_before=detect_schema_version(), schema_version_after=FAZ11_SCHEMA_VERSION,
        datasets=datasets, enabled_backends=settings.enabled_vector_backends,
        enabled_dimensions=settings.enabled_dimensions,
        qdrant_requires_reingest="qdrant" in settings.enabled_vector_backends,
    )


def _copy_postgres_legacy(spec: RunSpec) -> dict[str, int]:
    with postgres.connection() as conn, conn.cursor() as cur:
        params = (spec.run_id, spec.dataset_id)
        cur.execute(
            """INSERT INTO run_videos(run_id,dataset_id,video_id,source_uri,split,duration_s,event_category)
               SELECT %s,dataset_id,video_id,source_uri,split,duration_s,event_category FROM videos WHERE dataset_id=%s
               ON CONFLICT DO NOTHING""", params,
        )
        cur.execute(
            """INSERT INTO run_segments(run_id,segment_id,dataset_id,video_id,t_start,t_end,caption,chunk_index)
               SELECT %s,segment_id,dataset_id,video_id,t_start,t_end,caption,0 FROM segments WHERE dataset_id=%s
               ON CONFLICT DO NOTHING""", params,
        )
        cur.execute(
            """INSERT INTO run_segment_metadata
               SELECT %s,m.* FROM segment_metadata m JOIN segments s USING(segment_id) WHERE s.dataset_id=%s
               ON CONFLICT DO NOTHING""", params,
        )
        cur.execute(
            """INSERT INTO run_segment_telemetry(
                 run_id,segment_id,timestamp_start,timestamp_end,latitude,longitude,altitude_m,velocity_mps,
                 roll,pitch,yaw,yaw_rate,gimbal_pitch,gimbal_heading,compass_heading,imu_summary,extra)
               SELECT %s,t.*, '{}'::jsonb FROM segment_telemetry t JOIN segments s USING(segment_id)
               WHERE s.dataset_id=%s ON CONFLICT DO NOTHING""", params,
        )
        cur.execute(
            """INSERT INTO run_retrieval_groundtruth
               SELECT %s,g.* FROM retrieval_groundtruth g WHERE dataset_id=%s ON CONFLICT DO NOTHING""", params,
        )
        counts: dict[str, int] = {}
        if "pgvector" in spec.enabled_backends:
            for dimension in spec.enabled_dimensions:
                table, _ = postgres.VECTOR_TABLES[dimension]
                cur.execute(
                    f"INSERT INTO {table}_runs(run_id,segment_id,dataset_id,chunk_index,v) "
                    f"SELECT %s,segment_id,dataset_id,0,v FROM {table} WHERE dataset_id=%s ON CONFLICT DO NOTHING",
                    params,
                )
                cur.execute(f"SELECT count(*) FROM {table}_runs WHERE run_id=%s", (spec.run_id,))
                counts[f"pgvector:{dimension}"] = int(cur.fetchone()[0])
        cur.execute(
            """INSERT INTO ingest_chunks(
                 run_id,dataset_id,video_id,video_path,chunk_index,chunk_start_s,chunk_end_s,
                 expected_segments,status,backend_status,chunk_hash,updated_at)
               SELECT %s,v.dataset_id,v.video_id,coalesce(v.source_uri,''),0,0,coalesce(v.duration_s,0),
                 count(s.segment_id),'committed','{}'::jsonb,'legacy-migration',now()
               FROM videos v JOIN segments s ON s.dataset_id=v.dataset_id AND s.video_id=v.video_id
               WHERE v.dataset_id=%s GROUP BY v.dataset_id,v.video_id,v.source_uri,v.duration_s
               ON CONFLICT DO NOTHING""", params,
        )
        conn.commit()
    return counts


def _copy_clickhouse_legacy(spec: RunSpec) -> dict[str, int]:
    counts: dict[str, int] = {}
    target = clickhouse.client()
    for dimension in spec.enabled_dimensions:
        target.command(
            f"""INSERT INTO seg_ch_{dimension}_runs
                SELECT toUUID({{run_id:String}}),0,segment_id,dataset_id,video_id,t_start,t_end,
                       if(isNaN(altitude_m),NULL,altitude_m),
                       if(isNaN(velocity_mps),NULL,velocity_mps),
                       if(isNaN(gimbal_pitch),NULL,gimbal_pitch),
                       person_count,vehicle_count,is_night,embedding
                FROM seg_ch_{dimension} WHERE dataset_id={{dataset_id:String}}""",
            parameters={"run_id": spec.run_id, "dataset_id": spec.dataset_id},
        )
        counts[f"clickhouse:{dimension}"] = clickhouse.count_run(spec.dataset_id, spec.run_id, dimension)
    return counts


def apply_migration(plan: MigrationPlan) -> dict[str, Any]:
    if plan.qdrant_requires_reingest:
        return {
            "status": "blocked", "reason": "Qdrant legacy points need manifest-driven re-ingest for run-scoped IDs",
            "active_runs_changed": False, "datasets": [],
        }
    postgres.init_schema(dimensions=plan.enabled_dimensions, include_vectors="pgvector" in plan.enabled_backends)
    if "clickhouse" in plan.enabled_backends:
        clickhouse.init_schema(plan.enabled_dimensions)
    store = PostgresRunStore()
    results = []
    for dataset in plan.datasets:
        expected = int(dataset["segments"])
        run_id = legacy_run_id(dataset["dataset_id"], dataset.get("source_hash"))
        spec = RunSpec(
            run_id=run_id, dataset_id=dataset["dataset_id"], dataset_version=dataset.get("dataset_version"),
            vector_provenance=dataset["vector_provenance"], model_id=None, model_revision=None,
            source_commit=None, enabled_backends=plan.enabled_backends,
            enabled_dimensions=plan.enabled_dimensions, manifest_hash=dataset.get("source_hash") or "legacy-unknown",
            expected_segments=expected,
        )
        store.create(spec, status="validating")
        counts = _copy_postgres_legacy(spec)
        if "clickhouse" in plan.enabled_backends:
            counts.update(_copy_clickhouse_legacy(spec))
        metadata_count = store.metadata_count(run_id)
        mismatches = {key: value for key, value in counts.items() if value != expected}
        if metadata_count != expected:
            mismatches["postgres_metadata"] = metadata_count
        if mismatches:
            store.set_run_status(run_id, "failed", error_summary={"count_mismatches": mismatches})
            results.append({"dataset_id": spec.dataset_id, "run_id": run_id, "status": "failed", "counts": counts})
            continue
        store.activate(spec, counts)
        results.append({"dataset_id": spec.dataset_id, "run_id": run_id, "status": "completed", "counts": counts})
    with postgres.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO schema_versions(component,version,applied_at) VALUES ('faz11',%s,now())
               ON CONFLICT(component) DO UPDATE SET version=EXCLUDED.version,applied_at=EXCLUDED.applied_at""",
            (FAZ11_SCHEMA_VERSION,),
        )
        conn.commit()
    return {
        "status": "pass" if all(row["status"] == "completed" for row in results) else "fail",
        "schema_version": FAZ11_SCHEMA_VERSION, "active_runs_changed": bool(results), "datasets": results,
    }


def plan_as_dict(plan: MigrationPlan) -> dict[str, Any]:
    return {**asdict(plan), "generated_at_utc": datetime.now(timezone.utc).isoformat()}


__all__ = ["FAZ11_SCHEMA_VERSION", "MigrationPlan", "apply_migration", "detect_schema_version", "plan_as_dict", "plan_migration"]
