from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol, Sequence

from app.db import postgres


RUN_STATUSES = ("created", "preflight_passed", "ingesting", "validating", "completed", "failed", "aborted")
CHUNK_STATUSES = ("pending", "writing", "committed", "failed")
LEGACY_NAMESPACE = uuid.UUID("87776ad7-d4ff-4994-8541-da8be764cbce")


def legacy_run_id(dataset_id: str, source_hash: str | None) -> str:
    return str(uuid.uuid5(LEGACY_NAMESPACE, f"{dataset_id}:{source_hash or 'unknown'}"))


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    dataset_id: str
    dataset_version: str | None
    vector_provenance: str
    model_id: str | None
    model_revision: str | None
    source_commit: str | None
    enabled_backends: tuple[str, ...]
    enabled_dimensions: tuple[int, ...]
    manifest_hash: str
    expected_segments: int | None


@dataclass(frozen=True)
class ChunkSpec:
    run_id: str
    dataset_id: str
    video_id: str
    video_path: str
    chunk_index: int
    chunk_start_s: float
    chunk_end_s: float
    expected_segments: int


class RunStore(Protocol):
    def active_run_id(self, dataset_id: str) -> str | None: ...
    def set_run_status(self, run_id: str, status: str, *, error_summary: Mapping[str, Any] | None = None) -> None: ...
    def set_chunk_status(self, spec: ChunkSpec, status: str, backend_status: Mapping[str, Any]) -> None: ...
    def chunks(self, run_id: str) -> Sequence[Mapping[str, Any]]: ...
    def metadata_count(self, run_id: str) -> int: ...
    def duplicate_count(self, run_id: str) -> int: ...
    def activate(self, spec: RunSpec, rows_per_backend: Mapping[str, Any]) -> None: ...


class VectorBackend(Protocol):
    name: str
    def delete_inactive_chunk(self, run_id: str, dataset_id: str, chunk_index: int, dimension: int) -> int: ...
    def count_run(self, dataset_id: str, run_id: str, dimension: int) -> int: ...


class RunCoordinator:
    """Coordinates idempotent chunk recovery and atomic control-plane activation."""

    def __init__(self, store: RunStore, backends: Mapping[str, VectorBackend]):
        self.store = store
        self.backends = backends

    def begin_chunk(self, spec: ChunkSpec, run: RunSpec) -> dict[str, Any]:
        if self.store.active_run_id(spec.dataset_id) == spec.run_id:
            raise ValueError("cannot retry/delete a chunk belonging to the active run")
        statuses: dict[str, Any] = {"postgres_metadata": {"status": "pending", "rows": 0}}
        for backend_name in run.enabled_backends:
            backend = self.backends[backend_name]
            for dimension in run.enabled_dimensions:
                deleted = backend.delete_inactive_chunk(
                    spec.run_id, spec.dataset_id, spec.chunk_index, dimension,
                )
                statuses[f"{backend_name}:{dimension}"] = {
                    "status": "pending", "rows": 0, "deleted_before_retry": deleted,
                }
        self.store.set_chunk_status(spec, "writing", statuses)
        return statuses

    def commit_chunk(
        self, spec: ChunkSpec, run: RunSpec, backend_rows: Mapping[str, int], *, metadata_rows: int,
    ) -> dict[str, Any]:
        expected = spec.expected_segments
        required = [f"{backend}:{dimension}" for backend in run.enabled_backends for dimension in run.enabled_dimensions]
        mismatches = {key: backend_rows.get(key) for key in required if backend_rows.get(key) != expected}
        if metadata_rows != expected:
            mismatches["postgres_metadata"] = metadata_rows
        statuses = {
            "postgres_metadata": {"status": "committed" if metadata_rows == expected else "failed", "rows": metadata_rows},
            **{
                key: {"status": "committed" if backend_rows.get(key) == expected else "failed", "rows": backend_rows.get(key, 0)}
                for key in required
            },
        }
        self.store.set_chunk_status(spec, "failed" if mismatches else "committed", statuses)
        if mismatches:
            raise ValueError(f"chunk row-count mismatch: {mismatches}")
        return statuses

    def finalize_run(self, run: RunSpec) -> dict[str, Any]:
        self.store.set_run_status(run.run_id, "validating")
        chunks = list(self.store.chunks(run.run_id))
        errors: dict[str, Any] = {}
        incomplete = [row for row in chunks if row["status"] != "committed"]
        if incomplete:
            errors["incomplete_chunks"] = len(incomplete)
        expected = run.expected_segments if run.expected_segments is not None else sum(
            int(row["expected_segments"]) for row in chunks
        )
        metadata_rows = self.store.metadata_count(run.run_id)
        if metadata_rows != expected:
            errors["postgres_metadata"] = {"expected": expected, "actual": metadata_rows}
        duplicates = self.store.duplicate_count(run.run_id)
        if duplicates:
            errors["duplicate_segments"] = duplicates
        counts: dict[str, int] = {}
        for backend_name in run.enabled_backends:
            for dimension in run.enabled_dimensions:
                key = f"{backend_name}:{dimension}"
                actual = self.backends[backend_name].count_run(run.dataset_id, run.run_id, dimension)
                counts[key] = actual
                if actual != expected:
                    errors[key] = {"expected": expected, "actual": actual}
        if errors:
            self.store.set_run_status(run.run_id, "failed", error_summary=errors)
            return {"status": "failed", "run_id": run.run_id, "active_changed": False, "errors": errors}
        self.store.activate(run, counts)
        return {"status": "completed", "run_id": run.run_id, "active_changed": True, "rows_per_backend": counts}


class PostgresRunStore:
    def create(self, spec: RunSpec, *, status: str = "created") -> None:
        if status not in RUN_STATUSES:
            raise ValueError(status)
        with postgres.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO ingest_runs(
                     run_id,dataset_id,dataset_version,status,vector_provenance,model_id,model_revision,
                     source_commit,enabled_backends,enabled_dimensions,manifest_hash,started_at,expected_segments
                   ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s)
                   ON CONFLICT(run_id) DO NOTHING""",
                (
                    spec.run_id, spec.dataset_id, spec.dataset_version, status, spec.vector_provenance,
                    spec.model_id, spec.model_revision, spec.source_commit,
                    __import__("json").dumps(spec.enabled_backends), __import__("json").dumps(spec.enabled_dimensions),
                    spec.manifest_hash, datetime.now(timezone.utc), spec.expected_segments,
                ),
            )
            conn.commit()

    def active_run_id(self, dataset_id: str) -> str | None:
        snapshot = postgres.get_active_run_snapshot(dataset_id)
        return None if snapshot is None else str(snapshot["run_id"])

    def set_run_status(self, run_id: str, status: str, *, error_summary: Mapping[str, Any] | None = None) -> None:
        if status not in RUN_STATUSES:
            raise ValueError(status)
        import json

        with postgres.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE ingest_runs SET status=%s,error_summary=%s::jsonb,finished_at=CASE WHEN %s IN ('failed','aborted') THEN now() ELSE finished_at END WHERE run_id=%s",
                (status, json.dumps(error_summary) if error_summary is not None else None, status, run_id),
            )
            conn.commit()

    def set_chunk_status(self, spec: ChunkSpec, status: str, backend_status: Mapping[str, Any]) -> None:
        if status not in CHUNK_STATUSES:
            raise ValueError(status)
        import json

        with postgres.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO ingest_chunks(
                     run_id,dataset_id,video_id,video_path,chunk_index,chunk_start_s,chunk_end_s,
                     expected_segments,status,backend_status,updated_at
                   ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,now())
                   ON CONFLICT(run_id,video_id,chunk_index) DO UPDATE SET
                     status=EXCLUDED.status,backend_status=EXCLUDED.backend_status,updated_at=now()""",
                (
                    spec.run_id, spec.dataset_id, spec.video_id, spec.video_path, spec.chunk_index,
                    spec.chunk_start_s, spec.chunk_end_s, spec.expected_segments, status,
                    json.dumps(backend_status),
                ),
            )
            conn.commit()

    def chunks(self, run_id: str) -> Sequence[Mapping[str, Any]]:
        with postgres.connection() as conn:
            _, extras = postgres._driver()
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM ingest_chunks WHERE run_id=%s ORDER BY video_id,chunk_index", (run_id,))
                return [dict(row) for row in cur.fetchall()]

    def metadata_count(self, run_id: str) -> int:
        with postgres.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM run_segments WHERE run_id=%s", (run_id,))
            return int(cur.fetchone()[0])

    def duplicate_count(self, run_id: str) -> int:
        with postgres.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM (SELECT segment_id FROM run_segments WHERE run_id=%s GROUP BY segment_id HAVING count(*)>1) d",
                (run_id,),
            )
            return int(cur.fetchone()[0])

    def activate(self, spec: RunSpec, rows_per_backend: Mapping[str, Any]) -> None:
        import json

        with postgres.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE ingest_runs SET status='completed',finished_at=now(),rows_per_backend=%s::jsonb WHERE run_id=%s AND status='validating'",
                (json.dumps(rows_per_backend), spec.run_id),
            )
            if cur.rowcount != 1:
                raise RuntimeError("run activation transition failed")
            cur.execute(
                """INSERT INTO dataset_active_runs(dataset_id,active_run_id,activated_at) VALUES (%s,%s,now())
                   ON CONFLICT(dataset_id) DO UPDATE SET active_run_id=EXCLUDED.active_run_id,activated_at=EXCLUDED.activated_at""",
                (spec.dataset_id, spec.run_id),
            )
            conn.commit()


def new_run_spec(**values: Any) -> RunSpec:
    return RunSpec(run_id=str(uuid.uuid4()), **values)


__all__ = [
    "CHUNK_STATUSES", "RUN_STATUSES", "ChunkSpec", "PostgresRunStore", "RunCoordinator",
    "RunSpec", "legacy_run_id", "new_run_spec",
]
