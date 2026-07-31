from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
import yaml

from app.db import migrations
from app.db.ingest_runs import RunSpec, legacy_run_id


REPO_ROOT = Path(__file__).resolve().parents[2]


def _plan(datasets, *, enabled_backends=("clickhouse",), enabled_dimensions=(512,)):
    return migrations.MigrationPlan(
        schema_version_before=None, schema_version_after=migrations.FAZ11_SCHEMA_VERSION,
        datasets=tuple(datasets), enabled_backends=enabled_backends,
        enabled_dimensions=enabled_dimensions, qdrant_requires_reingest=False,
    )


class _FakeRunStore:
    """Mirrors the real PostgresRunStore contract that matters for apply_migration:
    create() is ON CONFLICT(run_id) DO NOTHING (never resets an existing row's
    status), and activate() only succeeds when the row is currently 'validating'
    (a real UPDATE ... WHERE status='validating' guard in production)."""

    def __init__(self, metadata_counts):
        self.rows: dict[str, str] = {}
        self.metadata_counts = metadata_counts
        self.activated: list[str] = []

    def create(self, spec, *, status="created"):
        self.rows.setdefault(spec.run_id, status)

    def set_run_status(self, run_id, status, *, error_summary=None):
        self.rows[run_id] = status

    def metadata_count(self, run_id):
        return self.metadata_counts.get(run_id, 0)

    def activate(self, spec, rows_per_backend):
        if self.rows.get(spec.run_id) != "validating":
            raise RuntimeError("run activation transition failed")
        self.rows[spec.run_id] = "completed"
        self.activated.append(spec.run_id)


def test_plan_migration_source_is_select_only():
    """Static contract check: plan_migration() must never write. Executing it
    against a live DB is not possible without Docker/psycopg on this host, so
    this asserts the function body contains no write statement instead."""
    source = inspect.getsource(migrations.plan_migration)
    forbidden = ("INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER")
    for keyword in forbidden:
        assert keyword not in source.upper(), f"plan_migration() must be read-only, found {keyword}"
    assert "SELECT" in source.upper()


def test_migration_never_drops_or_truncates_legacy_tables():
    source = inspect.getsource(migrations)
    for forbidden in ("DROP TABLE", "TRUNCATE", "DROP COLUMN"):
        assert forbidden not in source.upper(), f"migration module must never {forbidden} legacy schema"
    # Legacy tables (videos, segments, segment_metadata, segment_telemetry,
    # retrieval_groundtruth) may only be read (SELECT/JOIN), never written.
    legacy_tables = ("videos", "segments", "segment_metadata", "segment_telemetry", "retrieval_groundtruth")
    for table in legacy_tables:
        for match in re.finditer(rf"(INSERT INTO|UPDATE|DELETE FROM)\s+{table}\b", source, re.IGNORECASE):
            pytest.fail(f"legacy table {table!r} must not be written: {match.group(0)!r}")


def test_apply_requires_explicit_flag_and_plan_is_default_mutually_exclusive():
    from scripts.migrate_faz11_schema import main as migrate_main
    import argparse
    import inspect as _inspect

    source = _inspect.getsource(migrate_main)
    assert "required=True" in source
    assert "--plan" in source and "--apply" in source


def test_copy_clickhouse_legacy_clears_existing_run_rows_before_insert(monkeypatch):
    """Regression test for the idempotency fix: ClickHouse INSERT has no
    ON CONFLICT/UPSERT, so a retried --apply for the same deterministic legacy
    run_id must clear prior rows for that run first, or it would duplicate them."""
    calls: list[str] = []

    class _FakeTarget:
        def command(self, sql, parameters=None):
            calls.append(("insert", parameters["run_id"]))

    fake_clickhouse = type("FakeCH", (), {})()
    fake_clickhouse.client = lambda: _FakeTarget()
    fake_clickhouse.delete_run = lambda dataset_id, run_id, dimension: calls.append(("delete", run_id))
    fake_clickhouse.count_run = lambda dataset_id, run_id, dimension: 3
    monkeypatch.setattr(migrations, "clickhouse", fake_clickhouse)

    spec = RunSpec(
        run_id="run-1", dataset_id="d", dataset_version="v", vector_provenance="real",
        model_id=None, model_revision=None, source_commit=None, enabled_backends=("clickhouse",),
        enabled_dimensions=(512,), manifest_hash="h", expected_segments=3,
    )
    counts = migrations._copy_clickhouse_legacy(spec)

    assert calls == [("delete", "run-1"), ("insert", "run-1")]
    assert counts == {"clickhouse:512": 3}


def test_apply_migration_retry_after_failure_can_still_activate(monkeypatch):
    """Before the fix, store.create()'s ON CONFLICT DO NOTHING left a failed
    run's status stuck, so activate()'s WHERE status='validating' guard would
    raise on every retry even after the underlying mismatch was corrected."""
    dataset = {
        "dataset_id": "fixture", "dataset_version": "v1", "source_hash": "hash1",
        "vector_provenance": "real", "segments": 3,
    }
    plan = _plan([dataset], enabled_backends=(), enabled_dimensions=(512,))
    run_id = legacy_run_id("fixture", "hash1")

    monkeypatch.setattr(migrations.postgres, "init_schema", lambda **kwargs: None)
    monkeypatch.setattr(migrations, "_copy_clickhouse_legacy", lambda spec: {})

    # First attempt: postgres copy under-reports (simulated failure).
    store = _FakeRunStore(metadata_counts={run_id: 1})
    monkeypatch.setattr(migrations, "PostgresRunStore", lambda: store)
    monkeypatch.setattr(migrations, "_copy_postgres_legacy", lambda spec: {})
    monkeypatch.setattr(migrations.postgres, "connection", _noop_connection)

    first = migrations.apply_migration(plan)
    assert first["status"] == "fail"
    assert first["datasets"][0]["status"] == "failed"
    assert store.rows[run_id] == "failed"
    assert store.activated == []

    # Second attempt (retry): metadata now matches expected.
    store.metadata_counts[run_id] = 3
    second = migrations.apply_migration(plan)
    assert second["status"] == "pass"
    assert second["datasets"][0]["status"] == "completed"
    assert store.rows[run_id] == "completed"
    assert store.activated == [run_id]


class _NoopCursor:
    def execute(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _NoopConn:
    def cursor(self, *args, **kwargs):
        return _NoopCursor()

    def commit(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _noop_connection():
    return _NoopConn()


def test_qdrant_enabled_blocks_apply_and_leaves_active_pointer_untouched():
    plan = _plan([], enabled_backends=("qdrant",))
    plan = migrations.MigrationPlan(**{**plan.__dict__, "qdrant_requires_reingest": True})
    result = migrations.apply_migration(plan)
    assert result["status"] == "blocked"
    assert result["active_runs_changed"] is False
    assert result["datasets"] == []


def test_pgvector_lives_inside_the_same_postgres_container():
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    assert "pgvector" not in services, "pgvector must not be a separate service/container"
    assert "pgvector" in services["pg"]["image"], "pgvector extension must live in the pg service image"
