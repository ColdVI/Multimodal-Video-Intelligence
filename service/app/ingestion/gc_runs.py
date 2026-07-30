from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from app.config import settings
from app.db import postgres
from app.db.registry import BACKEND_REGISTRY


PROTECTED_STATUSES = {"created", "preflight_passed", "ingesting", "validating"}


def select_gc_candidates(
    runs: Iterable[dict[str, Any]], *, now: datetime, retain_previous_completed: int, min_age_hours: float,
) -> list[dict[str, Any]]:
    rows = list(runs)
    active_ids = {row["run_id"] for row in rows if row.get("is_active")}
    keep = set(active_ids)
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset_id"], []).append(row)
    for dataset_rows in by_dataset.values():
        previous = sorted(
            (row for row in dataset_rows if row["status"] == "completed" and row["run_id"] not in active_ids),
            key=lambda row: row.get("finished_at") or row["started_at"], reverse=True,
        )
        keep.update(row["run_id"] for row in previous[:retain_previous_completed])
    threshold = now - timedelta(hours=min_age_hours)
    candidates = []
    for row in rows:
        age_anchor = row.get("finished_at") or row["started_at"]
        if row["run_id"] in keep or row["status"] in PROTECTED_STATUSES or age_anchor > threshold:
            continue
        if row["status"] in {"completed", "failed", "aborted"}:
            candidates.append(row)
    return sorted(candidates, key=lambda row: (row["dataset_id"], row["started_at"]))


def _list_runs() -> list[dict[str, Any]]:
    with postgres.connection() as conn:
        _, extras = postgres._driver()
        with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT r.*, (a.active_run_id=r.run_id) AS is_active
                   FROM ingest_runs r LEFT JOIN dataset_active_runs a ON a.dataset_id=r.dataset_id
                   ORDER BY r.dataset_id,r.started_at"""
            )
            return [dict(row) for row in cur.fetchall()]


def _delete_metadata(dataset_id: str, run_id: str) -> dict[str, int]:
    with postgres.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM dataset_active_runs WHERE active_run_id=%s", (run_id,))
        if cur.fetchone():
            raise ValueError("active run cannot be garbage-collected")
        counts: dict[str, int] = {}
        for table in (
            "run_retrieval_groundtruth", "telemetry_field_registry", "run_segment_metadata",
            "run_segment_telemetry", "run_segments", "run_videos", "ingest_chunks",
        ):
            cur.execute(f"DELETE FROM {table} WHERE run_id=%s", (run_id,))
            counts[table] = int(cur.rowcount)
        cur.execute("DELETE FROM ingest_runs WHERE run_id=%s AND dataset_id=%s", (run_id, dataset_id))
        counts["ingest_runs"] = int(cur.rowcount)
        conn.commit()
        return counts


def run_gc(*, retain_previous_completed: int, min_age_hours: float, dry_run: bool) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    candidates = select_gc_candidates(
        _list_runs(), now=now, retain_previous_completed=retain_previous_completed,
        min_age_hours=min_age_hours,
    )
    report: list[dict[str, Any]] = []
    for row in candidates:
        item: dict[str, Any] = {"dataset_id": row["dataset_id"], "run_id": str(row["run_id"]), "status": row["status"]}
        if not dry_run:
            backend_counts = {}
            for backend_name in settings.enabled_vector_backends:
                adapter = BACKEND_REGISTRY[backend_name]
                for dimension in settings.enabled_dimensions:
                    backend_counts[f"{backend_name}:{dimension}"] = adapter.delete_run(
                        row["dataset_id"], str(row["run_id"]), dimension,
                    )
            item["backend_rows_deleted"] = backend_counts
            item["metadata_rows_deleted"] = _delete_metadata(row["dataset_id"], str(row["run_id"]))
        report.append(item)
    return {
        "status": "pass", "dry_run": dry_run, "generated_at_utc": now.isoformat(),
        "retain_previous_completed": retain_previous_completed, "min_age_hours": min_age_hours,
        "candidate_count": len(report), "runs": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely garbage-collect inactive run-versioned data")
    parser.add_argument("--retain-previous-completed", type=int, default=1)
    parser.add_argument("--min-age-hours", type=float, default=24.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.retain_previous_completed < 0 or args.min_age_hours < 0:
        parser.error("retention and age values must be non-negative")
    report = run_gc(
        retain_previous_completed=args.retain_previous_completed,
        min_age_hours=args.min_age_hours, dry_run=args.dry_run,
    )
    payload = json.dumps(report, indent=2, default=str, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
