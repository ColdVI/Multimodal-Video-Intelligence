from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "service"))

from app.db.migrations import apply_migration, plan_as_dict, plan_migration  # noqa: E402


def _not_run(mode: str, exc: Exception) -> dict:
    return {
        "schema_version": 1,
        "status": "not_run",
        "mode": mode,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": f"database migration could not be inspected on this host: {type(exc).__name__}: {exc}",
        "required_command": "PYTHONPATH=service python scripts/migrate_faz11_schema.py --plan --output artifacts/faz11/schema_migration_report.json",
        "apply_command": "PYTHONPATH=service python scripts/migrate_faz11_schema.py --apply --output artifacts/faz11/schema_migration_report.json",
        "expected_environment": "Healthy configured PostgreSQL and enabled vector backends; backup/snapshot managed by the operator; no volume deletion",
        "destructive_actions_performed": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan or apply the additive Faz 11 run-scoped schema migration")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "artifacts/faz11/schema_migration_report.json")
    args = parser.parse_args()
    selected = "apply" if args.apply else "dry_run" if args.dry_run else "plan"
    try:
        plan = plan_migration()
        payload = {"schema_version": 1, "status": "pass", "mode": selected, "plan": plan_as_dict(plan)}
        if args.apply:
            payload["result"] = apply_migration(plan)
            payload["status"] = payload["result"]["status"]
    except Exception as exc:
        payload = _not_run(selected, exc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str, sort_keys=True))
    return 0 if payload["status"] == "pass" else 5


if __name__ == "__main__":
    raise SystemExit(main())
