from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.main import SearchRequest
from app.search.engine import search


def run_equivalence(
    *, dataset_id: str, backend: str, dimension: int, query: str,
    metadata_filters: dict[str, Any], telemetry_filters: dict[str, Any], top_k: int,
) -> dict[str, Any]:
    common = dict(
        query=query, dataset_id=dataset_id, backend=backend, strategy="exact", dimension=dimension,
        metadata_filters=metadata_filters, telemetry_filters=telemetry_filters,
        pattern="C" if backend == "pgvector" else "A", top_k=top_k, repeats=1,
    )
    legacy = search(SearchRequest(**common, filter_execution_mode="legacy_candidate_ids"))
    pushdown = search(SearchRequest(**common, filter_execution_mode="pushdown"))
    legacy_ids = [row["segment_id"] for row in legacy["results"]]
    pushdown_ids = [row["segment_id"] for row in pushdown["results"]]
    passed = legacy_ids == pushdown_ids and pushdown["diagnostics"]["filter_correctness"]
    return {
        "schema_version": 1, "status": "pass" if passed else "fail",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "dataset_id": dataset_id,
        "run_id": pushdown["run_id"], "backend": backend, "dimension": dimension,
        "strategy": "exact", "legacy_ids": legacy_ids, "pushdown_ids": pushdown_ids,
        "candidate_count": pushdown["diagnostics"]["candidate_count"],
        "filter_correctness": pushdown["diagnostics"]["filter_correctness"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare exact legacy and native-pushdown filtered IDs")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--backend", required=True, choices=("clickhouse", "qdrant", "pgvector"))
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--query", default="traffic")
    parser.add_argument("--metadata-filters", default="{}")
    parser.add_argument("--telemetry-filters", default="{}")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_equivalence(
        dataset_id=args.dataset_id, backend=args.backend, dimension=args.dimension,
        query=args.query, metadata_filters=json.loads(args.metadata_filters),
        telemetry_filters=json.loads(args.telemetry_filters), top_k=args.top_k,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
