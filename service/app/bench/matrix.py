from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

from app.bench.protocol import (
    FLOAT32_EXACT_DIMENSIONS,
    baseline_comparable,
    result_interpretable,
)
from app.bench.quality import evaluate_capera
from app.config import settings
from app.search.engine import PATTERN_EXECUTION_IMPLEMENTED
from app.search.strategies import SUPPORTED_STRATEGIES


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
FIELDS = (
    "suite", "test_id", "status", "reason", "embedding_mode", "hardware_profile",
    "interpretable", "evaluation", "query_id", "backend", "strategy", "dimension",
    "selectivity", "p50_ms", "p95_ms", "p99_ms", "candidate_count",
    "returned_count", "underfilled", "underfilled_reason", "metric", "value",
    "details_json",
)


def _hardware() -> str:
    return f"{platform.system()}-{platform.machine()}-docker-cpu"


def _row(suite: str, test_id: str, status: str, **values: Any) -> dict[str, Any]:
    base = {field: None for field in FIELDS}
    base.update({
        "suite": suite, "test_id": test_id, "status": status,
        "embedding_mode": settings.embedding_mode, "hardware_profile": _hardware(),
        **values,
    })
    return base


def _readiness(profile: str) -> dict[str, Any]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.readiness_check import collect_readiness

    return collect_readiness(profile)


def _reason(result: dict[str, Any]) -> str:
    return "; ".join(
        f"{item['id']}: {item['detail']}"
        for item in result["checks"] if item["status"] != "PASS"
    )


def _search(client: httpx.Client, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post("http://localhost:8000/search", json=payload)
    response.raise_for_status()
    return response.json()


def _payload(query: str, backend: str, strategy: str, dimension: int, **updates: Any) -> dict[str, Any]:
    payload = {
        "query": query, "dataset_id": "auair", "backend": backend,
        "strategy": strategy, "dimension": dimension,
        "metadata_filters": {}, "telemetry_filters": {},
        "pattern": "C" if backend == "pgvector" else "A",
        "top_k": 10, "repeats": 1,
    }
    payload.update(updates)
    return payload


def run_t1(client: httpx.Client, _: bool) -> list[dict[str, Any]]:
    ready = _readiness("system")
    if not ready["ready"]:
        return [_row("T1", "T1.readiness", "SKIP", reason=_reason(ready))]
    rows = []
    query = "dense traffic"
    for dimension in FLOAT32_EXACT_DIMENSIONS:
        reference = _search(client, _payload(query, "numpy_exact", "exact", dimension))
        reference_ids = [item["segment_id"] for item in reference["results"]]
        backend_ids = []
        for backend in ("clickhouse", "qdrant", "pgvector"):
            response = _search(client, _payload(query, backend, "exact", dimension))
            actual = [item["segment_id"] for item in response["results"]]
            backend_ids.append(actual)
            rows.append(_row(
                "T1", f"T1.exact.{backend}.{dimension}", "PASS" if actual == reference_ids else "FAIL",
                backend=backend, strategy="exact", dimension=dimension,
                interpretable=True, metric="exact_top10_equal", value=actual == reference_ids,
                details_json={"reference": reference_ids, "actual": actual},
            ))
        rows.append(_row(
            "T1", f"T1.cross_backend.{dimension}",
            "PASS" if backend_ids[0] == backend_ids[1] == backend_ids[2] else "FAIL",
            dimension=dimension, interpretable=True, metric="cross_backend_exact_equality",
            value=backend_ids[0] == backend_ids[1] == backend_ids[2],
        ))
    response_pg = _search(client, _payload(query, "pgvector", "exact", 2048))
    response_ch = _search(client, _payload(query, "clickhouse", "exact", 2048))
    pg_ids = {item["segment_id"] for item in response_pg["results"]}
    ch_ids = {item["segment_id"] for item in response_ch["results"]}
    rows.append(_row(
        "T1", "T1.halfvec_2048", "EXPLORATORY",
        reason="2048d pgvector is halfvec; exact equality is not a gate",
        evaluation="halfvec_quantization", dimension=2048, interpretable=True,
        metric="top10_set_agreement", value=len(pg_ids & ch_ids) / max(1, len(ch_ids)),
    ))
    return rows


def run_t2(client: httpx.Client, _: bool) -> list[dict[str, Any]]:
    ready = _readiness("system")
    if not ready["ready"]:
        return [_row("T2", "T2.readiness", "SKIP", reason=_reason(ready))]
    rows = []
    for backend in ("clickhouse", "qdrant", "pgvector"):
        response = _search(client, _payload(
            "traffic", backend, "exact", 512,
            telemetry_filters={"altitude_m": [-100, -50]},
        ))
        diagnostics = response["diagnostics"]
        ok = (
            diagnostics["returned_count"] == 0
            and diagnostics["underfilled_reason"] == "candidate_shortage"
            and diagnostics["quality_vs_groundtruth"] is None
        )
        rows.append(_row(
            "T2", f"T2.negative.{backend}", "PASS" if ok else "FAIL",
            backend=backend, strategy="exact", dimension=512, interpretable=True,
            candidate_count=diagnostics["candidate_count"],
            returned_count=diagnostics["returned_count"],
            underfilled=diagnostics["underfilled"],
            underfilled_reason=diagnostics["underfilled_reason"],
        ))
    return rows


def _semantic_queries() -> list[dict[str, Any]]:
    data = json.loads((FIXTURES / "queries_semantic.json").read_text(encoding="utf-8"))
    selected = {"S01", "S03", "S05", "S13", "S17", "S19"}
    return [item for item in data if item["id"] in selected]


def _selectivities() -> list[tuple[str, dict[str, Any]]]:
    thresholds = json.loads(
        (settings.artifacts_root / "research" / "selectivity_thresholds_v2.json").read_text(encoding="utf-8")
    )["auair"]["altitude_m"]
    return [
        ("100pct", {}),
        ("10pct", {"altitude_m": [thresholds["upper_10pct"], None]}),
        ("1pct", {"altitude_m": [thresholds["upper_1pct"], None]}),
    ]


def run_t3(client: httpx.Client, quick: bool) -> list[dict[str, Any]]:
    ready = _readiness("system")
    if not ready["ready"]:
        return [_row("T3", "T3.readiness", "SKIP", reason=_reason(ready))]
    queries = _semantic_queries()[:2] if quick else _semantic_queries()
    repetitions = 1 if quick else 10
    warmups = 0 if quick else 3
    rows = []
    reference_cache: dict[tuple[str, int, str], set[str]] = {}
    for query in queries:
        for backend in ("clickhouse", "qdrant", "pgvector"):
            for strategy in SUPPORTED_STRATEGIES[backend]:
                for dimension in (2048, 1024, 512, 256):
                    for selectivity, filters in _selectivities():
                        payload = _payload(
                            query["en"], backend, strategy, dimension,
                            telemetry_filters=filters, repeats=repetitions,
                        )
                        try:
                            for _ in range(warmups):
                                _search(client, {**payload, "repeats": 1})
                            response = _search(client, payload)
                            key = (query["id"], dimension, selectivity)
                            if key not in reference_cache:
                                reference = _search(client, _payload(
                                    query["en"], "numpy_exact", "exact", dimension,
                                    telemetry_filters=filters,
                                ))
                                reference_cache[key] = {
                                    item["segment_id"] for item in reference["results"]
                                }
                            actual = {item["segment_id"] for item in response["results"]}
                            reference_ids = reference_cache[key]
                            recall = len(actual & reference_ids) / len(reference_ids) if reference_ids else None
                            exact_gate = (
                                strategy != "exact"
                                or dimension == 2048
                                or recall == 1.0
                            )
                            diagnostics = response["diagnostics"]
                            rows.append(_row(
                                "T3", f"T3.{query['id']}.{backend}.{strategy}.{dimension}.{selectivity}",
                                "PASS" if exact_gate else "FAIL",
                                query_id=query["id"], backend=backend, strategy=strategy,
                                dimension=dimension, selectivity=selectivity,
                                p50_ms=response["timings_stats"]["p50"],
                                p95_ms=response["timings_stats"]["p95"],
                                interpretable=result_interpretable(
                                    settings.embedding_mode, strategy, "ann_recall_vs_exact"
                                ),
                                metric="ann_recall_vs_exact", value=recall,
                                candidate_count=diagnostics["candidate_count"],
                                returned_count=diagnostics["returned_count"],
                                underfilled=diagnostics["underfilled"],
                                underfilled_reason=diagnostics["underfilled_reason"],
                            ))
                        except Exception as exc:
                            rows.append(_row(
                                "T3", f"T3.{query['id']}.{backend}.{strategy}.{dimension}.{selectivity}",
                                "FAIL", reason=f"{type(exc).__name__}: {exc}",
                                query_id=query["id"], backend=backend, strategy=strategy,
                                dimension=dimension, selectivity=selectivity,
                            ))
    return rows


def run_t4(_: httpx.Client, __: bool) -> list[dict[str, Any]]:
    if not PATTERN_EXECUTION_IMPLEMENTED:
        return [_row("T4", "T4.patterns", "SKIP", reason="pattern not implemented")]
    return [_row("T4", "T4.patterns", "FAIL", reason="pattern tests must be implemented with the paths")]


def run_t5(client: httpx.Client, _: bool) -> list[dict[str, Any]]:
    ready = _readiness("system")
    if not ready["ready"]:
        return [_row("T5", "T5.readiness", "SKIP", reason=_reason(ready))]
    rows = []
    for top_n in (20, 50, 100, 200):
        response = _search(client, _payload(
            "dense traffic", "clickhouse", "exact", 2048,
            adaptive_mrl={"enabled": True, "base_dim": 256, "top_n": top_n},
        ))
        diagnostics = response["diagnostics"]
        status = "FAIL" if diagnostics["underfilled_reason"] == "ann_filter_loss" else "PASS"
        rows.append(_row(
            "T5", f"T5.adaptive.{top_n}", status, backend="clickhouse",
            strategy="exact", dimension=2048, interpretable=True,
            returned_count=diagnostics["returned_count"],
            underfilled=diagnostics["underfilled"],
            underfilled_reason=diagnostics["underfilled_reason"],
        ))
    return rows


def run_t6(client: httpx.Client, _: bool) -> list[dict[str, Any]]:
    ready = _readiness("system")
    if not ready["ready"]:
        return [_row("T6", "T6.readiness", "SKIP", reason=_reason(ready))]
    cases = [
        ("unknown_dataset", _payload("x", "clickhouse", "exact", 512, dataset_id="unknown"), 400),
        ("foreign_strategy", _payload("x", "qdrant", "iterative_scan", 512), 400),
        ("unknown_dimension", _payload("x", "clickhouse", "exact", 999), 400),
        ("unicode", _payload("🚁 drone footage", "clickhouse", "exact", 512), 200),
        ("injection", _payload("'; DROP TABLE segments; --", "clickhouse", "exact", 512), 200),
    ]
    rows = []
    for name, payload, expected in cases:
        response = client.post("http://localhost:8000/search", json=payload)
        rows.append(_row(
            "T6", f"T6.{name}", "PASS" if response.status_code == expected else "FAIL",
            reason=f"HTTP {response.status_code}, expected {expected}", interpretable=True,
        ))
    return rows


def run_t7(_: httpx.Client, __: bool) -> list[dict[str, Any]]:
    ready = _readiness("system")
    if not ready["ready"]:
        return [_row("T7", "T7.readiness", "SKIP", reason=_reason(ready))]
    stats = httpx.get("http://localhost:8000/stats", timeout=60).json()
    rows = []
    for dataset in stats["datasets"]:
        for dimension in (2048, 1024, 512, 256):
            counts = [
                int(dataset["segments"]),
                int(dataset["pgvector"][str(dimension)]),
                int(dataset["clickhouse"][str(dimension)]),
                int(dataset["qdrant"][str(dimension)]),
            ]
            rows.append(_row(
                "T7", f"T7.counts.{dataset['dataset_id']}.{dimension}",
                "PASS" if len(set(counts)) == 1 else "FAIL",
                dimension=dimension, interpretable=True, metric="row_count_equality",
                value=counts[0], details_json={"counts": counts},
            ))
        if not dataset["has_captions"]:
            rows.append(_row(
                "T7", f"T7.no_gt.{dataset['dataset_id']}",
                "PASS" if int(dataset.get("groundtruth", 0)) == 0 else "FAIL",
                interpretable=True, metric="groundtruth_rows",
                value=int(dataset.get("groundtruth", 0)),
            ))
    return rows


def run_t8(_: httpx.Client, __: bool) -> list[dict[str, Any]]:
    ready = _readiness("quality")
    if not ready["ready"]:
        return [_row("T8", "T8.readiness", "SKIP", reason=_reason(ready))]
    report = evaluate_capera(settings.artifacts_root / "embeddings")
    rows = []
    for dimension, metrics in report["by_dimension"].items():
        for metric, value in metrics.items():
            status = "FAIL" if metric == "r_at_1" and int(dimension) == 2048 and value <= 0.05 else "PASS"
            rows.append(_row(
                "T8", f"T8.quality.{dimension}.{metric}", status,
                dimension=int(dimension), interpretable=True, metric=metric, value=value,
                evaluation="quality_gate",
            ))
    for query_id, finding in report["exploratory"].items():
        rows.append(_row(
            "T8", f"T8.exploratory.{query_id}", "EXPLORATORY",
            query_id=query_id, interpretable=True, evaluation="exploratory",
            reason=finding["reason"], details_json=finding,
        ))
    rows.append(_row(
        "T8", "T8.halfvec_2048", "EXPLORATORY", dimension=2048,
        interpretable=True, evaluation="halfvec_quantization",
        details_json=report["halfvec_2048_quantization"],
    ))
    report_path = settings.artifacts_root / "research" / "capera_quality_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def run_t9(client: httpx.Client, _: bool) -> list[dict[str, Any]]:
    ready = _readiness("system")
    if not ready["ready"]:
        return [_row("T9", "T9.readiness", "SKIP", reason=_reason(ready))]
    response = _search(client, _payload("dense traffic", "clickhouse", "exact", 512, repeats=10))
    current = {
        "embedding_mode": settings.embedding_mode, "hardware_profile": _hardware(),
        "p50_ms": response["timings_stats"]["p50"],
    }
    path = settings.artifacts_root / "perf_baseline.json"
    baseline = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if baseline is None or not baseline_comparable(current, baseline):
        path.write_text(json.dumps(current, indent=2), encoding="utf-8")
        return [_row(
            "T9", "T9.baseline", "PASS", reason="new mode/hardware-specific baseline written",
            interpretable=True, p50_ms=current["p50_ms"],
        )]
    ok = current["p50_ms"] <= 2 * baseline["p50_ms"]
    return [_row(
        "T9", "T9.regression", "PASS" if ok else "FAIL",
        reason=f"current={current['p50_ms']} baseline={baseline['p50_ms']}",
        interpretable=True, p50_ms=current["p50_ms"],
    )]


def run_t10(_: httpx.Client, __: bool) -> list[dict[str, Any]]:
    try:
        import playwright  # noqa: F401
    except ImportError:
        return [_row("T10", "T10.playwright", "SKIP", reason="Playwright package/browser not installed")]
    return [_row("T10", "T10.playwright", "SKIP", reason="Playwright browser binary not verified")]


RUNNERS = {
    "T1": run_t1, "T2": run_t2, "T3": run_t3, "T4": run_t4, "T5": run_t5,
    "T6": run_t6, "T7": run_t7, "T8": run_t8, "T9": run_t9, "T10": run_t10,
}


def _serialize(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def _write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Faz 8 test matrix summary", "",
        "| Sınıf | Koşu | Geçti | Kaldı | Skip | Exploratory | Skip nedeni | Yorumlanabilir |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for suite in RUNNERS:
        selected = [row for row in rows if row["suite"] == suite]
        if not selected:
            continue
        counts = Counter(row["status"] for row in selected)
        reasons = sorted({str(row["reason"]) for row in selected if row["status"] == "SKIP"})
        interpretable = [row["interpretable"] for row in selected if row["interpretable"] is not None]
        label = "evet" if all(interpretable) else "kısmen/hayır"
        lines.append(
            f"| {suite} | {len(selected)} | {counts['PASS']} | {counts['FAIL']} | "
            f"{counts['SKIP']} | {counts['EXPLORATORY']} | {'; '.join(reasons) or '—'} | {label} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(suite: str, out: Path, *, quick: bool = False) -> list[dict[str, Any]]:
    suites = list(RUNNERS) if suite == "all" else [suite]
    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=300.0) as client:
        for selected in suites:
            rows.extend(RUNNERS[selected](client, quick))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows([
            {key: _serialize(value) for key, value in row.items()} for row in rows
        ])
    _write_summary(rows, settings.artifacts_root / "research" / "test_matrix_summary.md")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=(*RUNNERS, "all"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="T3: 2 queries, no warmup, one repeat")
    args = parser.parse_args()
    started = time.perf_counter()
    rows = run(args.suite, args.out, quick=args.quick)
    counts = Counter(row["status"] for row in rows)
    print(json.dumps({
        "suite": args.suite, "rows": len(rows), "status": counts,
        "out": str(args.out), "elapsed_s": round(time.perf_counter() - started, 3),
    }, ensure_ascii=False, default=dict))
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
