from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import httpx

from app.config import settings
from app.search.strategies import SUPPORTED_STRATEGIES


QUERIES = [
    "kalabalık trafik", "yüksek irtifadan trafik", "hızlı uçuş", "çok sayıda araç",
    "insanların olduğu yol", "boş yol", "yoğun kavşak", "alçak irtifa",
    "yavaş drone", "park etmiş araçlar", "gündüz trafik", "gece sahnesi",
    "otobüsü göster", "yürüyen insan", "araç ve insan", "şehir üzerinde uçuş",
    "kırsal yol", "virajlı yol", "yüksek hızlı hareket", "geniş açı trafik",
]

FIELDS = [
    "block", "embedding_mode", "backend", "backend_version", "dimension", "storage_type",
    "dataset_id", "corpus_size", "scale_level", "real_unique_segments", "physical_rows",
    "replication_factor", "filter_selectivity_target", "filter_selectivity_actual",
    "strategy", "pattern", "returned_count", "underfilled", "underfilled_reason",
    "filter_correctness", "interpretable",
    "topk_agreement", "ann_recall_vs_exact", "quality_vs_groundtruth",
    "p50_ms", "p95_ms", "p99_ms", "cold_ms", "ingest_s", "index_build_s",
    "vector_storage_mb", "index_storage_mb", "metadata_storage_mb", "storage_amplification",
    "index_params_json", "settings_json", "plan_used_vector_index",
    "indexed_vectors_count", "max_limit_setting", "hardware_profile",
]

VERSIONS = {"clickhouse": "25.8", "qdrant": "1.12.4", "pgvector": "pg16"}
STORAGE_TYPES = {"clickhouse": "Array(Float32)", "qdrant": "float32", "pgvector": "vector/halfvec"}


def _thresholds(dataset_id: str) -> list[tuple[str, float | None, float | None]]:
    path = settings.artifacts_root / "research" / "selectivity_thresholds_v2.json"
    payload = json.loads(path.read_text(encoding="utf-8"))[dataset_id]["altitude_m"]
    return [
        ("100pct", None, 1.0),
        ("50pct", payload["upper_50pct"], 0.5),
        ("10pct", payload["upper_10pct"], 0.1),
        ("1pct", payload["upper_1pct"], 0.01),
        ("0.1pct", payload["upper_0_1pct"], 0.001),
    ]


def _configs(dataset_id: str):
    defaults = {"clickhouse": "prefilter", "qdrant": "prefilter", "pgvector": "iterative_scan"}
    for backend in ("clickhouse", "qdrant", "pgvector"):
        for dimension in (2048, 512):
            for strategy in SUPPORTED_STRATEGIES[backend]:
                for target, threshold, fraction in _thresholds(dataset_id):
                    yield backend, strategy, dimension, target, threshold, fraction
        for dimension in (1024, 256):
            for target, threshold, fraction in _thresholds(dataset_id):
                yield backend, defaults[backend], dimension, target, threshold, fraction


def _search(client: httpx.Client, api_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post(f"{api_url}/search", json=payload)
    response.raise_for_status()
    return response.json()


def _agreement(actual: list[dict[str, Any]], reference: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    actual_ids = {row["segment_id"] for row in actual}
    reference_ids = {row["segment_id"] for row in reference}
    if not reference_ids:
        return None, None
    intersection = len(actual_ids & reference_ids)
    recall = intersection / len(reference_ids)
    union = actual_ids | reference_ids
    jaccard = intersection / len(union) if union else None
    return jaccard, recall


def run(api_url: str, out: Path, dataset_id: str, smoke: bool) -> list[dict[str, Any]]:
    stats = httpx.get(f"{api_url}/stats", timeout=60).json()
    dataset = next(row for row in stats["datasets"] if row["dataset_id"] == dataset_id)
    corpus_size = int(dataset["segments"])
    rows: list[dict[str, Any]] = []
    hardware = f"{platform.system()}-{platform.machine()}-docker-cpu"
    configs = list(_configs(dataset_id))
    with httpx.Client(timeout=300.0) as client:
        for index, (backend, strategy, dimension, target, threshold, target_fraction) in enumerate(configs):
            query_set = [QUERIES[index % len(QUERIES)]] if smoke else QUERIES
            responses: list[dict[str, Any]] = []
            agreements: list[float] = []
            recalls: list[float] = []
            cold_ms = None
            error = None
            for query_index, query in enumerate(query_set):
                payload = {
                    "query": query,
                    "dataset_id": dataset_id,
                    "backend": backend,
                    "strategy": strategy,
                    "dimension": dimension,
                    "adaptive_mrl": {"enabled": False, "base_dim": 256, "top_n": 100},
                    "metadata_filters": {},
                    "telemetry_filters": {} if threshold is None else {"altitude_m": [threshold, None]},
                    "pattern": "C" if backend == "pgvector" else "A",
                    "top_k": 10,
                    "repeats": 1 if smoke else 10,
                }
                try:
                    response = _search(client, api_url, payload)
                    responses.append(response)
                    reference_payload = {
                        **payload,
                        "backend": "numpy_exact",
                        "strategy": "exact",
                        "pattern": "A",
                        "repeats": 1,
                    }
                    reference = _search(client, api_url, reference_payload)
                    agreement, recall = _agreement(response.get("results", []), reference.get("results", []))
                    if agreement is not None:
                        agreements.append(agreement)
                    if recall is not None:
                        recalls.append(recall)
                    if query_index == 0:
                        cold_ms = response["timings_ms"]["total"]
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    break
            last = responses[-1] if responses else None
            p50_values = sorted(float(item["timings_stats"]["p50"]) for item in responses)
            p95_values = sorted(float(item["timings_stats"]["p95"]) for item in responses)
            candidate_count = int(last["diagnostics"]["candidate_count"]) if last else 0
            diagnostics = last["diagnostics"] if last else {}
            settings_json = {
                "warmup": 0 if smoke else 3,
                "measured_repetitions": 1 if smoke else 10,
                "query_count": len(query_set),
                "execution": "smoke" if smoke else "L2",
                "error": error,
                "notes": diagnostics.get("notes", []),
            }
            rows.append({
                "block": "C", "embedding_mode": settings.embedding_mode,
                "backend": backend, "backend_version": VERSIONS[backend],
                "dimension": dimension, "storage_type": STORAGE_TYPES[backend],
                "dataset_id": dataset_id, "corpus_size": corpus_size, "scale_level": "L2",
                "real_unique_segments": corpus_size, "physical_rows": corpus_size,
                "replication_factor": 1.0, "filter_selectivity_target": target,
                "filter_selectivity_actual": candidate_count / corpus_size if corpus_size else None,
                "strategy": strategy, "pattern": "C" if backend == "pgvector" else "A",
                "returned_count": diagnostics.get("returned_count", 0),
                "underfilled": diagnostics.get("underfilled", True),
                "underfilled_reason": diagnostics.get("underfilled_reason"),
                "filter_correctness": diagnostics.get("filter_correctness", False),
                "interpretable": not (
                    settings.embedding_mode == "synthetic" and strategy != "exact"
                ),
                "topk_agreement": sum(agreements) / len(agreements) if agreements else None,
                "ann_recall_vs_exact": sum(recalls) / len(recalls) if recalls else None,
                "quality_vs_groundtruth": None,
                "p50_ms": p50_values[len(p50_values) // 2] if p50_values else None,
                "p95_ms": p95_values[min(len(p95_values) - 1, int(len(p95_values) * 0.95))] if p95_values else None,
                "p99_ms": None, "cold_ms": cold_ms, "ingest_s": None, "index_build_s": None,
                "vector_storage_mb": None, "index_storage_mb": None, "metadata_storage_mb": None,
                "storage_amplification": None,
                "index_params_json": json.dumps({"dimension": dimension}, separators=(",", ":")),
                "settings_json": json.dumps(settings_json, ensure_ascii=False, separators=(",", ":")),
                "plan_used_vector_index": diagnostics.get("plan_used_vector_index"),
                "indexed_vectors_count": diagnostics.get("indexed_vectors_count"),
                "max_limit_setting": 100, "hardware_profile": hardware,
            })
    if len(rows) < 150:
        raise AssertionError(f"benchmark row contract failed: {len(rows)} < 150")
    if any(not row["embedding_mode"] for row in rows):
        raise AssertionError("embedding_mode must be populated on every row")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L2", choices=["L2"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", default="auair")
    parser.add_argument("--api-url", default=os.getenv("BENCH_API_URL", "http://localhost:8000"))
    parser.add_argument("--smoke", action="store_true", help="one measured query per config; still emits the full 150-row matrix")
    args = parser.parse_args()
    started = time.perf_counter()
    rows = run(args.api_url.rstrip("/"), args.out, args.dataset, args.smoke)
    print(json.dumps({
        "rows": len(rows), "out": str(args.out), "embedding_mode": settings.embedding_mode,
        "execution": "smoke" if args.smoke else "L2", "elapsed_s": round(time.perf_counter() - started, 3),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
