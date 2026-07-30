from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = (2048, 1024, 512, 256)
BACKENDS = ("pgvector", "clickhouse", "qdrant")


def _protocol() -> dict[str, Any]:
    config = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    capera = config["datasets"]["capera"]
    return {
        "split": str(capera["quality_split"]),
        "items": int(capera["quality_item_count"]),
        "captions_per_item": int(capera["captions_per_item"]),
        "queries": int(capera["quality_query_count"]),
    }


def _http_json(url: str, *, payload: dict[str, Any] | None = None, timeout_s: float = 30.0) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data,
        headers={} if data is None else {"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _check(check_id: str, profile: str, ok: bool, detail: str) -> dict[str, Any]:
    return {
        "id": check_id, "profile": profile,
        "status": "PASS" if ok else "FAIL", "detail": detail,
    }


def _vector_contract(vectors: np.ndarray, rows: int) -> tuple[bool, str]:
    if vectors.shape != (rows, 2048):
        return False, f"shape={vectors.shape}, expected=({rows}, 2048)"
    if vectors.dtype != np.float32:
        return False, f"dtype={vectors.dtype}, expected=float32"
    if not np.isfinite(vectors).all():
        return False, "NaN/Inf found"
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        return False, f"L2 norm max error={float(np.max(np.abs(norms - 1.0))):.7g}"
    return True, f"shape={vectors.shape}, float32, finite, L2 normalized"


def validate_capera_artifacts(
    root: Path,
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected = protocol or _protocol()
    required = {
        "items": root / "capera_2048.npy",
        "item_ids": root / "capera_ids.parquet",
        "queries": root / "capera_queries_2048.npy",
        "query_ids": root / "capera_query_ids.parquet",
        "fixed_queries": root / "query_embeddings.json",
        "manifest": root / "embedding_manifest.json",
    }
    missing = [path.name for path in required.values() if not path.exists()]
    if missing:
        return {"ok": False, "detail": f"missing: {', '.join(missing)}"}
    try:
        item_vectors = np.load(required["items"], mmap_mode="r")
        query_vectors = np.load(required["queries"], mmap_mode="r")
        item_ids = pd.read_parquet(required["item_ids"])
        query_ids = pd.read_parquet(required["query_ids"])
        manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
        fixed = json.loads(required["fixed_queries"].read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "detail": f"artifact open failed: {type(exc).__name__}: {exc}"}

    item_ok, item_detail = _vector_contract(item_vectors, expected["items"])
    query_ok, query_detail = _vector_contract(query_vectors, expected["queries"])
    required_item_columns = {"segment_id"}
    required_query_columns = {
        "query_id", "query_text", "relevant_segment_id", "relevant_video_id",
        "caption_index", "caption_source",
    }
    columns_ok = required_item_columns <= set(item_ids) and required_query_columns <= set(query_ids)
    item_unique = len(item_ids) == expected["items"] and item_ids["segment_id"].nunique() == expected["items"]
    query_unique = len(query_ids) == expected["queries"] and query_ids["query_id"].nunique() == expected["queries"]
    prefix = f"capera:{expected['split']}__"
    split_ok = (
        item_ids["segment_id"].astype(str).str.startswith(prefix).all()
        and query_ids["relevant_segment_id"].astype(str).str.startswith(prefix).all()
        and query_ids["relevant_video_id"].astype(str).str.startswith(f"{expected['split']}__").all()
    ) if columns_ok else False
    provenance_ok = (
        set(query_ids["caption_source"].astype(str).unique()) == {"unknown"}
        if columns_ok else False
    )
    per_video = query_ids.groupby("relevant_video_id").size() if columns_ok else pd.Series(dtype=int)
    caption_count_ok = (
        len(per_video) == expected["items"]
        and (per_video == expected["captions_per_item"]).all()
    )
    manifest_ok = (
        int(manifest.get("item_count", -1)) == expected["items"]
        and int(manifest.get("query_count", -1)) == expected["queries"]
        and manifest.get("split") == expected["split"]
        and manifest.get("embedding_mode") == "real"
        and bool(manifest.get("model_revision"))
    )
    fixed_queries = fixed.get("queries", fixed)
    fixed_ok = isinstance(fixed_queries, dict) and 0 < len(fixed_queries) < expected["queries"]
    ok = all((
        item_ok, query_ok, columns_ok, item_unique, query_unique, split_ok,
        provenance_ok, caption_count_ok, manifest_ok, fixed_ok,
    ))
    return {
        "ok": ok,
        "detail": (
            f"items[{item_detail}], queries[{query_detail}], item_ids={len(item_ids)}, "
            f"query_ids={len(query_ids)}, split={split_ok}, provenance_unknown={provenance_ok}, "
            f"five_per_video={caption_count_ok}, manifest={manifest_ok}, fixed_demo={len(fixed_queries)}"
        ),
        "manifest": manifest,
    }


def _system_checks(api_url: str, ui_url: str) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    try:
        health = _http_json(f"{api_url}/health")
        ok = health.get("status") == "ok" and all(health.get(name) for name in ("pg", "ch", "qdrant"))
        checks.append(_check("A0.1", "system", ok, f"health={health.get('status')}"))
    except Exception as exc:
        health = {}
        checks.append(_check("A0.1", "system", False, f"{type(exc).__name__}: {exc}"))
    try:
        ready = _http_json(f"{api_url}/readiness-data")
        schema = ready.get("pg_schema", {})
        ok = bool(schema) and all(schema.values())
        checks.append(_check("A0.2", "system", ok, f"pg_schema={schema}"))
    except Exception as exc:
        checks.append(_check("A0.2", "system", False, f"{type(exc).__name__}: {exc}"))
    try:
        stats = _http_json(f"{api_url}/stats")
        auair = next(row for row in stats["datasets"] if row["dataset_id"] == "auair")
        ok = int(auair["segments"]) == 1866 and all(
            int(auair[backend][str(dimension)]) == 1866
            for backend in BACKENDS for dimension in DIMENSIONS
        )
        checks.append(_check("A0.3", "system", ok, "AU-AIR 1866 x 4 dimensions x 3 backends"))
    except Exception as exc:
        checks.append(_check("A0.3", "system", False, f"{type(exc).__name__}: {exc}"))
    try:
        response = _http_json(
            f"{api_url}/search",
            payload={
                "query": "dense traffic", "dataset_id": "auair", "backend": "clickhouse",
                "strategy": "exact", "dimension": 512, "top_k": 10, "repeats": 1,
            },
        )
        ok = response.get("diagnostics", {}).get("returned_count", 0) > 0
        checks.append(_check("A0.4", "system", ok, "T1-T7 search path available"))
    except Exception as exc:
        checks.append(_check("A0.4", "system", False, f"{type(exc).__name__}: {exc}"))
    try:
        with urllib.request.urlopen(ui_url, timeout=10) as response:
            ok = response.status == 200
        checks.append(_check("A0.5", "system", ok, f"UI HTTP {response.status}"))
    except Exception as exc:
        checks.append(_check("A0.5", "system", False, f"{type(exc).__name__}: {exc}"))
    return checks


def _quality_checks(api_url: str, ui_url: str, artifact_root: Path) -> list[dict[str, Any]]:
    expected = _protocol()
    artifact = validate_capera_artifacts(artifact_root, expected)
    checks = [_check("A1.1", "quality", artifact["ok"], artifact["detail"])]
    try:
        stats = _http_json(f"{api_url}/stats")
        capera = next(
            (row for row in stats["datasets"] if row["dataset_id"] == "capera"),
            None,
        )
        if capera is None:
            raise ValueError("CapERA cached ingest missing")
        count_ok = int(capera["segments"]) == expected["items"] and all(
            int(capera[backend][str(dimension)]) == expected["items"]
            for backend in BACKENDS for dimension in DIMENSIONS
        )
        checks.append(_check(
            "A1.2", "quality", count_ok,
            f"CapERA DB={capera.get('segments')} items; expected {expected['items']} x 4 x 3",
        ))
    except Exception as exc:
        checks.append(_check("A1.2", "quality", False, f"{type(exc).__name__}: {exc}"))
    try:
        health = _http_json(f"{api_url}/health")
        benchmark_path = artifact_root.parent / "research" / "hybrid_text_benchmark.json"
        benchmark = (
            json.loads(benchmark_path.read_text(encoding="utf-8"))
            if benchmark_path.exists() else {}
        )
        benchmark_ok = (
            benchmark.get("fallback_basis") == "warm_p50"
            and isinstance(benchmark.get("model_load_ms"), (int, float))
            and isinstance(benchmark.get("cold_query_ms"), (int, float))
            and isinstance(benchmark.get("warm_p50_ms"), (int, float))
            and benchmark.get("synthetic_fallback") is False
        )
        selected = benchmark.get("selected_mode")
        mode_ok = (
            (
                health.get("embedding_mode") == "hybrid_text" and selected == "hybrid_text"
                or health.get("embedding_mode") == "cached" and selected == "cached_only"
            )
            and health.get("embedding", {}).get("synthetic_fallback") is False
            and benchmark_ok
        )
        checks.append(_check(
            "A1.3", "quality", mode_ok,
            (
                f"embedding_mode={health.get('embedding_mode')}, "
                f"selected_mode={selected}, warm_p50_ms={benchmark.get('warm_p50_ms')}, "
                f"synthetic_fallback={benchmark.get('synthetic_fallback')}"
            ),
        ))
    except Exception as exc:
        checks.append(_check("A1.3", "quality", False, f"{type(exc).__name__}: {exc}"))
    try:
        ready = _http_json(f"{api_url}/readiness-data")
        gt = ready["capera_groundtruth"]
        gt_ok = (
            gt["rows"] == expected["queries"]
            and gt["unique_queries"] == expected["queries"]
            and gt["videos"] == expected["items"]
            and gt["caption_source_unknown"] == expected["queries"]
            and gt["non_test_video_ids"] == 0
        )
        checks.append(_check("A1.4", "quality", gt_ok, f"groundtruth={gt}"))
    except Exception as exc:
        checks.append(_check("A1.4", "quality", False, f"{type(exc).__name__}: {exc}"))
    try:
        with urllib.request.urlopen(ui_url, timeout=10) as response:
            body = response.read().decode("utf-8", errors="ignore")
            ui_ok = response.status == 200 and "SENTET" not in body.upper()
        checks.append(_check("A1.5", "quality", ui_ok, "UI reachable; non-synthetic mode required"))
    except Exception as exc:
        checks.append(_check("A1.5", "quality", False, f"{type(exc).__name__}: {exc}"))
    return checks


def collect_readiness(
    profile: str,
    *,
    api_url: str = "http://localhost:8000",
    ui_url: str = "http://localhost:7860",
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    system = _system_checks(api_url.rstrip("/"), ui_url.rstrip("/"))
    checks = system
    if profile == "quality":
        checks = system + _quality_checks(
            api_url.rstrip("/"), ui_url.rstrip("/"),
            artifact_root or REPO_ROOT / "artifacts" / "embeddings",
        )
    ready = all(item["status"] == "PASS" for item in checks)
    return {
        "profile": profile, "ready": ready,
        "system_ready": all(item["status"] == "PASS" for item in system),
        "checks": checks,
        "note": "Quality/A1 absence never blocks the system profile or T1-T7.",
    }


def _print_table(result: dict[str, Any]) -> None:
    print(f"Readiness profile={result['profile']} ready={result['ready']}")
    print("| Check | Profile | Status | Detail |")
    print("|---|---|---|---|")
    for item in result["checks"]:
        detail = str(item["detail"]).replace("|", "/").replace("\n", " ")
        print(f"| {item['id']} | {item['profile']} | {item['status']} | {detail} |")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Faz 8 profiled readiness gate")
    parser.add_argument("--profile", choices=("system", "quality"), default="system")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--api-url", default=os.getenv("API_URL", "http://localhost:8000"))
    parser.add_argument("--ui-url", default=os.getenv("UI_URL", "http://localhost:7860"))
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args(argv)
    result = collect_readiness(
        args.profile, api_url=args.api_url, ui_url=args.ui_url,
        artifact_root=args.artifact_root,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_table(result)
    return 1 if args.strict and not result["ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
