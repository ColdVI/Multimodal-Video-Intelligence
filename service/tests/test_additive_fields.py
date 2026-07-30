from __future__ import annotations

import httpx

from faz8_support import readiness


def test_search_results_expose_additive_event_split_and_counts():
    readiness("system")
    response = httpx.post(
        "http://localhost:8000/search",
        json={
            "query": "dense traffic", "dataset_id": "auair", "backend": "clickhouse",
            "strategy": "exact", "dimension": 512, "top_k": 5, "repeats": 1,
        },
        timeout=60,
    )
    response.raise_for_status()
    results = response.json()["results"]
    assert results, "expected at least one result for auair"
    for row in results:
        for key in ("event_category", "split", "person_count", "vehicle_count", "bus_count"):
            assert key in row
        # existing contract fields untouched by the additive change
        for key in ("segment_id", "video_id", "t_start", "t_end", "caption", "file_path", "score"):
            assert key in row


def test_facets_expose_additive_counts_block_without_removing_existing_fields():
    readiness("system")
    response = httpx.get("http://localhost:8000/facets/auair", timeout=30)
    response.raise_for_status()
    data = response.json()
    for key in ("dataset_id", "event_categories", "splits", "video_ids", "telemetry"):
        assert key in data
    assert "counts" in data
    for key in ("person_count", "vehicle_count", "bus_count"):
        assert key in data["counts"]
        lo, hi = data["counts"][key]
        assert lo <= hi


def test_mixed_provenance_database_reports_auair_as_synthetic_with_danger_banner():
    """FAZ10 §1/§3.4: a mixed database (real CapERA vectors alongside synthetic AU-AIR
    vectors) must not let a global embedding_mode banner claim AU-AIR results are real.
    Provenance is a per-dataset DB column, not a settings.embedding_mode reflection."""
    readiness("system")

    stats = httpx.get("http://localhost:8000/stats", timeout=30).json()
    auair_row = next(row for row in stats["datasets"] if row["dataset_id"] == "auair")
    assert auair_row["vector_provenance"] == "synthetic"

    health = httpx.get(
        "http://localhost:8000/health", params={"dataset_id": "auair"}, timeout=30,
    ).json()
    assert health["embedding"]["vector_provenance"] == "synthetic"
    assert health["embedding"]["level"] == "danger"

    response = httpx.post(
        "http://localhost:8000/search",
        json={
            "query": "dense traffic", "dataset_id": "auair", "backend": "clickhouse",
            "strategy": "exact", "dimension": 512, "top_k": 5, "repeats": 1,
        },
        timeout=60,
    )
    response.raise_for_status()
    assert response.json()["vector_provenance"] == "synthetic"
