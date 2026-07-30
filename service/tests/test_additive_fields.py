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
