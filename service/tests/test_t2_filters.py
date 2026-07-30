from __future__ import annotations

import httpx

from app.bench.protocol import underfilled_classification
from faz8_support import readiness


def test_t2_underfilled_reason_distinguishes_shortage_from_ann_loss():
    assert underfilled_classification(3, 3, 10) == "candidate_shortage"
    assert underfilled_classification(100, 3, 10) == "ann_filter_loss"
    assert underfilled_classification(100, 10, 10) is None


def test_t2_negative_filter_is_expected_candidate_shortage():
    readiness("system")
    response = httpx.post("http://localhost:8000/search", json={
        "query": "traffic", "dataset_id": "auair", "backend": "clickhouse",
        "strategy": "exact", "dimension": 512,
        "telemetry_filters": {"altitude_m": [-100, -50]}, "top_k": 10,
    }, timeout=60)
    response.raise_for_status()
    diagnostics = response.json()["diagnostics"]
    assert diagnostics["returned_count"] == 0
    assert diagnostics["underfilled_reason"] == "candidate_shortage"
    assert diagnostics["quality_vs_groundtruth"] is None
