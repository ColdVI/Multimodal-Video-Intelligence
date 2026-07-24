import pathlib

import numpy as np

from notebooks.colab_dashboard import (
    aggregate_metrics,
    build_report_html,
    choose_sequences,
    rank_records,
    record_matches,
    report_warnings,
)


def test_choose_sequences_prioritizes_report_set():
    names = ["z", "uav0000072_04488_v", "a", "uav0000013_01073_v"]
    assert choose_sequences(names, 3) == [
        "uav0000013_01073_v",
        "uav0000072_04488_v",
        "a",
    ]


def test_filter_and_cosine_ranking_are_exact():
    records = [
        {"video_id": "a", "bus_count": 1, "embedding": [1.0, 0.0]},
        {"video_id": "b", "bus_count": 1, "embedding": [0.0, 1.0]},
        {"video_id": "c", "bus_count": 0, "embedding": [1.0, 0.0]},
    ]
    filters = [("bus_count", ">=", 1)]
    assert record_matches(records[0], filters)
    assert not record_matches(records[2], filters)
    ranked = rank_records(records, np.array([1.0, 0.0]), filters=filters, top_k=5)
    assert [row["video_id"] for row in ranked] == ["a", "b"]
    assert ranked[0]["score"] == 1.0


def test_aggregate_metrics_groups_model_and_filter():
    rows = [
        {"model": "m", "filter": True, "precision@k": 1.0, "recall@k": 0.5,
         "n_hits": 1, "n_gt": 2, "latency_ms": 10.0},
        {"model": "m", "filter": True, "precision@k": 0.5, "recall@k": 1.0,
         "n_hits": 2, "n_gt": 2, "latency_ms": 20.0},
    ]
    result = aggregate_metrics(rows)[0]
    assert result["queries"] == 2
    assert result["mean_precision@k"] == 0.75
    assert result["mean_recall@k"] == 0.75
    assert result["total_hits"] == 3


def test_report_keeps_methodological_warnings_visible():
    metrics = [{
        "model": "m", "filter": True, "query": "q", "category": "tekli",
        "precision@k": 1.0, "recall@k": 1.0, "n_hits": 1, "n_gt": 1,
        "latency_ms": 3.0,
    }]
    manifest = {"backend": "exact_in_memory_cosine", "selected_videos": 5, "windows": 7}
    warnings = report_warnings(metrics, manifest, top_k=10)
    assert any("kesifseldir" in item for item in warnings)
    assert any("doygunlasabilir" in item for item in warnings)
    report = build_report_html("Başlık", "Yusuf", "Not", manifest, metrics, [], 10)
    assert "exact_in_memory_cosine" in report
    assert "ClickHouse" in report
    assert "Yusuf" in report
