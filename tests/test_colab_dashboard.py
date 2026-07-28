import pathlib

import numpy as np

from notebooks.colab_dashboard import (
    aggregate_metrics,
    build_report_html,
    category_breakdown,
    choose_sequences,
    filters_to_sql,
    histogram_bins,
    rank_records,
    record_matches,
    report_warnings,
    worst_queries,
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
    assert any("saptanabilir" in item and "150" in item for item in warnings)
    assert any("doygunlasabilir" in item for item in warnings)
    report = build_report_html("Başlık", "Yusuf", "Not", manifest, metrics, [], 10)
    assert "exact_in_memory_cosine" in report
    assert "ClickHouse" in report
    assert "Yusuf" in report


def test_report_warnings_flags_recall_ceiling_when_n_gt_at_or_below_top_k():
    metrics = [
        {"query": "q1", "n_gt": 5, "precision@k": 1.0},  # n_gt<=top_k(10) -> tavan riski
        {"query": "q2", "n_gt": 21, "precision@k": 0.5},  # n_gt>top_k -> tavan degil
    ]
    warnings = report_warnings(metrics, None, top_k=10)
    assert any("tavan" in item and "1/2" in item for item in warnings)


def test_report_warnings_no_ceiling_flag_when_all_n_gt_above_top_k():
    metrics = [{"query": "q1", "n_gt": 21, "precision@k": 0.5}]
    warnings = report_warnings(metrics, None, top_k=10)
    assert not any("tavan" in item for item in warnings)


def test_filters_to_sql_empty_filters():
    assert filters_to_sql([]) == "(filtre yok - yalniz semantik arama)"


def test_filters_to_sql_renders_where_clause():
    filters = [("person_count", ">=", 3), ("bus_count", ">=", 1)]
    sql = filters_to_sql(filters)
    assert sql == "WHERE person_count >= 3 AND bus_count >= 1"


def test_histogram_bins_counts_correctly():
    values = [0.05, 0.15, 0.15, 0.95, 0.5]
    bins = histogram_bins(values, n_bins=10, lo=0.0, hi=1.0)
    assert len(bins) == 10
    assert sum(bins) == 5
    assert bins[0] == 1  # 0.05 -> bin 0
    assert bins[1] == 2  # 0.15, 0.15 -> bin 1
    assert bins[9] == 1  # 0.95 -> bin 9


def test_histogram_bins_clips_out_of_range_values():
    bins = histogram_bins([-0.5, 1.5], n_bins=5)
    assert sum(bins) == 2  # kaybolmadi, en yakin uc bine tikildi
    assert bins[0] == 1
    assert bins[4] == 1


def test_histogram_bins_empty_input():
    assert histogram_bins([], n_bins=5) == [0, 0, 0, 0, 0]


def test_worst_queries_returns_lowest_scoring_first():
    rows = [
        {"query": "a", "precision@k": 0.9},
        {"query": "b", "precision@k": 0.1},
        {"query": "c", "precision@k": 0.5},
    ]
    worst = worst_queries(rows, metric="precision@k", n=2)
    assert [r["query"] for r in worst] == ["b", "c"]


def test_worst_queries_empty_rows():
    assert worst_queries([]) == []


def test_category_breakdown_reuses_bench_report_aggregation():
    rows = [
        {"category": "tekli", "n_gt": 5, "mrr": 1.0, "ndcg@10": 0.8, "map": 0.7,
         "by_k": {10: {"recall@k": 0.6, "precision@k": 0.9}}},
        {"category": "tekli", "n_gt": 3, "mrr": 0.5, "ndcg@10": 0.4, "map": 0.3,
         "by_k": {10: {"recall@k": 0.4, "precision@k": 0.7}}},
    ]
    result = category_breakdown(rows)
    assert len(result) == 1
    assert result[0]["category"] == "tekli"
    assert result[0]["n_gt"] == 4.0  # (5+3)/2
    assert result[0]["mrr"] == 0.75
