from reports.clickhouse_search_report import _compact_query_plan, render_clickhouse_report


def test_report_renders_sql_results_and_methodology():
    evidence = {
        "clickhouse_version": "26.7.1",
        "catalog_query_count": 1,
        "total_rows_across_model_tables": 7,
        "vector_exact_and_hnsw_ranking_equal": True,
        "vector_exact_and_hnsw_distances_equal": False,
        "vector_exact_and_hnsw_max_distance_delta": 0.0001,
        "settings": {"vector_search_filter_strategy": "auto"},
        "methodology": ["Bu bir self-probe testidir."],
        "queries": [{
            "query_id": "exact_filter",
            "title": "Exact filtre",
            "kind": "exact_filter",
            "description": "Kolon filtresi",
            "filename": "03_exact_filter.sql",
            "sql": "SELECT video_id FROM clips WHERE bus_count >= 1",
            "rows": [{"video_id": "v1"}],
            "row_count": 1,
            "statistics": {"elapsed": 0.001},
            "client_elapsed_ms": 2.0,
            "vector_index_in_plan": False,
            "query_plan": "ReadFromMergeTree",
        }],
    }
    output = render_clickhouse_report(evidence)
    assert "ClickHouse Search Lab" in output
    assert "03_exact_filter.sql" in output
    assert "bus_count &gt;= 1" in output
    assert "self-probe" in output
    assert "ReadFromMergeTree" in output
    assert "HNSW planı: yok" in output
    long_vector = "[" + ", ".join(str(index / 10) for index in range(32)) + "]"
    assert _compact_query_plan(long_vector) == "[query_vector omitted]"


def test_scope_badge_html_included_when_provided(monkeypatch):
    evidence = {
        "clickhouse_version": "26.7.1", "catalog_query_count": 1,
        "total_rows_across_model_tables": 7,
        "vector_exact_and_hnsw_ranking_equal": True,
        "vector_exact_and_hnsw_distances_equal": False,
        "vector_exact_and_hnsw_max_distance_delta": 0.0001,
        "settings": {}, "methodology": [], "queries": [],
    }
    out = render_clickhouse_report(evidence, scope_badge_html="<div>MARKER_BADGE</div>")
    assert "MARKER_BADGE" in out


def test_scope_badge_html_absent_by_default_matches_v1_output():
    evidence = {
        "clickhouse_version": "26.7.1", "catalog_query_count": 1,
        "total_rows_across_model_tables": 7,
        "vector_exact_and_hnsw_ranking_equal": True,
        "vector_exact_and_hnsw_distances_equal": False,
        "vector_exact_and_hnsw_max_distance_delta": 0.0001,
        "settings": {}, "methodology": [], "queries": [],
    }
    with_default = render_clickhouse_report(evidence)
    with_explicit_empty = render_clickhouse_report(evidence, scope_badge_html="")
    assert with_default == with_explicit_empty
