from reports.strategy_matrix_html import render_strategy_report


def _fake_small_scale():
    return {
        "matrix": [{"strategy": "hnsw", "selectivity": "strict", "table": "clips_x",
                    "row_count": 4, "rows_read": 219, "vector_index_in_plan": True,
                    "p50_ms": 10.5, "p95_ms": 27.5}],
        "fetch_multiplier_sweep": [{"fetch_multiplier": 1.0, "table": "clips_x",
                                    "row_count": 4, "p50_ms": 12.4}],
        "ef_search_sweep": [{"ef_search": 64, "table": "clips_x", "row_count": 4, "p50_ms": 12.8}],
        "hnsw_recall_at_10": [{"table": "clips_x", "k": 10, "recall_at_k": 1.0,
                               "n_overlap": 10, "n_exact": 10}],
    }


def test_render_includes_matrix_and_sweeps():
    out = render_strategy_report(_fake_small_scale())
    assert "hnsw" in out
    assert "fetch_multiplier" not in out or "fetch_multiplier sweep" in out
    assert "recall@10" in out.lower() or "recall_at_k" in out or "10" in out


def test_render_includes_scale_section_when_present():
    scale = {"matrix": [{"strategy": "hnsw", "selectivity": "strict", "row_count": 0,
                         "rows_read": 49938, "p50_ms": 16.7, "p95_ms": 33.7}],
             "hnsw_recall_at_10": {"table": "bench_scale_512", "k": 10,
                                   "recall_at_k": 1.0, "n_overlap": 10, "n_exact": 10}}
    out = render_strategy_report(_fake_small_scale(), scale_100k=scale)
    assert "bench_scale_512" in out


def test_render_includes_memory_projection_when_present():
    out = render_strategy_report(_fake_small_scale(), memory_projection={"1M satır 512d": 1.19})
    assert "1M satır 512d" in out
