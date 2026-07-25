import pytest

from search.query import _build_sql


def test_auto_strategy_appends_no_settings():
    sql = _build_sql("1", "clips_xclip_hf_zeroshot", [0.1, 0.2], 10, "auto")
    assert "SETTINGS" not in sql


def test_exact_strategy_disables_vector_search_optimization():
    sql = _build_sql("1", "clips_xclip_hf_zeroshot", [0.1], 10, "exact")
    assert "query_plan_try_use_vector_search = 0" in sql


def test_prefilter_strategy_sets_filter_strategy():
    sql = _build_sql("1", "clips_xclip_hf_zeroshot", [0.1], 10, "prefilter")
    assert "vector_search_filter_strategy = 'prefilter'" in sql


def test_postfilter_rescore_strategy_sets_rescoring_and_multiplier():
    sql = _build_sql("1", "clips_xclip_hf_zeroshot", [0.1], 10, "postfilter_rescore")
    assert "vector_search_filter_strategy = 'postfilter'" in sql
    assert "vector_search_with_rescoring = 1" in sql
    assert "vector_search_index_fetch_multiplier = 5" in sql


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="bilinmeyen strategy"):
        _build_sql("1", "clips_xclip_hf_zeroshot", [0.1], 10, "not_a_strategy")


def test_build_sql_includes_table_and_limit():
    sql = _build_sql("bus_count >= 1", "clips_siglip2_frameavg", [0.1], 42, "auto")
    assert "FROM clips_siglip2_frameavg" in sql
    assert "WHERE bus_count >= 1" in sql
    assert "LIMIT 42" in sql
