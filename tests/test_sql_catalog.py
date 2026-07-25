import pytest

from search.sql_catalog import (
    QUERY_SPECS,
    assert_read_only_sql,
    get_query_spec,
    validate_catalog,
)


def test_catalog_is_complete_and_read_only():
    validate_catalog()
    assert len(QUERY_SPECS) == 15
    assert {spec.kind for spec in QUERY_SPECS} == {
        "inventory",
        "exact_filter",
        "vector_exact",
        "vector_ann",
        "hybrid_prefilter",
        "hybrid_postfilter",
        "matrix",
    }


def test_strategy_matrix_covers_four_strategies_and_two_selectivities():
    matrix_specs = [spec for spec in QUERY_SPECS if spec.kind == "matrix"]
    assert len(matrix_specs) == 8
    assert {spec.strategy for spec in matrix_specs} == {
        "bruteforce", "hnsw", "prefilter", "postfilter_rescore",
    }
    assert {spec.selectivity for spec in matrix_specs} == {"loose", "strict"}
    for strategy in ("bruteforce", "hnsw", "prefilter", "postfilter_rescore"):
        selectivities = {s.selectivity for s in matrix_specs if s.strategy == strategy}
        assert selectivities == {"loose", "strict"}


def test_sql_for_table_swaps_table_name_only():
    spec = get_query_spec("matrix_hnsw_loose")
    swapped = spec.sql_for_table("clips_siglip2_frameavg")
    assert "clips_siglip2_frameavg" in swapped
    assert "clips_xclip_hf_zeroshot" not in swapped
    assert "person_count >= 1" in swapped


def test_exact_and_vector_contracts_are_not_conflated():
    exact = get_query_spec("exact_filter").sql()
    brute = get_query_spec("similarity_exact_bruteforce").sql()
    ann = get_query_spec("similarity_hnsw").sql()
    assert "cosineDistance" not in exact
    assert "WHERE bus_count >= 1" in exact
    assert "cosineDistance" in brute
    assert "query_plan_try_use_vector_search = 0" in brute
    assert "query_plan_try_use_vector_search = 1" in ann


def test_hybrid_prefilter_and_postfilter_settings_are_explicit():
    pre = get_query_spec("hybrid_prefilter").sql()
    post = get_query_spec("hybrid_postfilter_rescore").sql()
    assert "vector_search_filter_strategy = 'prefilter'" in pre
    assert "vector_search_filter_strategy = 'postfilter'" in post
    assert "vector_search_with_rescoring = 1" in post
    assert "vector_search_index_fetch_multiplier = 5" in post


@pytest.mark.parametrize("keyword", ["DROP", "INSERT", "ALTER", "DELETE"])
def test_read_only_guard_rejects_mutations(keyword):
    with pytest.raises(ValueError, match="yasak SQL"):
        assert_read_only_sql(f"SELECT 1; {keyword} TABLE clips_x")
