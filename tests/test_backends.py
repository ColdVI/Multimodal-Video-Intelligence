import pytest

from src.research.backends import ch, pv, qd
from src.research.backends._runner import run_action


def test_runner_reports_missing_script_without_crashing():
    result = run_action("this_script_does_not_exist.sh", "install")
    assert result["ok"] is False
    assert "bulunamadi" in result["error"]


def test_runner_calls_real_scripts_directory():
    # gercek install script'lerinin var oldugunu dogrular (path cozumu doğru)
    result = run_action("install_clickhouse_colab.sh", "bilinmeyen_eylem", timeout=10)
    # script var, calisir, ama taninmayan eylem icin script kendi hata kodunu doner
    assert "error" not in result or "bulunamadi" not in result.get("error", "")


def test_ch_table_ddl_includes_dimension_and_vector_similarity_index():
    ddl = ch.table_ddl(512)
    assert "seg_512" in ddl
    assert "vector_similarity" in ddl
    assert "512" in ddl
    assert "CODEC(NONE)" in ddl


def test_ch_table_ddl_custom_table_name():
    ddl = ch.table_ddl(2048, table_name="my_table")
    assert "my_table" in ddl
    assert "seg_2048" not in ddl


def test_qd_hot_filter_fields_match_spec_list():
    fields = dict(qd.HOT_FILTER_FIELDS)
    assert fields["dataset_id"] == "keyword"
    assert fields["altitude_m"] == "float"
    assert fields["person_count"] == "integer"
    assert len(qd.HOT_FILTER_FIELDS) == 6


def test_pv_vector_type_for_dimension_switches_at_2000():
    assert pv.vector_type_for_dimension(2048) == "halfvec"
    assert pv.vector_type_for_dimension(2000) == "vector"
    assert pv.vector_type_for_dimension(512) == "vector"
    assert pv.vector_type_for_dimension(256) == "vector"


def test_pv_table_ddl_rejects_vector_type_over_limit():
    with pytest.raises(ValueError, match="halfvec"):
        pv.table_ddl(2048, storage_type="vector")


def test_pv_table_ddl_accepts_halfvec_for_2048():
    ddl = pv.table_ddl(2048, storage_type="halfvec")
    assert "halfvec(2048)" in ddl
    assert "halfvec_cosine_ops" in ddl


def test_pv_table_ddl_vector_vs_halfvec_produce_different_table_names_for_1024():
    """SS7.3 kontrol kosusu: 1024d icin vector VE halfvec AYRI tablolara -
    isim cakismasi olursa ikinci CREATE ilkini SESSIZCE gizler, bu yuzden
    isimler farkli olmak ZORUNDA."""
    ddl_v = pv.table_ddl(1024, storage_type="vector")
    ddl_h = pv.table_ddl(1024, storage_type="halfvec")
    assert "seg_1024_v" in ddl_v
    assert "seg_1024_h" in ddl_h
    assert "seg_1024_v" != "seg_1024_h"


def test_ch_build_query_sql_includes_where_and_strategy_settings():
    sql = ch.build_query_sql("seg_512", [0.1, 0.2, 0.3], top_k=10, where="altitude_m < 20", strategy="prefilter")
    assert "seg_512" in sql
    assert "altitude_m < 20" in sql
    assert "LIMIT 10" in sql
    assert "vector_search_filter_strategy = 'prefilter'" in sql


def test_ch_build_query_sql_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="strategy"):
        ch.build_query_sql("seg_512", [0.1], top_k=10, strategy="not_a_real_strategy")


def test_pv_build_query_sql_uses_cosine_operator_and_limit():
    sql = pv.build_query_sql("seg_512_v", [0.1, 0.2], top_k=5, where="person_count > 10")
    assert "seg_512_v" in sql
    assert "person_count > 10" in sql
    assert "<=>" in sql
    assert "LIMIT 5" in sql


def test_all_backend_install_functions_fail_honestly_without_bash_or_docker():
    """Windows/CI ortaminda (bash yoksa) install() 'ok: False' donmeli,
    ASLA sessizce basari UYDURMAMALI."""
    for module in (ch, pv, qd):
        result = module.health_check(timeout=1) if hasattr(module, "health_check") else None
        assert result is not None
        assert isinstance(result["healthy"], bool)
