import numpy as np
import pytest

from bench import adaptive_mrl
from datasets import visdrone as visdrone_module


class _FakeResult:
    def __init__(self, rows):
        self.result_rows = rows


class _FakeClient:
    """SQL metnindeki tablo adina gore canned satir donen sahte client -
    cok sayida (referans + adaptif x candidate_k) cagriyi cagri sirasina
    guvenmeden dogru yanitlamak icin."""
    def __init__(self, rows_by_table: dict, rerank_rows: list):
        self.rows_by_table = rows_by_table
        self.rerank_rows = rerank_rows
        self.queries = []

    def query(self, sql):
        self.queries.append(sql)
        if "WHERE (video_id, t_start) IN" in sql:
            return _FakeResult(self.rerank_rows)
        for table, rows in self.rows_by_table.items():
            if f"FROM {table}\n" in sql or sql.strip().endswith(table):
                return _FakeResult(rows)
        for table, rows in self.rows_by_table.items():
            if table in sql:
                return _FakeResult(rows)
        return _FakeResult([])


class _FakeEmbedder:
    def embed_text(self, text):
        vec = np.zeros(2048, dtype=np.float32)
        vec[0] = 1.0
        return vec


@pytest.fixture(autouse=True)
def _mock_embedder(monkeypatch):
    monkeypatch.setattr(adaptive_mrl, "get_embedder", lambda name: _FakeEmbedder())


def _client_with_uniform_rows():
    rows = [("uav0000013_01073_v", 0.0, 2.32, 0.1)]
    rows_by_table = {f"clips_qwen3vl_emb_{d}": rows for d in (256, 512, 2048)}
    return _FakeClient(rows_by_table, rerank_rows=rows)


def test_run_query_returns_14_rows_6_reference_plus_2x4_adaptive():
    ch = _client_with_uniform_rows()
    gt_by_vid = {"uav0000013_01073_v": [(0.0, 2.32)]}
    rows = adaptive_mrl.run_query(ch, "visdrone", "otobüsü göster", gt_by_vid,
                                  use_filters=True, merge_gap_tol=10.0)
    assert len(rows) == 6 + 2 * 4

    reference_rows = [r for r in rows if r["candidate_k"] is None]
    adaptive_rows = [r for r in rows if r["candidate_k"] is not None]
    assert len(reference_rows) == 6
    assert len(adaptive_rows) == 8


def test_reference_rows_have_no_candidate_k_or_stage_latencies():
    ch = _client_with_uniform_rows()
    gt_by_vid = {"uav0000013_01073_v": [(0.0, 2.32)]}
    rows = adaptive_mrl.run_query(ch, "visdrone", "otobüsü göster", gt_by_vid,
                                  use_filters=True, merge_gap_tol=10.0)
    ref = next(r for r in rows if r["strategy"] == "2048d_exact")
    assert ref["candidate_k"] is None
    assert ref["stage1_dimension"] is None
    assert ref["stage1_latency_s"] is None
    assert ref["agreement_vs_2048_exact_at10"] is None
    assert ref["rerank_dimension"] == 2048


def test_adaptive_rows_sweep_all_four_candidate_k_values():
    ch = _client_with_uniform_rows()
    gt_by_vid = {"uav0000013_01073_v": [(0.0, 2.32)]}
    rows = adaptive_mrl.run_query(ch, "visdrone", "otobüsü göster", gt_by_vid,
                                  use_filters=True, merge_gap_tol=10.0)
    ks_seen = sorted({r["candidate_k"] for r in rows
                      if r["strategy"] == "adaptive_mrl_256_to_2048"})
    assert ks_seen == list(adaptive_mrl.CANDIDATE_K_SWEEP)


def test_adaptive_rows_carry_correct_dimensions_and_agreement_is_float():
    ch = _client_with_uniform_rows()
    gt_by_vid = {"uav0000013_01073_v": [(0.0, 2.32)]}
    rows = adaptive_mrl.run_query(ch, "visdrone", "otobüsü göster", gt_by_vid,
                                  use_filters=True, merge_gap_tol=10.0)
    a256 = next(r for r in rows
               if r["strategy"] == "adaptive_mrl_256_to_2048" and r["candidate_k"] == 50)
    assert a256["stage1_dimension"] == 256
    assert a256["rerank_dimension"] == 2048
    assert isinstance(a256["agreement_vs_2048_exact_at10"], float)
    assert 0.0 <= a256["agreement_vs_2048_exact_at10"] <= 1.0


def test_underfilled_true_when_returned_less_than_final_k():
    rows = [("uav0000013_01073_v", 0.0, 2.32, 0.1)]  # sadece 1 sonuc, final_k=10
    rows_by_table = {f"clips_qwen3vl_emb_{d}": rows for d in (256, 512, 2048)}
    ch = _FakeClient(rows_by_table, rerank_rows=rows)
    gt_by_vid = {"uav0000013_01073_v": [(0.0, 2.32)]}
    result_rows = adaptive_mrl.run_query(ch, "visdrone", "otobüsü göster", gt_by_vid,
                                         use_filters=True, merge_gap_tol=10.0)
    assert all(r["underfilled"] for r in result_rows)


def test_empty_candidates_skip_rerank_query_and_mark_underfilled():
    rows_by_table = {f"clips_qwen3vl_emb_{d}": [] for d in (256, 512, 2048)}
    ch = _FakeClient(rows_by_table, rerank_rows=[])
    gt_by_vid = {}
    rows = adaptive_mrl.run_query(ch, "visdrone", "tren göster", gt_by_vid,
                                  use_filters=True, merge_gap_tol=10.0)
    adaptive_rows = [r for r in rows if r["candidate_k"] is not None]
    assert all(r["candidate_count"] == 0 for r in adaptive_rows)
    assert all(r["underfilled"] for r in adaptive_rows)
    assert all(r["sql"]["rerank"] is None for r in adaptive_rows)


def test_run_adaptive_mrl_bench_end_to_end_real_visdrone_gt_fake_clickhouse():
    ch = _client_with_uniform_rows()
    result = adaptive_mrl.run_adaptive_mrl_bench(
        subset_sequences=["uav0000013_01073_v"], client=ch)
    assert result["dataset_id"] == "visdrone"
    assert result["n_queries"] == 28  # gercek build_queries()
    assert result["n_sequences"] == 1
    assert len(result["rows"]) == 28 * 14
    assert result["pilot_warning"] is not None
    assert "28" in result["pilot_warning"]
    assert "bağlayıcı" in result["pilot_warning"]


def test_pilot_warning_absent_when_threshold_reached(monkeypatch):
    # HEM adaptive_mrl'in HEM datasets.visdrone'un kendi build_queries
    # bagi bagimsiz isim baglamalari (from X import Y) - ikisi de sahtelenmeli
    # yoksa adapter.ground_truth() gercek 28 sorguyla KeyError atar.
    fake_queries = {f"q{i}": (lambda F, N, fps: []) for i in range(150)}
    monkeypatch.setattr(adaptive_mrl, "build_queries", lambda: fake_queries)
    monkeypatch.setattr(visdrone_module, "build_queries", lambda: fake_queries)
    ch = _client_with_uniform_rows()
    result = adaptive_mrl.run_adaptive_mrl_bench(subset_sequences=["uav0000013_01073_v"], client=ch)
    assert result["n_queries"] == 150
    assert result["pilot_warning"] is None
