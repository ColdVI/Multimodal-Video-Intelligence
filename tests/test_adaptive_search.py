import numpy as np
import pytest

from search import adaptive


class _FakeResult:
    def __init__(self, rows):
        self.result_rows = rows


class _FakeClient:
    """Cagri sirasina gore stage1 sonra rerank donen sahte ClickHouse
    client'i - sql metnini de kaydeder ki 'rerank SADECE adaylari
    kullaniyor mu' dogrulanabilsin."""
    def __init__(self, stage1_rows, rerank_rows):
        self.stage1_rows = stage1_rows
        self.rerank_rows = rerank_rows
        self.queries = []

    def query(self, sql):
        self.queries.append(sql)
        if len(self.queries) == 1:
            return _FakeResult(self.stage1_rows)
        return _FakeResult(self.rerank_rows)


class _FakeEmbedder:
    dim = 2048

    def embed_text(self, text):
        return np.array([1.0, 0.0] + [0.0] * 2046, dtype=np.float32)


@pytest.fixture(autouse=True)
def _mock_embedder(monkeypatch):
    monkeypatch.setattr(adaptive, "get_embedder", lambda name: _FakeEmbedder())


def test_build_rerank_sql_only_references_candidate_tuples():
    candidates = [("vidA", 0.0, 8.0, 0.1), ("vidB", 4.0, 12.0, 0.2)]
    sql = adaptive._build_rerank_sql("clips_qwen3vl_emb_2048", candidates, np.zeros(2048), 10)
    assert "clips_qwen3vl_emb_2048" in sql
    assert "('vidA',0.000000)" in sql
    assert "('vidB',4.000000)" in sql
    assert "LIMIT 10" in sql


def test_adaptive_search_rejects_equal_dims():
    with pytest.raises(ValueError):
        adaptive.adaptive_search("otobüsü göster", stage1_dim=256, rerank_dim=256,
                                 candidate_k=50, client=_FakeClient([], []))


def test_adaptive_search_two_stage_flow_and_field_shape():
    stage1_rows = [("v1", 0.0, 8.0, 0.05), ("v2", 0.0, 8.0, 0.1)]
    rerank_rows = [("v1", 0.0, 8.0, 0.02)]  # rerank sadece 1 sonuc dondurdu
    client = _FakeClient(stage1_rows, rerank_rows)

    result = adaptive.adaptive_search("otobüsü göster", stage1_dim=256, rerank_dim=2048,
                                      candidate_k=50, final_k=10, client=client)

    assert result["candidate_count"] == 2
    assert result["returned_count"] == 1
    assert result["underfilled"] is True  # 1 < final_k(10)
    assert result["rows"] == rerank_rows
    assert len(client.queries) == 2
    # stage1 kucuk tabloyu, rerank buyuk tabloyu hedeflemeli
    assert "clips_qwen3vl_emb_256" in client.queries[0]
    assert "clips_qwen3vl_emb_2048" in client.queries[1]
    # rerank SADECE stage1'in dondurdugu adaylari referans almali (tum tablo taramasi degil)
    assert "'v1'" in client.queries[1] and "'v2'" in client.queries[1]
    assert "WHERE (video_id, t_start) IN" in client.queries[1]


def test_adaptive_search_not_underfilled_when_final_k_met():
    stage1_rows = [(f"v{i}", 0.0, 8.0, 0.01 * i) for i in range(20)]
    rerank_rows = [(f"v{i}", 0.0, 8.0, 0.01 * i) for i in range(10)]
    client = _FakeClient(stage1_rows, rerank_rows)
    result = adaptive.adaptive_search("kamyonu göster", stage1_dim=512, rerank_dim=2048,
                                      candidate_k=20, final_k=10, client=client)
    assert result["underfilled"] is False
    assert result["returned_count"] == 10


def test_adaptive_search_empty_candidates_short_circuits_without_rerank_query():
    client = _FakeClient([], [])
    result = adaptive.adaptive_search("tren göster", stage1_dim=256, rerank_dim=2048,
                                      candidate_k=50, final_k=10, client=client)
    assert result["candidate_count"] == 0
    assert result["returned_count"] == 0
    assert result["underfilled"] is True
    assert result["rerank_sql"] is None
    assert len(client.queries) == 1  # rerank hic cagrilmadi


def test_agreement_at_k_full_overlap():
    rows = [("v1", 0.0, 8.0, 0.1), ("v2", 0.0, 8.0, 0.2)]
    assert adaptive.agreement_at_k(rows, rows, k=10) == 1.0


def test_agreement_at_k_partial_overlap():
    exact = [("v1", 0.0, 8.0, 0.1), ("v2", 0.0, 8.0, 0.2)]
    adaptive_rows = [("v1", 0.0, 8.0, 0.1), ("v3", 0.0, 8.0, 0.3)]
    assert adaptive.agreement_at_k(adaptive_rows, exact, k=10) == 0.5


def test_agreement_at_k_empty_exact_rows_is_zero_not_misleadingly_perfect():
    assert adaptive.agreement_at_k([("v1", 0.0, 8.0, 0.1)], [], k=10) == 0.0
