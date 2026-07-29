import numpy as np
import pytest

from app.search.reference import common_exact, quality_fields
from app.search.strategies import STRATEGIES, clickhouse_limit_settings, validate


def corpus():
    rng=np.random.default_rng(42); matrix=rng.standard_normal((200,32)).astype(np.float32); matrix/=np.linalg.norm(matrix,axis=1,keepdims=True)
    query=rng.standard_normal(32).astype(np.float32); query/=np.linalg.norm(query)
    return matrix,query,[f"item-{i:03d}" for i in range(200)]


@pytest.mark.parametrize("backend,strategy",[(b,s) for b,values in STRATEGIES.items() if b!="milvus" for s in values])
def test_backend_strategy_contract_on_200_item_corpus(backend,strategy):
    validate(backend,strategy); matrix,query,ids=corpus(); result=common_exact(matrix,query,ids,10)
    assert len(result)==10 and all(np.isfinite(score) for _,score in result)


def test_exact_reference_recall_at_10_is_one():
    matrix,query,ids=corpus(); expected=common_exact(matrix,query,ids,10)
    for backend in ("clickhouse","qdrant","pgvector","numpy_exact"):
        actual=common_exact(matrix,query,ids,10)
        assert len({x[0] for x in actual}&{x[0] for x in expected})/10==1.0


def test_negative_control_is_underfilled_and_quality_is_null():
    matrix,query,ids=corpus(); result=common_exact(matrix,query,ids,10,mask=np.zeros(200,dtype=bool)); quality=quality_fields(0,len(result))
    assert result==[] and quality["returned_count"]==0
    assert quality["r_at_1"] is None and quality["ndcg"] is None and quality["quality_vs_groundtruth"] is None


def test_clickhouse_top_k_200_raises_query_limit_not_an_error():
    assert clickhouse_limit_settings(200)==(200,"max_limit_for_vector_search_queries=200")
    assert clickhouse_limit_settings(10)==(100,"")
