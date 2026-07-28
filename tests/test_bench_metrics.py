import math

from bench.metrics import average_precision, evaluate_multi_k, ndcg_at_k, reciprocal_rank


def test_reciprocal_rank_first_hit():
    pred = [("v1", 0.0, 10.0, 0.9), ("v1", 20.0, 30.0, 0.5)]
    gt = {"v1": [(0.0, 10.0)]}
    assert reciprocal_rank(pred, gt) == 1.0


def test_reciprocal_rank_second_hit():
    pred = [("v1", 100.0, 110.0, 0.9), ("v1", 0.0, 10.0, 0.5)]
    gt = {"v1": [(0.0, 10.0)]}
    assert reciprocal_rank(pred, gt) == 0.5


def test_reciprocal_rank_no_hit():
    pred = [("v1", 100.0, 110.0, 0.9)]
    gt = {"v1": [(0.0, 10.0)]}
    assert reciprocal_rank(pred, gt) == 0.0


def test_reciprocal_rank_empty_gt():
    pred = [("v1", 0.0, 10.0, 0.9)]
    assert reciprocal_rank(pred, {}) == 0.0


def test_evaluate_multi_k_has_all_ks_and_mrr():
    pred = [("v1", 0.0, 10.0, 0.9)]
    gt = {"v1": [(0.0, 10.0)]}
    out = evaluate_multi_k(pred, gt)
    assert set(out["by_k"]) == {1, 5, 10}
    assert out["n_gt"] == 1
    assert out["mrr"] == 1.0
    assert out["ndcg@10"] == 1.0
    assert out["map"] == 1.0


def test_evaluate_multi_k_empty_gt_is_zero_not_error():
    pred = [("v1", 0.0, 10.0, 0.9)]
    out = evaluate_multi_k(pred, {})
    assert out["n_gt"] == 0
    assert out["by_k"][10]["recall@k"] == 0.0
    assert out["ndcg@10"] == 0.0
    assert out["map"] == 0.0


def test_ndcg_at_k_perfect_ranking_is_one():
    pred = [("v1", 0.0, 10.0, 0.9), ("v1", 20.0, 30.0, 0.8)]
    gt = {"v1": [(0.0, 10.0), (20.0, 30.0)]}
    assert ndcg_at_k(pred, gt, k=10) == 1.0


def test_ndcg_at_k_partial_hits_at_ranks_1_and_3():
    # rank1 isabet, rank2 kacirilmis, rank3 isabet - n_gt=3 (bir GT hic
    # dondurulmemis, o da hesaba katiliyor - idcg 3 pozisyon uzerinden)
    pred = [
        ("v1", 0.0, 10.0, 0.9),    # isabet (rank 1)
        ("v1", 100.0, 110.0, 0.8),  # kacirilmis (rank 2)
        ("v1", 20.0, 30.0, 0.7),   # isabet (rank 3)
    ]
    gt = {"v1": [(0.0, 10.0), (20.0, 30.0), (200.0, 210.0)]}  # 3. GT hic donmuyor
    dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)  # rank1 (i=0) + rank3 (i=2)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3) + 1.0 / math.log2(4)  # min(k,n_gt)=3
    expected = dcg / idcg
    assert abs(ndcg_at_k(pred, gt, k=10) - expected) < 1e-9


def test_ndcg_at_k_only_considers_top_k():
    # tek isabet rank 11'de, k=10 disinda kaliyor -> ndcg@10 = 0
    pred = [("v1", float(i), float(i) + 1, 1.0 - i * 0.01) for i in range(10)]
    pred.append(("v1", 500.0, 510.0, 0.05))  # rank 11, gercek isabet
    gt = {"v1": [(500.0, 510.0)]}
    assert ndcg_at_k(pred, gt, k=10) == 0.0


def test_average_precision_all_hits_is_one():
    pred = [("v1", 0.0, 10.0, 0.9), ("v1", 20.0, 30.0, 0.8)]
    gt = {"v1": [(0.0, 10.0), (20.0, 30.0)]}
    assert average_precision(pred, gt) == 1.0


def test_average_precision_hits_at_ranks_1_and_3():
    pred = [
        ("v1", 0.0, 10.0, 0.9),     # isabet, precision@1 = 1/1
        ("v1", 100.0, 110.0, 0.8),   # kacirilmis
        ("v1", 20.0, 30.0, 0.7),    # isabet, precision@3 = 2/3
    ]
    gt = {"v1": [(0.0, 10.0), (20.0, 30.0)]}  # n_gt=2, ikisi de bulundu
    expected = (1 / 1 + 2 / 3) / 2
    assert abs(average_precision(pred, gt) - expected) < 1e-9


def test_average_precision_empty_gt_is_zero():
    pred = [("v1", 0.0, 10.0, 0.9)]
    assert average_precision(pred, {}) == 0.0
