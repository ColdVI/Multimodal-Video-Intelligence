from bench.metrics import evaluate_multi_k, reciprocal_rank


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


def test_evaluate_multi_k_empty_gt_is_zero_not_error():
    pred = [("v1", 0.0, 10.0, 0.9)]
    out = evaluate_multi_k(pred, {})
    assert out["n_gt"] == 0
    assert out["by_k"][10]["recall@k"] == 0.0
