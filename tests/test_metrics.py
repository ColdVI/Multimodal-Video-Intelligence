from eval.metrics import evaluate, t_iou


def test_tiou_identical():
    assert t_iou((0, 10), (0, 10)) == 1.0


def test_tiou_no_overlap():
    assert t_iou((0, 10), (20, 30)) == 0.0


def test_tiou_partial():
    assert abs(t_iou((0, 10), (5, 15)) - (5 / 15)) < 1e-9


def test_evaluate_perfect_match():
    pred = [("v1", 0.0, 10.0, 0.9)]
    gt = {"v1": [(0.0, 10.0)]}
    m = evaluate(pred, gt, k=10, iou_thr=0.5)
    assert m["precision@k"] == 1.0
    assert m["recall@k"] == 1.0


def test_evaluate_wrong_video_no_match():
    pred = [("v2", 0.0, 10.0, 0.9)]
    gt = {"v1": [(0.0, 10.0)]}
    m = evaluate(pred, gt, k=10, iou_thr=0.5)
    assert m["precision@k"] == 0.0
    assert m["recall@k"] == 0.0


def test_evaluate_below_iou_threshold_no_match():
    pred = [("v1", 0.0, 3.0, 0.9)]
    gt = {"v1": [(0.0, 10.0)]}
    m = evaluate(pred, gt, k=10, iou_thr=0.5)
    assert m["precision@k"] == 0.0


def test_evaluate_respects_k():
    pred = [("v1", 0.0, 10.0, 0.9), ("v1", 20.0, 30.0, 0.5)]
    gt = {"v1": [(0.0, 10.0)]}
    m = evaluate(pred, gt, k=1, iou_thr=0.5)
    assert m["n_pred"] == 1
    assert m["precision@k"] == 1.0


def test_evaluate_each_gt_matched_once():
    pred = [("v1", 0.0, 10.0, 0.9), ("v1", 0.5, 10.5, 0.8)]
    gt = {"v1": [(0.0, 10.0)]}
    m = evaluate(pred, gt, k=10, iou_thr=0.5)
    assert m["n_hits"] == 1
