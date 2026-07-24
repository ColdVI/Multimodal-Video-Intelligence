"""Segment-bazli retrieval metrikleri: temporal IoU + precision/recall@k."""


def t_iou(a, b) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def evaluate(pred, gt, k: int = 10, iou_thr: float = 0.5) -> dict:
    """pred: (video_id, t0, t1, score) skora gore azalan sirali.
    gt: {video_id: [(t0, t1), ...]}"""
    gt_flat = [(vid, iv) for vid, ivs in gt.items() for iv in ivs]
    matched = set()
    hits = 0
    top = pred[:k]
    for vid, t0, t1, _ in top:
        for gi, (gvid, giv) in enumerate(gt_flat):
            if gi in matched or gvid != vid:
                continue
            if t_iou((t0, t1), giv) >= iou_thr:
                matched.add(gi)
                hits += 1
                break
    n_pred = len(top)
    return {
        "precision@k": hits / n_pred if n_pred else 0.0,
        "recall@k": len(matched) / len(gt_flat) if gt_flat else 0.0,
        "n_gt": len(gt_flat),
        "n_pred": n_pred,
        "n_hits": hits,
    }
