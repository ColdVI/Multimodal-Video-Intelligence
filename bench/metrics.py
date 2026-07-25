"""eval/metrics.py::evaluate()'i sarar (Faz 1 madde 1): K listesi
(1,5,10), MRR ve sorgu basina n_gt ekler. Temporal-IoU eslestirme mantigini
tekrarlamaz - eval/metrics.py tek kaynak kalir."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.metrics import evaluate, t_iou

DEFAULT_KS = (1, 5, 10)


def reciprocal_rank(pred, gt, iou_thr: float = 0.5) -> float:
    """pred: (video_id,t0,t1,score) skora gore azalan sirali.
    gt: {video_id: [(t0,t1), ...]}. Ilk dogru eslesmenin 1/rank'i; hic
    eslesme veya bos GT icin 0.0."""
    gt_flat = [(vid, iv) for vid, ivs in gt.items() for iv in ivs]
    if not gt_flat:
        return 0.0
    for rank, (vid, t0, t1, _score) in enumerate(pred, start=1):
        for gvid, giv in gt_flat:
            if gvid == vid and t_iou((t0, t1), giv) >= iou_thr:
                return 1.0 / rank
    return 0.0


def evaluate_multi_k(pred, gt, ks=DEFAULT_KS, iou_thr: float = 0.5) -> dict:
    """Donus: {"mrr": float, "n_gt": int, "by_k": {k: eval/metrics.evaluate()
    sozlugu}}."""
    n_gt = sum(len(ivs) for ivs in gt.values())
    by_k = {k: evaluate(pred, gt, k=k, iou_thr=iou_thr) for k in ks}
    return {
        "mrr": reciprocal_rank(pred, gt, iou_thr=iou_thr),
        "n_gt": n_gt,
        "by_k": by_k,
    }
