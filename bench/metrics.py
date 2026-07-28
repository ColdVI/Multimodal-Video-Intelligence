"""eval/metrics.py::evaluate()'i sarar (Faz 1 madde 1): K listesi
(1,5,10), MRR, nDCG@10, MAP ve sorgu basina n_gt ekler. Temporal-IoU
eslestirme mantigini tekrarlamaz - eval/metrics.py tek kaynak kalir.

OKUMA UYARISI (dis bir analizde bu proje icin gercek veriyle dogrulandi -
bkz. TASKS.md): recall@1/precision@1 COK-DOGRULU (bir sorgunun onlarca
GT penceresi olabildigi) bir kurulumda per-query n_gt tarafindan domine
edilir - n_gt=21 olan bir sorguda max recall@1 = 1/21 ~ 0.048, modelin
kalitesinden bagimsiz bir tavan. MRR ve nDCG@k bu soruna daha dayanikli
(siraya duyarli, tek bir "en iyi ilk sonuc mu" sorusuna indirgeniyor).
Bu projede recall@10/precision@10 de kontrol edilmeli: gercek n_gt
dagilimimizda (config.yaml: bench.subset, 28 sorgu) sorgularin buyuk
cogunlugu n_gt>10 tasiyor (22/24 negatif-kontrol-disi sorgu) - yani
recall@10 GENELDE tavana yapismis DEGIL, aksine cogu sorguda tavanin
(10/n_gt) hayli altinda kaliyor. Bunu varsaymak yerine her yeni sorgu
setinde n_gt dagilimini tekrar kontrol edin (rows[i]["n_gt"])."""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.metrics import evaluate, t_iou

DEFAULT_KS = (1, 5, 10)


def _hit_sequence(pred, gt, iou_thr: float = 0.5) -> list:
    """pred sirasina gore (skora gore azalan), her tahminin ONCEDEN
    eslesmemis bir GT araligini (ayni video + IoU>=thr) isabet ettirip
    ettirmedigini dondurur - greedy bire-bir, eval/metrics.py::evaluate()
    ile AYNI eslestirme kurali (bkz. o modulun docstring'i). MRR/nDCG/MAP
    hepsi bu TEK diziyi tuketir, kendi eslestirme mantiklarini tekrar
    yazmazlar."""
    gt_flat = [(vid, iv) for vid, ivs in gt.items() for iv in ivs]
    matched = set()
    hits = []
    for vid, t0, t1, _score in pred:
        hit = False
        for gi, (gvid, giv) in enumerate(gt_flat):
            if gi in matched or gvid != vid:
                continue
            if t_iou((t0, t1), giv) >= iou_thr:
                matched.add(gi)
                hit = True
                break
        hits.append(hit)
    return hits


def reciprocal_rank(pred, gt, iou_thr: float = 0.5) -> float:
    """pred: (video_id,t0,t1,score) skora gore azalan sirali.
    gt: {video_id: [(t0,t1), ...]}. Ilk dogru eslesmenin 1/rank'i; hic
    eslesme veya bos GT icin 0.0."""
    if not any(iv for ivs in gt.values() for iv in ivs):
        return 0.0
    for rank, hit in enumerate(_hit_sequence(pred, gt, iou_thr=iou_thr), start=1):
        if hit:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(pred, gt, k: int = 10, iou_thr: float = 0.5) -> float:
    """Ikili alaka (isabet/degil) uzerinden nDCG@k. recall@k'nin aksine
    n_gt tarafindan tavana vurmuyor - sadece "ilk k sonuc ne kadar iyi
    siralanmis" sorusuna cevap veriyor, kac tane KACIRILDIGINA degil."""
    n_gt = sum(len(ivs) for ivs in gt.values())
    if n_gt == 0:
        return 0.0
    hits = _hit_sequence(pred, gt, iou_thr=iou_thr)[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, hit in enumerate(hits) if hit)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, n_gt)))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(pred, gt, iou_thr: float = 0.5) -> float:
    """MAP'in tek-sorguluk bileseni: isabet edilen her rank'teki
    precision@rank'in n_gt'ye bolunmus ortalamasi. n_gt=0 icin 0.0."""
    n_gt = sum(len(ivs) for ivs in gt.values())
    if n_gt == 0:
        return 0.0
    hits = _hit_sequence(pred, gt, iou_thr=iou_thr)
    running_hits = 0
    precision_sum = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            running_hits += 1
            precision_sum += running_hits / rank
    return precision_sum / n_gt


def evaluate_multi_k(pred, gt, ks=DEFAULT_KS, iou_thr: float = 0.5) -> dict:
    """Donus: {"mrr": float, "ndcg@10": float, "map": float, "n_gt": int,
    "by_k": {k: eval/metrics.evaluate() sozlugu}}. nDCG@10 sabit k=10
    kullanir (DEFAULT_KS'teki en genis retrieval derinligiyle tutarli)."""
    n_gt = sum(len(ivs) for ivs in gt.values())
    by_k = {k: evaluate(pred, gt, k=k, iou_thr=iou_thr) for k in ks}
    return {
        "mrr": reciprocal_rank(pred, gt, iou_thr=iou_thr),
        "ndcg@10": ndcg_at_k(pred, gt, k=10, iou_thr=iou_thr),
        "map": average_precision(pred, gt, iou_thr=iou_thr),
        "n_gt": n_gt,
        "by_k": by_k,
    }
