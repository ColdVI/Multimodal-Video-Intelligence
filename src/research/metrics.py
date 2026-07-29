"""Corpus-duzeyi (whole-clip) caption<->video retrieval metrikleri: CapERA/
MSR-VTT gibi 'GT hazir' datasetler icin (VisDrone'un interval/temporal-IoU
metrikleri bench/metrics.py::evaluate_multi_k'de kalir - farkli GT sekli,
farkli fonksiyon). scripts/validate_msrvtt.py::compute_retrieval_metrics
KARE (1 caption <-> 1 video, index hizali) varsayimini kullanir; CapERA'da
video basina 5 caption oldugu icin (KARE OLMAYAN GT) buradaki
corpus_retrieval_metrics() her sorgu icin bir GT-index-KUMESI kabul eder.

paired_bootstrap_ci burada YENIDEN YAZILMAZ - bench/stats.py tek kaynak,
bu modul onu re-export eder (spec SS3.3/SS7.1 "paired bootstrap %95 CI")."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from bench.stats import paired_bootstrap_ci  # noqa: E402,F401


def corpus_retrieval_metrics(sim_matrix: np.ndarray, gt_indices: list, ks=(1, 5, 10)) -> dict:
    """sim_matrix[i,j] = sorgu i ile corpus item j arasi cosine benzerlik.
    gt_indices[i]: sorgu i icin dogru corpus index'lerinin kumesi/listesi
    (CapERA: video basina >=1 caption oldugu icin corpus->query yonunde
    tek video birden fazla sorgunun GT'si olabilir; bu fonksiyon sorgu
    yonunde çalışır, cagiran taraf gt_indices'i hazirlar).
    Donus: R@1/R@5/R@10 (yuzde), MRR, MeanRank, MedianRank, nDCG@10, MAP, n."""
    n = sim_matrix.shape[0]
    ranks = np.zeros(n)
    reciprocal_ranks = np.zeros(n)
    ndcgs = np.zeros(n)
    aps = np.zeros(n)
    for i in range(n):
        order = np.argsort(-sim_matrix[i], kind="stable")
        gt_set = set(int(x) for x in gt_indices[i])
        if not gt_set:
            ranks[i] = np.nan
            continue
        hits = [1 if int(idx) in gt_set else 0 for idx in order]
        first_hit_rank = next((r + 1 for r, h in enumerate(hits) if h), None)
        ranks[i] = first_hit_rank if first_hit_rank is not None else len(order) + 1
        reciprocal_ranks[i] = 1.0 / ranks[i]

        k10 = min(10, len(hits))
        dcg = sum(1.0 / np.log2(r + 2) for r in range(k10) if hits[r])
        idcg = sum(1.0 / np.log2(r + 2) for r in range(min(k10, len(gt_set))))
        ndcgs[i] = dcg / idcg if idcg > 0 else 0.0

        running_hits, precision_sum = 0, 0.0
        for rank, h in enumerate(hits, start=1):
            if h:
                running_hits += 1
                precision_sum += running_hits / rank
        aps[i] = precision_sum / len(gt_set) if gt_set else 0.0

    valid = ~np.isnan(ranks)
    valid_ranks = ranks[valid]
    out = {
        "n": n,
        "n_valid": int(valid.sum()),
    }
    for k in ks:
        out[f"R@{k}"] = float(np.mean(valid_ranks <= k) * 100) if len(valid_ranks) else 0.0
    out["MRR"] = float(np.mean(reciprocal_ranks[valid])) if valid.any() else 0.0
    out["MeanRank"] = float(np.mean(valid_ranks)) if len(valid_ranks) else float("nan")
    out["MedianRank"] = float(np.median(valid_ranks)) if len(valid_ranks) else float("nan")
    out["nDCG@10"] = float(np.mean(ndcgs[valid])) if valid.any() else 0.0
    out["MAP"] = float(np.mean(aps[valid])) if valid.any() else 0.0
    return out


__all__ = ["corpus_retrieval_metrics", "paired_bootstrap_ci"]
