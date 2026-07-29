"""Referans: common exact (spec SS6). Tum ANN recall'lari TEK bir referansa
karsi olculur: float32, tek thread, stable sort, numpy - hicbir backend'in
kendi 'exact' modu (ClickHouse dahil - bf16 quantization + rescoring +
thread nondeterminizmi tasir) mutlak referans SAYILMAZ."""
import numpy as np


def exact_ranking(corpus_embeddings: np.ndarray, query_embedding: np.ndarray) -> np.ndarray:
    """corpus_embeddings: (n, d) L2-normalized float32. query_embedding: (d,).
    Donus: corpus index'lerinin skora gore azalan sirasi (stable sort -
    esit skorlarda giris sirasi korunur, kosudan kosuya degismez)."""
    E = np.asarray(corpus_embeddings, dtype=np.float32)
    q = np.asarray(query_embedding, dtype=np.float32)
    scores = E @ q
    order = np.argsort(-scores, kind="stable")
    return order


def recall_at_k_vs_exact(backend_top_k_ids: list, exact_order: np.ndarray, k: int) -> float:
    """backend_top_k_ids: backend'in dondurdugu corpus index listesi (skora
    gore azalan). exact_order ile ilk k'daki kesisim orani."""
    if k <= 0:
        return 0.0
    exact_top_k = set(int(i) for i in exact_order[:k])
    backend_top_k = set(int(i) for i in backend_top_k_ids[:k])
    if not exact_top_k:
        return 0.0
    return len(exact_top_k & backend_top_k) / len(exact_top_k)


def topk_agreement(a_ids: list, b_ids: list, k: int) -> float:
    """Jaccard benzerligi: iki top-k kumesinin uzerinden anlasma orani."""
    a = set(int(i) for i in a_ids[:k])
    b = set(int(i) for i in b_ids[:k])
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


__all__ = ["exact_ranking", "recall_at_k_vs_exact", "topk_agreement"]
