"""Adaptif Matryoshka (MRL) arama: kucuk boyutlu tabloda aday uret, sadece
o adaylari tam boyutlu (2048d) exact cosine ile yeniden skorla. Mevcut
2048d embedding'leri YENIDEN URETMEZ - qwen3vl_emb_{stage1,rerank} tabloları
zaten dolu olmali (bkz. scripts/mrl_truncate_embeddings.py).

Neden iki asama: kucuk boyutta HNSW/exact arama ucuz ama biraz kayipli;
adaylari (candidate_k, top_k'dan buyuk) tam boyutta yeniden skorlamak,
tum tabloyu 2048d'de taramadan kalite kaybinin cogunu geri kazandirir.
2048d rerank SADECE stage1 adaylarini kullanir - tum tabloyu TARAMAZ."""
import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_config
from models import get_embedder
from scripts.mrl_truncate_embeddings import truncate_and_renormalize

from .parser import parse
from .query import _build_sql, _fmt_vector, _get_client


def _build_rerank_sql(table: str, candidates: list, qvec, final_k: int) -> str:
    """SADECE candidates listesindeki (video_id, t_start) ciftlerini
    yeniden skorlar - tum tabloyu taramaz. ClickHouse tuple IN sozdizimi."""
    tuples = ",".join(f"('{vid}',{t0:.6f})" for vid, t0, _t1, _d in candidates)
    return f"""
        SELECT video_id, t_start, t_end,
               cosineDistance(embedding, {_fmt_vector(qvec)}) AS dist
        FROM {table}
        WHERE (video_id, t_start) IN ({tuples})
        ORDER BY dist ASC
        LIMIT {final_k}
    """


def adaptive_search(q: str, stage1_dim: int, rerank_dim: int, candidate_k: int,
                    final_k: int = 10, use_filters: bool = True,
                    strategy: str = "auto", client=None) -> dict:
    """Donus (bench harness bunu satir alanlarina eslestirir):
    {
      "rows": [(video_id, t_start, t_end, dist), ...] rerank skoruna gore,
      "candidate_count": int, "returned_count": int, "underfilled": bool,
      "stage1_latency_s": float, "rerank_latency_s": float,
      "stage1_sql": str, "rerank_sql": str,
    }
    underfilled = returned_count < final_k (Codex'in FAIL kriteri - "hizli
    ama eksik" bir sonuc basari sayilmaz, bkz. unified_search_harness_
    duzeltmeler.md giris paragrafi)."""
    if stage1_dim == rerank_dim:
        raise ValueError("adaptive_search iki FARKLI boyut gerektirir "
                         f"(stage1_dim == rerank_dim == {stage1_dim})")
    cfg = load_config()
    ch = client if client is not None else _get_client(cfg)
    p = parse(q)

    rerank_emb = get_embedder(f"qwen3vl_emb_{rerank_dim}")
    full_vec = rerank_emb.embed_text(p.semantic)
    stage1_vec = truncate_and_renormalize(full_vec, stage1_dim)

    where = "1"
    if use_filters and p.filters:
        where = " AND ".join(f"{c} {op} {v}" for c, op, v in p.filters)

    stage1_table = f"clips_qwen3vl_emb_{stage1_dim}"
    stage1_sql = _build_sql(where, stage1_table, stage1_vec, candidate_k, strategy)
    t0 = time.perf_counter()
    candidates = ch.query(stage1_sql).result_rows
    stage1_latency_s = time.perf_counter() - t0

    rerank_table = f"clips_qwen3vl_emb_{rerank_dim}"
    if not candidates:
        return {
            "rows": [], "candidate_count": 0, "returned_count": 0, "underfilled": True,
            "stage1_latency_s": stage1_latency_s, "rerank_latency_s": 0.0,
            "stage1_sql": stage1_sql, "rerank_sql": None,
        }
    rerank_sql = _build_rerank_sql(rerank_table, candidates, full_vec, final_k)
    t0 = time.perf_counter()
    rows = ch.query(rerank_sql).result_rows
    rerank_latency_s = time.perf_counter() - t0

    return {
        "rows": rows,
        "candidate_count": len(candidates),
        "returned_count": len(rows),
        "underfilled": len(rows) < final_k,
        "stage1_latency_s": stage1_latency_s,
        "rerank_latency_s": rerank_latency_s,
        "stage1_sql": stage1_sql,
        "rerank_sql": rerank_sql,
    }


def agreement_at_k(adaptive_rows: list, exact_rows: list, k: int = 10) -> float:
    """adaptive_rows'un exact_rows'a (ayni sorgu, saf 2048d exact arama)
    top-k'da ne kadar ORTUSTUGU - KALITE degil UYUM olcer (bkz.
    unified_search_harness_duzeltmeler.md D2). exact_rows bos ise (sorgu
    hic sonuc uretmemis) 0.0 - hicbir sey uzerinde 'tam uyum' yaniltici olur."""
    if not exact_rows:
        return 0.0
    exact_keys = {(vid, t0) for vid, t0, _t1, _d in exact_rows[:k]}
    adaptive_keys = {(vid, t0) for vid, t0, _t1, _d in adaptive_rows[:k]}
    return len(exact_keys & adaptive_keys) / len(exact_keys)


__all__ = ["adaptive_search", "agreement_at_k"]
