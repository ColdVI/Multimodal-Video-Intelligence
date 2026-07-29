"""Faz 3: Adaptive Matryoshka (MRL) retrieval bench. Kucuk-boyut aday +
2048d rerank stratejilerini (search/adaptive.py) exact/HNSW referanslarina
karsi VisDrone'un 28 sorgusunda kosar.

BAGLAYICILIK UYARISI: 28 sorgu, bu oturumda daha once konulan 150-sorgu
istatistiksel-guc esiginin ALTINDA. Bu modul harness'i TEKNIK VE TANISAL
olarak calistirir (kod dogru mu, underfilled var mi, agreement/quality
ayrimi tutarli mi) - ama ciktisi PILOT/underpowered'dir, kesin bir "X
boyutu Y'den iyi" uretim karari CIKARILAMAZ. Baglayici karar icin MSR-VTT'nin
1000 sorguluk GPU kosumu (Faz 6) beklenir. pilot_warning alani bu yuzden
n_queries'e gore DINAMIK - sabit yazilmadi.

Sorgu metni HER SORGU icin TEK SEFER 2048d embed edilir; 256d/512d bundan
turetilir (truncate_and_renormalize) - modeli ayni sorgu icin tekrar tekrar
cagirmamak icin (bkz. embedding-once-per-query bulgusu)."""
import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bench.metrics import evaluate_multi_k
from common import load_config
from datasets.visdrone import VisDroneAdapter
from eval.make_groundtruth import build_queries, build_query_metadata
from models import get_embedder
from scripts.mrl_truncate_embeddings import truncate_and_renormalize
from search.adaptive import _build_rerank_sql, agreement_at_k
from search.merge import merge_intervals
from search.parser import parse
from search.query import _build_sql, _get_client

CANDIDATE_K_SWEEP = (20, 50, 100, 200)
FINAL_K = 10
STATISTICAL_POWER_THRESHOLD = 150

PILOT_WARNING_TEMPLATE = (
    "Bu sonuç {n} sorguluk pilot değerlendirmedir. "
    "İstatistiksel olarak bağlayıcı üretim kararı değildir."
)

# (strategy_label, dimension, ClickHouse strategy). 'exact' = query_plan_try_use_vector_search=0.
_REFERENCE_SPECS = [
    ("2048d_exact", 2048, "exact"),
    ("2048d_hnsw", 2048, "auto"),
    ("256d_exact", 256, "exact"),
    ("256d_hnsw", 256, "auto"),
    ("512d_exact", 512, "exact"),
    ("512d_hnsw", 512, "auto"),
]
# (strategy_label, stage1_dimension, rerank_dimension)
_ADAPTIVE_SPECS = [
    ("adaptive_mrl_256_to_2048", 256, 2048),
    ("adaptive_mrl_512_to_2048", 512, 2048),
]


def _where_and_semantic(q: str, use_filters: bool) -> tuple:
    p = parse(q)
    where = "1"
    if use_filters and p.filters:
        where = " AND ".join(f"{c} {op} {v}" for c, op, v in p.filters)
    return where, p.semantic


def _query_vectors(semantic: str) -> dict:
    vec_2048 = get_embedder("qwen3vl_emb_2048").embed_text(semantic)
    return {
        2048: vec_2048,
        512: truncate_and_renormalize(vec_2048, 512),
        256: truncate_and_renormalize(vec_2048, 256),
    }


def run_query(ch, dataset_id: str, q: str, gt_by_vid: dict, use_filters: bool,
             merge_gap_tol: float, query_meta: dict = None) -> list:
    """Bir sorgu icin 6 referans + 2 adaptif strateji (x4 candidate_k) = 14
    satir doner."""
    meta = (query_meta or {}).get(q, {})
    where, semantic = _where_and_semantic(q, use_filters)
    vecs = _query_vectors(semantic)

    def base_row(strategy_label):
        return {
            "dataset_id": dataset_id, "query_id": q, "strategy": strategy_label,
            "category": meta.get("category"), "lang": meta.get("lang"),
            "concept": meta.get("concept"), "final_k": FINAL_K,
        }

    rows_out = []
    exact_2048_rows = None
    for strategy_label, dim, ch_strategy in _REFERENCE_SPECS:
        table = f"clips_qwen3vl_emb_{dim}"
        sql = _build_sql(where, table, vecs[dim], FINAL_K, ch_strategy)
        t0 = time.perf_counter()
        raw = ch.query(sql).result_rows
        latency = time.perf_counter() - t0
        if strategy_label == "2048d_exact":
            exact_2048_rows = raw
        pred = merge_intervals(raw, gap_tol=merge_gap_tol)
        metrics = evaluate_multi_k(pred, gt_by_vid)
        rows_out.append({
            **base_row(strategy_label),
            "stage1_dimension": None, "rerank_dimension": dim, "candidate_k": None,
            "candidate_count": len(raw), "returned_count": len(raw),
            "underfilled": len(raw) < FINAL_K,
            "stage1_latency_s": None, "rerank_latency_s": None, "total_latency_s": latency,
            "agreement_vs_2048_exact_at10": None,
            "top_result_ids": [(r[0], r[1]) for r in raw[:FINAL_K]],
            "sql": {"single": sql},
            **metrics,
        })

    for strategy_label, stage1_dim, rerank_dim in _ADAPTIVE_SPECS:
        stage1_table = f"clips_qwen3vl_emb_{stage1_dim}"
        rerank_table = f"clips_qwen3vl_emb_{rerank_dim}"
        for candidate_k in CANDIDATE_K_SWEEP:
            stage1_sql = _build_sql(where, stage1_table, vecs[stage1_dim], candidate_k, "auto")
            t0 = time.perf_counter()
            candidates = ch.query(stage1_sql).result_rows
            stage1_latency = time.perf_counter() - t0

            if candidates:
                rerank_sql = _build_rerank_sql(rerank_table, candidates, vecs[rerank_dim], FINAL_K)
                t0 = time.perf_counter()
                rows = ch.query(rerank_sql).result_rows
                rerank_latency = time.perf_counter() - t0
            else:
                rerank_sql, rows, rerank_latency = None, [], 0.0

            pred = merge_intervals(rows, gap_tol=merge_gap_tol)
            metrics = evaluate_multi_k(pred, gt_by_vid)
            agreement = agreement_at_k(rows, exact_2048_rows, k=FINAL_K)
            rows_out.append({
                **base_row(strategy_label),
                "stage1_dimension": stage1_dim, "rerank_dimension": rerank_dim,
                "candidate_k": candidate_k,
                "candidate_count": len(candidates), "returned_count": len(rows),
                "underfilled": len(rows) < FINAL_K,
                "stage1_latency_s": stage1_latency, "rerank_latency_s": rerank_latency,
                "total_latency_s": stage1_latency + rerank_latency,
                "agreement_vs_2048_exact_at10": agreement,
                "top_result_ids": [(r[0], r[1]) for r in rows[:FINAL_K]],
                "sql": {"stage1": stage1_sql, "rerank": rerank_sql},
                **metrics,
            })
    return rows_out


def run_adaptive_mrl_bench(dataset_id: str = "visdrone", subset_sequences: list = None,
                           use_filters: bool = True, client=None) -> dict:
    cfg = load_config()
    ch = client if client is not None else _get_client(cfg)
    adapter = VisDroneAdapter(cfg)
    queries = build_queries()
    query_meta = build_query_metadata()
    sequences = subset_sequences or adapter.list_sequences()

    gt = {q: {} for q in queries}
    for seq_id in sequences:
        for q, ivs in adapter.ground_truth(seq_id).items():
            gt[q][seq_id] = ivs

    all_rows = []
    for q, gt_by_vid in gt.items():
        all_rows.extend(run_query(ch, dataset_id, q, gt_by_vid, use_filters,
                                  cfg["merge"]["gap_tolerance_s"], query_meta))

    n_queries = len(gt)
    return {
        "dataset_id": dataset_id,
        "n_queries": n_queries,
        "n_sequences": len(sequences),
        "candidate_k_sweep": list(CANDIDATE_K_SWEEP),
        "final_k": FINAL_K,
        "pilot_warning": (PILOT_WARNING_TEMPLATE.format(n=n_queries)
                          if n_queries < STATISTICAL_POWER_THRESHOLD else None),
        "rows": all_rows,
    }


__all__ = ["run_adaptive_mrl_bench", "run_query", "CANDIDATE_K_SWEEP", "FINAL_K",
          "PILOT_WARNING_TEMPLATE", "STATISTICAL_POWER_THRESHOLD"]
