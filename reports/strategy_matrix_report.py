"""Faz 2: ClickHouse arama katmani davranis dogrulamasi.

- Strateji matrisi: dort strateji (bruteforce/hnsw/prefilter/postfilter_rescore)
  x iki filtre secicilik (loose/strict) x iki tablo. Her hucrede p50/p95
  latency (N_REPEATS tekrar, N_WARMUP isinma sonrasi), EXPLAIN indexes=1'de
  vector index kullanimi, rows_read/bytes_read.
- Ayar deneyleri: vector_search_index_fetch_multiplier ve
  hnsw_candidate_list_size_for_search sweep'leri.
- HNSW recall@K: exact brute-force top-K ground truth kabul edilip HNSW
  top-K ile kesisim olculur.

HTTP yardimcilarini (_post/_query_json/_explain) reports/clickhouse_search_report.py
ile paylasir - ayni ClickHouse HTTP sozlesmesi tek yerde kalir."""
from __future__ import annotations

import re
import statistics
import time

from reports.clickhouse_search_report import _explain, _post, _query_json
from search.sql_catalog import QUERY_SPECS, assert_read_only_sql, get_query_spec

TABLES = ("clips_xclip_hf_zeroshot", "clips_siglip2_frameavg")
N_REPEATS = 50
N_WARMUP = 3
SWEEP_N_REPEATS = 20
SWEEP_N_WARMUP = 2


def _percentile(values: list, pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _run_repeated(endpoint: str, sql: str, n_repeats: int = N_REPEATS,
                  n_warmup: int = N_WARMUP) -> dict:
    assert_read_only_sql(sql)
    for _ in range(n_warmup):
        _query_json(endpoint, sql)
    latencies_ms = []
    last_response = None
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        last_response = _query_json(endpoint, sql)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    stats = (last_response or {}).get("statistics", {})
    return {
        "n_repeats": n_repeats,
        "p50_ms": round(_percentile(latencies_ms, 50), 3),
        "p95_ms": round(_percentile(latencies_ms, 95), 3),
        "mean_ms": round(statistics.mean(latencies_ms), 3),
        "row_count": len((last_response or {}).get("data", [])),
        "rows_read": stats.get("rows_read"),
        "bytes_read": stats.get("bytes_read"),
    }


def strategy_matrix(endpoint: str = "http://localhost:8123/") -> list[dict]:
    matrix_specs = [s for s in QUERY_SPECS if s.kind == "matrix"]
    results = []
    for spec in matrix_specs:
        for table in TABLES:
            sql = spec.sql_for_table(table)
            timing = _run_repeated(endpoint, sql)
            raw_plan = _explain(endpoint, sql)
            results.append({
                "query_id": spec.query_id,
                "strategy": spec.strategy,
                "selectivity": spec.selectivity,
                "table": table,
                "sql": sql,
                "vector_index_in_plan": "Description: vector_similarity" in raw_plan,
                **timing,
            })
    return results


def fetch_multiplier_sweep(endpoint: str = "http://localhost:8123/",
                           table: str = "clips_xclip_hf_zeroshot",
                           values=(1.0, 3.0, 10.0)) -> list[dict]:
    """Siki filtrede LIMIT-alti donme oranini fetch_multiplier'a gore olcer;
    bu ayar yalnizca postfiltering+rescoring icin gecerli (bkz. ClickHouse
    system.settings aciklamasi)."""
    base_sql = get_query_spec("matrix_postfilter_strict").sql_for_table(table)
    results = []
    for value in values:
        sql = re.sub(
            r"vector_search_index_fetch_multiplier = \d+(\.\d+)?",
            f"vector_search_index_fetch_multiplier = {value}", base_sql)
        timing = _run_repeated(endpoint, sql, n_repeats=SWEEP_N_REPEATS, n_warmup=SWEEP_N_WARMUP)
        results.append({"fetch_multiplier": value, "table": table, **timing})
    return results


def ef_search_sweep(endpoint: str = "http://localhost:8123/",
                    table: str = "clips_xclip_hf_zeroshot",
                    values=(64, 256, 512)) -> list[dict]:
    """hnsw_candidate_list_size_for_search (ef_search) icin recall/latency
    egrisi - siki filtreli HNSW sorgusu uzerinde."""
    base_sql = get_query_spec("matrix_hnsw_strict").sql_for_table(table)
    results = []
    for value in values:
        sql = base_sql.rstrip() + f"\n, hnsw_candidate_list_size_for_search = {value}"
        timing = _run_repeated(endpoint, sql, n_repeats=SWEEP_N_REPEATS, n_warmup=SWEEP_N_WARMUP)
        results.append({"ef_search": value, "table": table, **timing})
    return results


def hnsw_recall_at_k(endpoint: str = "http://localhost:8123/",
                     table: str = "clips_xclip_hf_zeroshot", k: int = 10) -> dict:
    """Exact brute-force top-K ground truth kabul edilir; HNSW top-K ile
    kesisim = recall@K. Loose (filtresize yakin) secicilik kullanilir ki
    filtre kaynakli aday kaybi recall dususuyle karismasin."""
    exact_sql = get_query_spec("matrix_bruteforce_loose").sql_for_table(table)
    hnsw_sql = get_query_spec("matrix_hnsw_loose").sql_for_table(table)
    exact_rows = _query_json(endpoint, exact_sql)["data"]
    hnsw_rows = _query_json(endpoint, hnsw_sql)["data"]
    exact_keys = {(r["video_id"], r["t_start"]) for r in exact_rows[:k]}
    hnsw_keys = {(r["video_id"], r["t_start"]) for r in hnsw_rows[:k]}
    overlap = exact_keys & hnsw_keys
    return {
        "table": table, "k": k,
        "recall_at_k": (len(overlap) / len(exact_keys)) if exact_keys else 0.0,
        "n_exact": len(exact_keys), "n_hnsw": len(hnsw_keys), "n_overlap": len(overlap),
    }


def collect_strategy_evidence(endpoint: str = "http://localhost:8123/") -> dict:
    matrix = strategy_matrix(endpoint)
    fetch_sweep = [row for table in TABLES
                   for row in fetch_multiplier_sweep(endpoint, table=table)]
    ef_sweep = [row for table in TABLES for row in ef_search_sweep(endpoint, table=table)]
    recall = [hnsw_recall_at_k(endpoint, table=table) for table in TABLES]
    return {
        "matrix": matrix,
        "fetch_multiplier_sweep": fetch_sweep,
        "ef_search_sweep": ef_sweep,
        "hnsw_recall_at_10": recall,
    }
