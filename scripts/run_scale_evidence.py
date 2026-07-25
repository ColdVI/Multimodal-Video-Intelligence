"""Faz 2 sentetik olcek kaniti: bench_scale_<dim> tablosunda dort stratejinin
p50/p95'i ve HNSW recall@10'u. build_scale_table.py'nin urettigi tabloyu
okur; uretim clips_* tablolarina dokunmaz."""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import load_config
from reports.strategy_matrix_report import (
    N_REPEATS, N_WARMUP, _run_repeated, hnsw_recall_at_k,
)
from search.sql_catalog import QUERY_SPECS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, help="ör. bench_scale_512")
    ap.add_argument("--n-repeats", type=int, default=N_REPEATS)
    ap.add_argument("--n-warmup", type=int, default=N_WARMUP)
    args = ap.parse_args()

    cfg = load_config()
    endpoint = f"http://{cfg['clickhouse']['host']}:{cfg['clickhouse']['port']}/"

    matrix_specs = [s for s in QUERY_SPECS if s.kind == "matrix"]
    rows = []
    for spec in matrix_specs:
        sql = spec.sql_for_table(args.table)
        timing = _run_repeated(endpoint, sql, n_repeats=args.n_repeats, n_warmup=args.n_warmup)
        rows.append({
            "strategy": spec.strategy, "selectivity": spec.selectivity,
            "table": args.table, **timing,
        })
        print(f"{spec.strategy:20s} {spec.selectivity:7s} rows={timing['row_count']:3d} "
              f"rows_read={timing['rows_read']!s:8s} p50={timing['p50_ms']:.3f}ms "
              f"p95={timing['p95_ms']:.3f}ms")

    recall = hnsw_recall_at_k(endpoint, table=args.table, k=10)
    print(f"HNSW recall@10 ({args.table}): {recall['recall_at_k']:.3f} "
          f"({recall['n_overlap']}/{recall['n_exact']})")

    out_path = pathlib.Path(f"artifacts/scale_evidence_{args.table}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"matrix": rows, "hnsw_recall_at_10": recall}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"JSON kanit: {out_path}")


if __name__ == "__main__":
    main()
