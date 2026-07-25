"""Faz 2 strateji matrisi + ayar sweep'leri + HNSW recall@10 kanitini uretir."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import load_config
from reports.strategy_matrix_report import collect_strategy_evidence


def main():
    cfg = load_config()
    endpoint = f"http://{cfg['clickhouse']['host']}:{cfg['clickhouse']['port']}/"
    evidence = collect_strategy_evidence(endpoint)

    out_path = pathlib.Path("artifacts/strategy_matrix_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON kanit: {out_path}")
    print(f"matrix satirlari: {len(evidence['matrix'])}")
    print(f"fetch_multiplier sweep: {len(evidence['fetch_multiplier_sweep'])}")
    print(f"ef_search sweep: {len(evidence['ef_search_sweep'])}")
    for r in evidence["hnsw_recall_at_10"]:
        print(f"HNSW recall@10 ({r['table']}): {r['recall_at_k']:.2f} "
              f"({r['n_overlap']}/{r['n_exact']})")


if __name__ == "__main__":
    main()
