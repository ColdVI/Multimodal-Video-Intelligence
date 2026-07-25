"""Faz 1 madde 1: tum run'lari tek benchmark_report.html + .json icinde
birlestirir. reports/clickhouse_search_report.py'deki deseni izler - ayni
veri hem insan hem test tarafindan okunur."""
import json
import pathlib
from collections import defaultdict

CATEGORY_METRIC_KEYS = ("recall@k", "precision@k")


def aggregate_by_category(rows: list, k: int = 10) -> dict:
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cat = r.get("category") or "?"
        agg[cat]["recall@10"].append(r["by_k"][k]["recall@k"])
        agg[cat]["precision@10"].append(r["by_k"][k]["precision@k"])
        agg[cat]["mrr"].append(r["mrr"])
    return {
        cat: {name: (sum(vals) / len(vals) if vals else 0.0) for name, vals in metrics.items()}
        for cat, metrics in agg.items()
    }


def _row_html(r: dict) -> str:
    by10 = r["by_k"][10]
    # n_gt=0 (negatif-kontrol) satirlarinda recall/precision tanim geregi
    # 0.0'dir - bu "basarisizlik" degil, bos GT'ye karsi hicbir tahmin
    # "dogru" sayilamamasi demektir. Boyle satirlarda asil anlamli sinyal
    # n_pred_intervals: sistem corpus'ta olmayan bir kavram icin kac sonuc
    # donduruyor (idealde az/hicbir sey).
    note = " (n_gt=0: P/R tanımsız, n_pred'e bak)" if r["n_gt"] == 0 else ""
    return (
        f"<tr><td>{r['query']}</td><td>{r.get('lang', '')}</td>"
        f"<td>{r.get('category', '')}</td><td>{r['n_gt']}</td>"
        f"<td>{r['n_pred_intervals']}</td>"
        f"<td>{by10['recall@k']:.2f}{note}</td><td>{by10['precision@k']:.2f}</td>"
        f"<td>{r['mrr']:.2f}</td></tr>"
    )


def _run_html(run: dict) -> str:
    spec = run["result"]["spec"]
    rows = run["result"]["rows"]
    cat_agg = aggregate_by_category(rows)
    rows_html = "\n".join(_row_html(r) for r in rows)
    cat_html = "".join(
        f"<tr><td>{c}</td><td>{m['recall@10']:.2f}</td>"
        f"<td>{m['precision@10']:.2f}</td><td>{m['mrr']:.2f}</td></tr>"
        for c, m in cat_agg.items()
    )
    duration = run["manifest"].get("duration_s") or 0.0
    return f"""
    <h2>{run['run_id']}</h2>
    <p>model={spec['model_name']} filtre={spec['use_filters']}
       strateji={spec['strategy']} donanim={spec['hardware_profile']}
       sure={duration:.1f}sn</p>
    <h3>Kategori ozeti</h3>
    <table border="1"><tr><th>kategori</th><th>recall@10</th><th>precision@10</th><th>mrr</th></tr>
    {cat_html}</table>
    <h3>Sorgu detayi</h3>
    <p><i>negatif-kontrol satırlarında (n_gt=0) recall/precision tanım gereği
    0.0'dır; anlamlı sinyal n_pred (sistemin corpus'ta olmayan bir kavram
    için kaç sonuç döndürdüğü) kolonudur.</i></p>
    <table border="1"><tr><th>sorgu</th><th>dil</th><th>kategori</th><th>n_gt</th>
    <th>n_pred</th><th>recall@10</th><th>precision@10</th><th>mrr</th></tr>
    {rows_html}</table>
    """


def write_report(results: list, out_dir: str = "artifacts") -> tuple:
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_payload = []
    for run in results:
        json_payload.append({
            "run_id": run["run_id"],
            "spec": run["result"]["spec"],
            "rows": run["result"]["rows"],
            "timing": run["result"]["timing"],
            "category_summary": aggregate_by_category(run["result"]["rows"]),
            "manifest": run["manifest"],
        })

    html = (
        '<!doctype html><html><head><meta charset="utf-8">'
        "<title>Benchmark raporu</title></head><body>"
        "<h1>Faz 1 benchmark raporu</h1>"
        + "".join(_run_html(run) for run in results)
        + "</body></html>"
    )

    html_path = out_path / "benchmark_report.html"
    json_path = out_path / "benchmark_report.json"
    html_path.write_text(html, encoding="utf-8")
    json_path.write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return str(html_path), str(json_path)
