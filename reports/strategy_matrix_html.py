"""Faz 2 strateji matrisi + olcek kanitini tek HTML'de goruntuler.
reports/clickhouse_search_report.py'deki _table_html deseniyle uyumlu,
basit bir olusturucu."""
from __future__ import annotations

import html


def _rows_to_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "<p>Sonuç yok.</p>"
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = []
    for row in rows:
        cells = []
        for c in columns:
            v = row.get(c, "")
            if isinstance(v, float):
                v = f"{v:.3f}"
            cells.append(f"<td>{html.escape(str(v))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def render_strategy_report(small_scale: dict, scale_100k: dict = None,
                           memory_projection: dict = None) -> str:
    matrix_cols = ["strategy", "selectivity", "table", "row_count", "rows_read",
                   "vector_index_in_plan", "p50_ms", "p95_ms"]
    matrix_html = _rows_to_table(small_scale["matrix"], matrix_cols)
    fetch_html = _rows_to_table(
        small_scale["fetch_multiplier_sweep"],
        ["fetch_multiplier", "table", "row_count", "p50_ms"])
    ef_html = _rows_to_table(
        small_scale["ef_search_sweep"], ["ef_search", "table", "row_count", "p50_ms"])
    recall_html = _rows_to_table(
        small_scale["hnsw_recall_at_10"],
        ["table", "k", "recall_at_k", "n_overlap", "n_exact"])

    scale_section = ""
    if scale_100k:
        scale_cols = ["strategy", "selectivity", "row_count", "rows_read", "p50_ms", "p95_ms"]
        scale_html = _rows_to_table(scale_100k["matrix"], scale_cols)
        r = scale_100k["hnsw_recall_at_10"]
        scale_section = f"""
        <section class="card"><h2>100K sentetik ölçek (bench_scale_512)</h2>
        <p>HNSW recall@10: {r['recall_at_k']:.3f} ({r['n_overlap']}/{r['n_exact']})</p>
        {scale_html}</section>"""

    memory_section = ""
    if memory_projection:
        rows = [{"senaryo": k, "gb": v} for k, v in memory_projection.items()]
        memory_html = _rows_to_table(rows, ["senaryo", "gb"])
        memory_section = f"""<section class="card"><h2>Bellek projeksiyonu (gerçek 100K/512d ölçümünden)</h2>
        {memory_html}
        <p>vector_similarity_index_cache_size varsayılanı 5 GB.</p></section>"""

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Faz 2: ClickHouse strateji matrisi</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}}
th,td{{border:1px solid #ddd;padding:6px;text-align:left}} th{{background:#f4f4f4}}
.card{{border:1px solid #ddd;border-radius:8px;padding:16px;margin:16px 0}}
</style></head><body>
<h1>Faz 2: ClickHouse arama katmanı — strateji matrisi</h1>
<section class="card"><h2>4 strateji × 2 seçicilik × 2 tablo (73 satır, gerçek smoke veri)</h2>
{matrix_html}</section>
<section class="card"><h2>fetch_multiplier sweep</h2>{fetch_html}</section>
<section class="card"><h2>ef_search (hnsw_candidate_list_size_for_search) sweep</h2>{ef_html}</section>
<section class="card"><h2>HNSW recall@10 (73 satır)</h2>{recall_html}</section>
{scale_section}
{memory_section}
</body></html>"""
