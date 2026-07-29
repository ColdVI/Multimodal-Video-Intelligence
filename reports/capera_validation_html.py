"""artifacts/capera_validation.json'i (scripts/migrate_capera_results.py
ciktisi) tek bir okunabilir HTML'de goruntuler. reports/msrvtt_validation_
html.py ile ayni desen (artifact_matrix backend, ClickHouse'a yazmayan
dataset'ler icin) - canli hesap yapmaz, zaten migrasyon edilmis kaniti
render eder."""
from __future__ import annotations

import html


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _rows_to_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "<p>Sonuç yok.</p>"
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(_fmt(row.get(c, '')))}</td>" for c in columns)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def render_capera_report(evidence: dict, scope_badge_html: str = "") -> str:
    """scope_badge_html: reports/scope_badge.py ciktisi, opsiyonel."""
    results = evidence.get("results", {})
    metric_cols = ["model", "n_videos", "n_queries", "embedding_dim", "recall_at_1",
                   "recall_at_5", "recall_at_10", "mrr", "peak_gpu_memory_mb"]
    rows = [{"model": name, **payload} for name, payload in results.items()]
    rows.sort(key=lambda r: -r.get("recall_at_1", 0))
    results_html = _rows_to_table(rows, metric_cols)

    gaps_html = "".join(f"<li>{html.escape(g)}</li>" for g in evidence.get("known_gaps", []))
    dm = evidence.get("dataset_manifest", {})
    dm_html = _rows_to_table([dm], list(dm)) if dm else ""

    counts = evidence.get("counts", {})
    counts_html = ""
    if counts:
        counts_rows = [
            {"kaynak": "manifest (caption JSON'ları)",
             "video": counts.get("manifest_video_count", "unknown"),
             "sorgu": counts.get("manifest_query_count", "unknown"),
             "başarısız": "—"},
            {"kaynak": "evaluated (all_results.json)",
             "video": counts.get("evaluated_video_count", "unknown"),
             "sorgu": counts.get("evaluated_query_count", "unknown"),
             "başarısız": counts.get("failed_video_count", "unknown")},
        ]
        counts_html = f"""<section class="card warn"><h2>Manifest ile değerlendirilen ayrı tutulur</h2>
        <p>Caption JSON'larındaki TOPLAM sayı ile all_results.json'da GERÇEKTEN
        değerlendirilen sayı aynı değildir - biri diğerinin tamamı olduğu
        VARSAYILMAZ.</p>
        {_rows_to_table(counts_rows, ["kaynak", "video", "sorgu", "başarısız"])}</section>"""

    return f"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>CapERA Model Karşılaştırması</title>
<style>
body{{font-family:Inter,Arial,sans-serif;max-width:1080px;margin:36px auto;padding:0 22px;color:#172033;line-height:1.55;background:#f7f9fc}}
.hero{{background:linear-gradient(120deg,#102a56,#2458a6);color:white;border-radius:18px;padding:24px}}
.card{{background:white;border:1px solid #dce4ef;border-radius:14px;padding:17px;margin:14px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}}
th,td{{border-bottom:1px solid #e5e9f0;padding:8px;text-align:left}} th{{background:#f1f4f9}}
.warn{{border-left:5px solid #e7a500;background:#fff9ee}}
</style></head><body>
<div class="hero"><h1>CapERA Model Karşılaştırması</h1><p>{html.escape(evidence.get('protocol', ''))}</p></div>
{scope_badge_html}
{counts_html}
<section class="card"><h2>Model sonuçları ({len(rows)} model, gerçek kayıtlı sonuçlar)</h2>
{results_html}</section>
<section class="card warn"><h2>Bilinen boşluklar</h2><ul>{gaps_html}</ul></section>
<section class="card"><h2>Dataset manifest</h2>{dm_html}
<p style="font-size:12px;color:#667085">{html.escape(evidence.get('source', ''))}</p></section>
</body></html>"""


__all__ = ["render_capera_report"]
