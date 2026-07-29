"""artifacts/pipeline_validation.json'i (scripts/validate_msrvtt.py ciktisi)
tek bir okunabilir HTML'de goruntuler. reports/clickhouse_search_report.py
ve reports/strategy_matrix_html.py'deki _table_html deseniyle uyumlu,
canli hesap yapmaz - JSON zaten olculmus kaniti render eder."""
from __future__ import annotations

import html

_METRIC_ORDER = ["R@1", "R@5", "R@10", "MedR"]


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.1f}"
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


def _comparison_rows(measured: dict, baseline: dict, flagged_keys: set) -> list[dict]:
    rows = []
    for key in _METRIC_ORDER:
        if key not in measured:
            continue
        m = measured[key]
        b = baseline.get(key)
        row = {
            "metrik": key,
            "ölçülen": m,
            "zero-shot baseline": b if b is not None else "—",
            "fark (puan)": round(abs(m - b), 1) if b is not None else "—",
            "durum": ("BAYRAKLI" if key in flagged_keys
                     else ("referans (gate yok)" if key == "MedR" else "temiz")),
        }
        rows.append(row)
    return rows


def _seconds_to_human(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} sa"
    if seconds >= 60:
        return f"{seconds / 60:.1f} dk"
    return f"{seconds:.1f} sn"


def render_msrvtt_report(evidence: dict, scope_badge_html: str = "") -> str:
    """scope_badge_html: reports/scope_badge.py ciktisi, opsiyonel - bos ise
    mevcut (v1) cikti birebir ayni kalir."""
    zero_shot = evidence.get("zero_shot_baseline", {})
    zs_name = zero_shot.get("name", "")
    zs_values = zero_shot.get("values", {})
    reference_only = evidence.get("reference_only_baselines", {})

    model_sections = []
    for model_name, payload in evidence.get("results", {}).items():
        measured = payload["measured"]
        red_flags = payload.get("red_flags", [])
        flagged_keys = set()
        flag_lines = []
        for rf in red_flags:
            for f in rf["flags"]:
                flag_lines.append(f)
                key = f.split(":", 1)[0].strip()
                flagged_keys.add(key)

        comparison_html = _rows_to_table(
            _comparison_rows(measured, zs_values, flagged_keys),
            ["metrik", "ölçülen", "zero-shot baseline", "fark (puan)", "durum"])

        flags_html = (
            "<ul>" + "".join(f"<li>{html.escape(f)}</li>" for f in flag_lines) + "</ul>"
            if flag_lines else "<p>Zero-shot baseline'a karşı kırmızı bayrak yok.</p>")

        chance_line = measured.get("mean_rank_vs_chance", "")
        video_s = measured.get("video_embed_total_s")
        text_s = measured.get("text_embed_total_s")
        timing_note = ""
        if video_s is not None:
            timing_note = (f"<p class='meta'>Video embed: {_seconds_to_human(video_s)} "
                          f"({measured.get('n_videos_embedded', '?')} video) · "
                          f"Metin embed: {_seconds_to_human(text_s or 0)} · "
                          f"Değerlendirilen çift: {measured.get('n_pairs_evaluated', '?')}</p>")

        model_sections.append(f"""
        <section class="card">
          <h2>{html.escape(model_name)}</h2>
          <div class="grid">
            <div class="stat"><b>R@1</b><br>{_fmt(measured.get('R@1'))}</div>
            <div class="stat"><b>R@5</b><br>{_fmt(measured.get('R@5'))}</div>
            <div class="stat"><b>R@10</b><br>{_fmt(measured.get('R@10'))}</div>
            <div class="stat"><b>MedR</b><br>{_fmt(measured.get('MedR'))}</div>
            <div class="stat"><b>n</b><br>{_fmt(measured.get('n'))}</div>
          </div>
          <p class="chance">{html.escape(chance_line)}</p>
          {timing_note}
          <h3>Zero-shot baseline karşılaştırması</h3>
          <p class="meta">{html.escape(zs_name)}</p>
          {comparison_html}
          <h3 class="{'warn-title' if flag_lines else ''}">Kırmızı bayraklar</h3>
          {flags_html}
        </section>""")

    reference_rows = [{"model": name, **values} for name, values in reference_only.items()]
    reference_html = _rows_to_table(reference_rows, ["model", "R@1", "R@5", "R@10", "MedR"])

    return f"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>MSR-VTT 1k-A Boru Hattı Doğrulaması</title>
<style>
body{{font-family:Inter,Arial,sans-serif;max-width:1000px;margin:36px auto;padding:0 22px;color:#172033;line-height:1.55;background:#f7f9fc}}
.hero{{background:linear-gradient(120deg,#102a56,#2458a6);color:white;border-radius:18px;padding:24px}}
.card{{background:white;border:1px solid #dce4ef;border-radius:14px;padding:17px;margin:14px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:10px;margin:14px 0}}
.stat{{background:#f1f4f9;border-radius:10px;padding:10px;text-align:center}}
.chance{{font-weight:600;color:#174a9c;background:#e6eeff;border-radius:8px;padding:8px 12px;display:inline-block}}
.meta{{color:#667085;font-size:13px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}}
th,td{{border-bottom:1px solid #e5e9f0;padding:8px;text-align:left}} th{{background:#f1f4f9}}
.warn-title{{color:#b3541e}}
ul{{margin:6px 0}}
</style></head><body>
<div class="hero"><h1>MSR-VTT 1k-A Boru Hattı Doğrulaması</h1><p>{html.escape(evidence.get('protocol', ''))}</p></div>
{scope_badge_html}
{''.join(model_sections)}
<section class="card"><h2>Referans-amaçlı baseline'lar (kırmızı bayrak için KULLANILMAZ)</h2>
<p class="meta">Fine-tune edilmiş retrieval modelleri - üst sınır göstergesi, zero-shot checkpoint'imizle doğrudan kıyaslanamaz.</p>
{reference_html}</section>
</body></html>"""


__all__ = ["render_msrvtt_report"]
