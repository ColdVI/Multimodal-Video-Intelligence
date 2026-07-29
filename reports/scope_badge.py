"""Her rapora tutarli bir 'kapsam rozeti' ekler: bu veri GERCEK mi
SENTETIK mi SMOKE_TEST mi PILOT mu, kac video/pencere/sorgu, amaci ne,
hangi iddialar desteklenir/desteklenmez (unified_search_harness_
duzeltmeler.md sonrasi "Reporting v2" istegi - raporun deney kapsami
raporun kendisinde acikca gorunsun, okuyan disari cikarim yapmak zorunda
kalmasin).

Inline stil kullanir (host raporun kendi <style>'ina BAGLI DEGIL) -
herhangi bir mevcut rapora (clickhouse_search_report, strategy_matrix_
report, msrvtt_validation_report...) drop-in eklenebilir, o raporun CSS
sinif isimlendirmesini bilmeye gerek yok."""
import html

_KIND_STYLES = {
    "SMOKE_TEST": {"bg": "#fff4e0", "border": "#e0a500", "fg": "#7a4a00", "label": "SMOKE TEST"},
    "REAL": {"bg": "#e4f0e9", "border": "#3c7a58", "fg": "#245c3f", "label": "GERÇEK VERİ"},
    "SYNTHETIC": {"bg": "#e6eeff", "border": "#3f5fa8", "fg": "#233e73", "label": "SENTETİK VERİ"},
    "PILOT": {"bg": "#f6e4df", "border": "#a8402c", "fg": "#7a2d1e", "label": "PİLOT / UNDERPOWERED"},
}


def render_scope_badge(kind: str, dataset: str, count, count_label: str,
                       purpose: str, can_claim: list, cannot_claim: list,
                       generated_at: str = None) -> str:
    """kind: 'SMOKE_TEST' | 'REAL' | 'SYNTHETIC' | 'PILOT'.
    count: int (bindelik ayraciyla bicimlendirilir) VEYA 'unknown' gibi bir
    string (bicimlendirilmeden oldugu gibi gosterilir - gercek sayi
    desteklenmiyorsa UYDURULMAZ, bkz. scripts/migrate_capera_results.py).
    can_claim/cannot_claim: kisa cumle listeleri - bu rapordan hangi
    sonuclarin cikarilabilecegi/cikarilamayacagi acikca yazilir."""
    if kind not in _KIND_STYLES:
        raise ValueError(f"bilinmeyen kind {kind!r}. Gecerli: {sorted(_KIND_STYLES)}")
    s = _KIND_STYLES[kind]
    count_str = f"{count:,}" if isinstance(count, int) else html.escape(str(count))
    can_html = "".join(f"<li>{html.escape(c)}</li>" for c in can_claim)
    cannot_html = "".join(f"<li>{html.escape(c)}</li>" for c in cannot_claim)
    date_html = (f'<span style="font-size:12px;opacity:.75">{html.escape(generated_at)}</span>'
                if generated_at else "")
    return f"""<div style="border:2px solid {s['border']};background:{s['bg']};color:{s['fg']};
border-radius:10px;padding:14px 18px;margin:0 0 18px;font-family:Arial,sans-serif;line-height:1.5">
  <div style="display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap">
    <span style="font-weight:700;letter-spacing:.04em;font-size:13px">{html.escape(s['label'])}</span>
    {date_html}
  </div>
  <div style="margin-top:6px;font-size:14px"><b>{html.escape(dataset)}</b> &middot; {count_str} {html.escape(count_label)}</div>
  <div style="margin-top:6px;font-size:13px">{html.escape(purpose)}</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px;font-size:12.5px">
    <div><b>Bu rapordan çıkarılabilir:</b><ul style="margin:4px 0 0;padding-left:18px">{can_html}</ul></div>
    <div><b>Bu rapordan ÇIKARILAMAZ:</b><ul style="margin:4px 0 0;padding-left:18px">{cannot_html}</ul></div>
  </div>
</div>"""


__all__ = ["render_scope_badge"]
