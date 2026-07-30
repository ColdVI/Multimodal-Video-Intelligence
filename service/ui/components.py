"""Pure functions that render HTML fragments for the Gradio UI.

None of these touch Gradio state — they take plain dicts/values from the
`/search`, `/facets`, `/health` API responses (see UI_REDESIGN_TALIMATI.md
§0.1-§0.3 for the exact contract) and return HTML strings that get placed
into `gr.HTML` components by `service/ui/app.py`.
"""

from __future__ import annotations

import html
from typing import Any


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fmt_seconds(value: Any) -> str:
    try:
        return f"{float(value):.1f}s"
    except (TypeError, ValueError):
        return "—"


# ---------------------------------------------------------------- top bar --

def top_bar(product_name: str, dataset_id: str | None, health: dict[str, Any]) -> str:
    def dot(name: str, ok: Any) -> str:
        css_class = "mvi-health-dot mvi-health-dot--ok" if ok else "mvi-health-dot"
        return f'<span class="{css_class}">{_esc(name)}</span>'

    dots = "".join(dot(name, health.get(key)) for name, key in (("pg", "pg"), ("ch", "ch"), ("qdrant", "qdrant")))
    dataset_html = f'<span class="mvi-product__dataset">dataset: {_esc(dataset_id)}</span>' if dataset_id else ""
    return (
        '<div class="mvi-topbar">'
        '<div class="mvi-product">'
        f'<span class="mvi-product__name">{_esc(product_name)}</span>{dataset_html}'
        "</div>"
        f'<div class="mvi-health-row"><div class="mvi-health-dots">{dots}</div></div>'
        "</div>"
    )


def status_badge(message: str, level: str) -> str:
    level = level if level in {"success", "warning", "danger", "info"} else "info"
    return (
        f'<div class="status-badge status-badge--{level}">'
        '<span class="status-badge__dot"></span>'
        f'<span class="status-badge__text">{_esc(message)}</span>'
        "</div>"
    )


def warning_banner(text: str, level: str = "warning") -> str:
    level = level if level in {"danger", "warning", "info"} else "warning"
    return f'<div class="warning-banner warning-banner--{level}">{_esc(text)}</div>'


def section_header(title: str, subtitle: str | None = None) -> str:
    sub = f'<span class="section-header__subtitle">{_esc(subtitle)}</span>' if subtitle else ""
    return f'<div class="section-header"><span class="section-header__title">{_esc(title)}</span>{sub}</div>'


def pattern_not_implemented_badge() -> str:
    return (
        '<span class="pattern-not-implemented" '
        'title="Şu an yalnızca etiket; yürütme yolları henüz ayrışmadı">not implemented</span>'
    )


def filter_group_header(title: str, active_count: int) -> str:
    badge_class = "filter-badge-count" if active_count else "filter-badge-count filter-badge-count--zero"
    return (
        '<div class="filter-group-header">'
        f"<span>{_esc(title)}</span>"
        f'<span class="{badge_class}">{active_count} aktif</span>'
        "</div>"
    )


# --------------------------------------------------------------- metrics ---

def metric_card(label: str, value: str, sublabel: str | None = None) -> str:
    sub = f'<div class="metric-card__sub">{_esc(sublabel)}</div>' if sublabel else ""
    return (
        '<div class="metric-card">'
        f'<div class="metric-card__label">{_esc(label)}</div>'
        f'<div class="metric-card__value">{_esc(value)}</div>{sub}'
        "</div>"
    )


def metric_grid(cards: list[str]) -> str:
    return f'<div class="metric-grid">{"".join(cards)}</div>'


def score_indicator(score: Any) -> str:
    if score is None:
        return '<div class="score-indicator"><span class="score-indicator__value">—</span></div>'
    pct = max(0.0, min(1.0, float(score))) * 100
    return (
        '<div class="score-indicator">'
        f'<span class="score-indicator__value">{float(score):.4f}</span>'
        '<div class="score-indicator__track">'
        f'<div class="score-indicator__fill" style="width:{pct:.1f}%"></div>'
        "</div></div>"
    )


def time_range_bar(t_start: Any, t_end: Any) -> str:
    try:
        start = float(t_start)
        end = float(t_end)
    except (TypeError, ValueError):
        start, end = 0.0, 0.0
    span = max(end - start, 0.01)
    reference = max(end * 1.25, end + 20.0, span * 4.0, 1.0)
    start_pct = max(0.0, min(100.0, (start / reference) * 100))
    width_pct = max(2.0, min(100.0 - start_pct, (span / reference) * 100))
    return (
        '<div class="time-range-bar" '
        'title="Yaklaşık konum — video toplam süresi /search sözleşmesinde yok">'
        '<div class="time-range-bar__track">'
        f'<div class="time-range-bar__fill" style="left:{start_pct:.1f}%;width:{width_pct:.1f}%"></div>'
        "</div>"
        f'<div class="time-range-bar__caption">{_fmt_seconds(start)} – {_fmt_seconds(end)} · yaklaşık konum</div>'
        "</div>"
    )


def telemetry_badges(result: dict[str, Any]) -> str:
    badges: list[str] = []
    if result.get("altitude_m") is not None:
        badges.append(f'<span class="telemetry-badge">İrtifa {float(result["altitude_m"]):.1f} m</span>')
    if result.get("velocity_mps") is not None:
        badges.append(f'<span class="telemetry-badge">Hız {float(result["velocity_mps"]):.2f} m/s</span>')
    if result.get("gimbal_pitch") is not None:
        badges.append(f'<span class="telemetry-badge">Gimbal {float(result["gimbal_pitch"]):.1f}°</span>')
    if result.get("event_category"):
        badges.append(f'<span class="telemetry-badge">{_esc(result["event_category"])}</span>')
    for key, label in (("person_count", "kişi"), ("vehicle_count", "araç"), ("bus_count", "otobüs")):
        value = result.get(key)
        if value is not None:
            badges.append(f'<span class="telemetry-badge">{label}: {int(value)}</span>')
    if not badges:
        return ""
    return f'<div class="telemetry-badges">{"".join(badges)}</div>'


def media_slot(
    result: dict[str, Any], src: str | None = None, *, reason: str | None = None,
    source_exists: bool | None = None,
) -> str:
    if src:
        return (
            '<div class="media-slot media-slot--player">'
            f'<video controls preload="none" style="width:100%;border-radius:var(--radius-sm)" src="{_esc(src)}"></video>'
            "</div>"
        )
    file_path = result.get("file_path") or ""
    basename = file_path.rsplit("/", 1)[-1] if file_path else "bilinmeyen kaynak"
    t_start, t_end = result.get("t_start"), result.get("t_end")
    time_label = (
        f"{_fmt_seconds(t_start)} – {_fmt_seconds(t_end)}"
        if t_start is not None and t_end is not None
        else "—"
    )
    reason_text = reason or "Embedding indekslendi — medya önizlemesi bu ortamda servis edilmiyor."
    source_text = "evet" if source_exists else "hayır" if source_exists is False else "bilinmiyor"
    return (
        '<div class="media-slot">'
        '<span class="media-slot__icon" aria-hidden="true">▤</span>'
        "<div>"
        f'<div class="media-slot__title">Segment {time_label} · {_esc(basename)}</div>'
        f'<div class="media-slot__sub">{_esc(reason_text)} Kaynak mevcut: {_esc(source_text)}</div>'
        "</div></div>"
    )


def search_result_card(result: dict[str, Any], rank: int, *, primary: bool = False) -> str:
    card_class = "result-card result-card--primary" if primary else "result-card"
    segment_id = result.get("segment_id", "")
    caption = result.get("caption")
    caption_html = f'<div class="result-card__caption">&ldquo;{_esc(caption)}&rdquo;</div>' if caption else ""
    return (
        f'<div class="{card_class}">'
        f'<div class="result-card__rank">#{rank}</div>'
        '<div class="result-card__body">'
        '<div class="result-card__identity">'
        f'<span class="result-card__video-id">{_esc(result.get("video_id", ""))}</span>'
        f'<span class="result-card__time">{_fmt_seconds(result.get("t_start"))} → {_fmt_seconds(result.get("t_end"))}</span>'
        "</div>"
        f"{time_range_bar(result.get('t_start'), result.get('t_end'))}"
        f"{telemetry_badges(result)}"
        f"{caption_html}"
        '<span class="segment-id" title="Kopyalamak için tıkla" '
        f'data-copy="{_esc(segment_id)}" onclick="navigator.clipboard.writeText(this.dataset.copy)">'
        f"{_esc(segment_id)}</span>"
        f"{media_slot(result)}"
        "</div>"
        f'<div class="result-card__aside">{score_indicator(result.get("score"))}</div>'
        "</div>"
    )


def result_list(results: list[dict[str, Any]]) -> str:
    if not results:
        return empty_state("no_results")
    cards = [search_result_card(row, index + 1, primary=(index == 0)) for index, row in enumerate(results)]
    return f'<div class="result-list">{"".join(cards)}</div>'


def result_detail_panel(
    result: dict[str, Any], meta: dict[str, Any], media: dict[str, Any] | None = None,
) -> str:
    media = media or {}
    fields = [
        ("Video ID", result.get("video_id")),
        ("Segment ID", result.get("segment_id")),
        ("Zaman aralığı", f"{_fmt_seconds(result.get('t_start'))} → {_fmt_seconds(result.get('t_end'))}"),
        ("Skor", f"{result['score']:.6f}" if result.get("score") is not None else None),
        ("İrtifa (m)", result.get("altitude_m")),
        ("Hız (m/s)", result.get("velocity_mps")),
        ("Gimbal pitch", result.get("gimbal_pitch")),
        ("Event category", result.get("event_category")),
        ("Split", result.get("split")),
        ("Kişi sayısı", result.get("person_count")),
        ("Araç sayısı", result.get("vehicle_count")),
        ("Otobüs sayısı", result.get("bus_count")),
        ("Caption", result.get("caption")),
        ("Dosya yolu", result.get("file_path")),
        ("Active run", meta.get("run_id")),
        ("Dataset version", meta.get("dataset_version")),
        ("Vector provenance", meta.get("vector_provenance")),
        ("Model", meta.get("model_id")),
        ("Model revision", meta.get("model_revision")),
        ("Filter mode", meta.get("filter_execution_mode")),
        ("Clip URL", media.get("clip_url")),
        ("Kaynak mevcut", media.get("source_exists")),
        ("Media nedeni", media.get("reason")),
    ]
    items = "".join(
        '<div class="diagnostics-item">'
        f'<div class="diagnostics-item__label">{_esc(label)}</div>'
        f'<div class="diagnostics-item__value">{_esc(value)}</div>'
        "</div>"
        for label, value in fields
        if value is not None
    )
    narrowing = f'{meta.get("candidate_count", "—")} aday → {meta.get("returned_count", "—")} sonuç'
    extra = result.get("extra") or {}
    extra_html = ""
    if extra:
        extra_html = (
            '<div class="diagnostics-notes"><strong>Extra telemetry (read-only)</strong>'
            f'<pre style="white-space:pre-wrap">{_esc(extra)}</pre></div>'
        )
    clip_url = media.get("clip_url")
    media_src = f'{meta.get("api_url", "")}{clip_url}' if clip_url else None
    return (
        '<div class="diagnostics-panel">'
        f'{media_slot(result, media_src, reason=media.get("reason"), source_exists=media.get("source_exists"))}'
        f'<div class="diagnostics-grid" style="margin-top:var(--space-4)">{items}</div>'
        '<div class="diagnostics-notes">'
        f'backend={_esc(meta.get("backend"))} · strategy={_esc(meta.get("strategy"))} · '
        f'dimension={_esc(meta.get("dimension"))} · filtre daraltması: {_esc(narrowing)}'
        f"</div>{extra_html}</div>"
    )


# --------------------------------------------------------------- latency ---

def latency_row(label: str, ms: float, max_ms: float, *, emphasize: bool = False) -> str:
    pct = 0.0 if max_ms <= 0 else max(2.0, min(100.0, (ms / max_ms) * 100))
    row_class = "latency-row latency-row--total" if emphasize else "latency-row"
    return (
        f'<div class="{row_class}">'
        f'<span class="latency-row__label">{_esc(label)}</span>'
        '<div class="latency-row__track">'
        f'<div class="latency-row__fill" style="width:{pct:.1f}%"></div>'
        "</div>"
        f'<span class="latency-row__value">{ms:.1f} ms</span>'
        "</div>"
    )


def latency_panel(timings: dict[str, Any], stats: dict[str, Any]) -> str:
    stages = [
        ("Filter", "filter"), ("Embed", "embed"), ("Vector search", "vector_search"),
        ("Hydrate", "hydrate"), ("Toplam", "total"),
    ]
    max_ms = max((float(timings.get(key, 0.0) or 0.0) for _, key in stages), default=1.0) or 1.0
    rows = "".join(
        latency_row(label, float(timings.get(key, 0.0) or 0.0), max_ms, emphasize=(key == "total"))
        for label, key in stages
    )
    cards = metric_grid([
        metric_card("p50", f'{stats.get("p50", 0):.1f} ms'),
        metric_card("p95", f'{stats.get("p95", 0):.1f} ms'),
        metric_card("Tekrar", str(stats.get("n_repeats", "—"))),
    ])
    return f'<div class="diagnostics-panel">{rows}<div style="margin-top:var(--space-4)">{cards}</div></div>'


# ----------------------------------------------------------- diagnostics ---

def _fmt_diag_value(key: str, value: Any) -> str:
    if value is None:
        return "ölçülmedi" if key == "indexed_vectors_count" else "—"
    if isinstance(value, bool):
        return "evet" if value else "hayır"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "—"
    return str(value)


def diagnostics_panel(diagnostics: dict[str, Any], embedding_mode: str) -> str:
    fields = [
        ("candidate_count", "Aday sayısı"), ("returned_count", "Dönen sonuç"),
        ("underfilled", "Az doldu mu"), ("underfilled_reason", "Az dolma nedeni"),
        ("plan_used_vector_index", "Vektör indeksi kullanıldı"),
        ("indexed_vectors_count", "İndekslenen vektör"),
        ("filter_correctness", "Filtre doğruluğu"),
        ("quality_vs_groundtruth", "Kalite (groundtruth)"),
        ("r_at_1", "R@1"), ("ndcg", "nDCG"),
    ]
    items = "".join(
        '<div class="diagnostics-item">'
        f'<div class="diagnostics-item__label">{_esc(label)}</div>'
        f'<div class="diagnostics-item__value">{_esc(_fmt_diag_value(key, diagnostics.get(key)))}</div>'
        "</div>"
        for key, label in fields
    )
    items += (
        '<div class="diagnostics-item">'
        '<div class="diagnostics-item__label">Embedding mode</div>'
        f'<div class="diagnostics-item__value">{_esc(embedding_mode)}</div>'
        "</div>"
    )
    notes = diagnostics.get("notes") or []
    notes_html = f'<div class="diagnostics-notes">notes: {_esc("; ".join(notes))}</div>' if notes else ""
    return (
        '<div class="diagnostics-panel">'
        f'<div class="diagnostics-grid">{items}</div>{notes_html}'
        "</div>"
    )


# ---------------------------------------------------- empty/error/loading --

_EMPTY_STATES: dict[str, tuple[str, str, str]] = {
    "no_query": ("🔍", "Henüz sorgu yapılmadı", "Sol paneldeki metin kutusuna bir sorgu yazıp Search'e basın."),
    "no_results": (
        "🗂️", "Sonuç bulunamadı",
        "Bu sorgu ve filtre kombinasyonu için eşleşen segment yok. Sorguyu genişletmeyi veya "
        "filtreleri gevşetmeyi deneyin.",
    ),
    "filter_too_narrow": (
        "🧭", "Filtreler çok dar",
        "Aktif filtreler aday havuzunu daralttı. \"Clear filters\" ile filtreleri temizleyip tekrar deneyin.",
    ),
    "cached_query_missing": (
        "💾", "Cached sorgu bulunamadı",
        "Bu embedding modunda yalnızca önceden üretilmiş sorgu embedding'leri kullanılabiliyor; "
        "bu metin cache'te yok.",
    ),
    "backend_unavailable": (
        "🔌", "Backend şu anda erişilemiyor",
        "Seçilen backend/servis bağlantısı başarısız oldu. Health panelinden bağlantı durumunu kontrol edin.",
    ),
    "quality_unavailable": (
        "📉", "Kalite metrikleri yok",
        "Bu dataset/sorgu için groundtruth tabanlı kalite ölçümü mevcut değil (quality_vs_groundtruth boş).",
    ),
    "no_selection": (
        "🧪", "Karşılaştırma yok", "En az bir backend ve bir boyut seçin.",
    ),
}


def empty_state(kind: str, *, detail: str | None = None) -> str:
    icon, title, body = _EMPTY_STATES.get(kind, ("⚠️", "Beklenmeyen durum", detail or ""))
    return (
        '<div class="empty-state">'
        f'<div class="empty-state__icon">{icon}</div>'
        f'<div class="empty-state__title">{_esc(title)}</div>'
        f'<div class="empty-state__body">{_esc(body)}</div>'
        "</div>"
    )


def loading_state(message: str, *, cold_start: bool = False) -> str:
    hint = (
        '<div class="empty-state__body">İlk sorguda model yüklenir (~28 s) ve ilk gerçek sorgu '
        "hesaplanır (~43 s). Sonraki sorgular ~0.7 s sürer.</div>"
        if cold_start else ""
    )
    progress = '<div class="cold-start-progress"><div class="cold-start-progress__fill"></div></div>' if cold_start else ""
    return (
        '<div class="empty-state empty-state--loading">'
        '<div class="empty-state__icon">⏳</div>'
        f'<div class="empty-state__title">{_esc(message)}</div>'
        f"{hint}{progress}"
        "</div>"
    )


def error_state(message: str, raw_detail: str | None = None) -> str:
    details = ""
    if raw_detail:
        details = (
            '<details style="margin-top:var(--space-2)">'
            '<summary style="cursor:pointer;color:var(--text-secondary);font-size:11px">Teknik detay</summary>'
            f'<pre style="white-space:pre-wrap;font-size:11px;color:var(--text-secondary)">{_esc(raw_detail)}</pre>'
            "</details>"
        )
    return (
        '<div class="empty-state">'
        '<div class="empty-state__icon">⚠️</div>'
        '<div class="empty-state__title">İstek tamamlanamadı</div>'
        f'<div class="empty-state__body">{_esc(message)}</div>'
        f"{details}"
        "</div>"
    )


# ------------------------------------------------------------ comparison ---

def comparison_group_header(title: str) -> str:
    return f'<div class="comparison-group-header">{_esc(title)}</div>'


def comparison_card(row: dict[str, Any]) -> str:
    interpretable = row.get("embedding_mode") not in (None, "synthetic")
    flag_class = "interpretable-flag--yes" if interpretable else "interpretable-flag--no"
    flag_text = "interpretable" if interpretable else "NOT interpretable"
    error = row.get("error")
    if error:
        body = f'<div class="comparison-card__metrics"><span>hata: {_esc(error)}</span></div>'
    else:
        body = (
            '<div class="comparison-card__metrics">'
            f'<span>p50 {row.get("p50_ms", 0):.1f} ms · p95 {row.get("p95_ms", 0):.1f} ms</span>'
            f'<span>recall_vs_exact: {_esc(row.get("recall_vs_exact"))}</span>'
            f'<span>returned: {row.get("returned_count", "—")} · '
            f'underfilled: {"evet" if row.get("underfilled") else "hayır"}</span>'
            "</div>"
        )
    return (
        '<div class="comparison-card">'
        '<div class="comparison-card__title">'
        f'<span>{_esc(row.get("backend"))} · {_esc(row.get("strategy"))} · {_esc(row.get("dimension"))}d</span>'
        f'<span class="interpretable-flag {flag_class}">{flag_text}</span>'
        "</div>"
        f"{body}</div>"
    )


def comparison_grid(cards: list[str]) -> str:
    return f'<div class="comparison-grid">{"".join(cards)}</div>'
