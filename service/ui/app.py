from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import httpx
import pandas as pd

from app.embedding.router import mode_details
from app.search.strategies import SUPPORTED_STRATEGIES
from ui import components


API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
PRODUCT_NAME = "Multimodal Video Intelligence"
RESULT_COLUMNS = [
    "video_id", "t_start", "t_end", "score", "caption", "file_path",
    "altitude_m", "velocity_mps", "gimbal_pitch",
    "event_category", "split", "person_count", "vehicle_count", "bus_count",
]
NUMERIC_METADATA_FIELDS = {"person_count", "vehicle_count", "bus_count"}
PRIMARY_FILTER_FIELDS = {"event_category", "split", "video_id", "altitude_m", "velocity_mps", "gimbal_pitch"}

CSS = (Path(__file__).resolve().parent / "static" / "theme.css").read_text(encoding="utf-8")


# ------------------------------------------------------------- API client --

def _get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    with httpx.Client(timeout=30.0) as client:
        response = client.get(f"{API_URL}{path}", params=params)
        response.raise_for_status()
        return response.json()


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    with httpx.Client(timeout=300.0) as client:
        response = client.post(f"{API_URL}{path}", json=payload)
        response.raise_for_status()
        return response.json()


def _error_detail(exc: httpx.HTTPStatusError) -> str:
    try:
        payload = exc.response.json()
    except Exception:
        return exc.response.text
    detail = payload.get("detail", payload)
    if isinstance(detail, list):
        return "; ".join(str(item.get("msg", item)) if isinstance(item, dict) else str(item) for item in detail)
    return str(detail)


def _datasets() -> list[str]:
    try:
        return [row["dataset_id"] for row in _get("/stats").get("datasets", [])]
    except Exception:
        return []


def _capabilities() -> dict[str, Any]:
    try:
        data = _get("/strategies")
        backends = list(data.get("enabled_backends") or [])
        dimensions = [int(value) for value in data.get("enabled_dimensions") or []]
        strategies = {name: list(data.get("strategies", {}).get(name, [])) for name in backends}
        if backends and dimensions and all(strategies.values()):
            return {"backends": backends, "dimensions": dimensions, "strategies": strategies}
    except Exception:
        pass
    return {"backends": ["clickhouse"], "dimensions": [512], "strategies": {"clickhouse": ["prefilter"]}}


def _range(lo: Any, hi: Any) -> list[float] | None:
    if lo is None or hi is None:
        return None
    return [float(lo), float(hi)]


# --------------------------------------------------------- sample queries --
# Talimat §2.2: örnek sorgular tests/fixtures/queries_semantic.json'dan okunur,
# uydurulmaz. app/config.py::_capera_protocol ile aynı iki-adaylı dev/konteyner
# yol deseni kullanılır (bkz. Dockerfile.ui'deki ek COPY satırı).

def _sample_queries_path() -> Path | None:
    candidates = (
        Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "queries_semantic.json",
        Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "queries_semantic.json",
    )
    return next((candidate for candidate in candidates if candidate.exists()), None)


def _sample_queries() -> list[dict[str, str]]:
    path = _sample_queries_path()
    if path is None:
        return []
    wanted = {"S01", "S02", "S03", "S04", "S05", "S06"}
    data = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in data if row.get("id") in wanted]


SAMPLE_QUERIES = _sample_queries()


# ------------------------------------------------------------- filter UI ---

def _dropdown_update(values: list[str] | None) -> Any:
    values = values or []
    return gr.update(choices=values, value=None, visible=bool(values))


def _slider_updates(telemetry: dict[str, Any], name: str) -> tuple[Any, Any]:
    bounds = telemetry.get(name)
    if not bounds:
        return gr.update(visible=False), gr.update(visible=False)
    lo, hi = bounds
    return (
        gr.update(minimum=lo, maximum=hi, value=lo, visible=True),
        gr.update(minimum=lo, maximum=hi, value=hi, visible=True),
    )


def load_facets(dataset_id: str | None):
    if not dataset_id:
        hidden = gr.update(visible=False)
        empty_dd = gr.update(choices=[], value=None, visible=False)
        return (
            empty_dd, empty_dd, empty_dd,
            hidden, hidden, hidden, hidden, hidden, hidden,
            {}, components.filter_group_header("Filtreler", 0), "",
            gr.update(value=[], visible=False), gr.update(choices=[False, True], value=None, visible=False),
        )
    data = _get(f"/facets/{dataset_id}")
    try:
        schema = _get(f"/datasets/{dataset_id}/filter-schema")
    except Exception:
        schema = {"fields": []}
    data["filter_schema"] = schema
    telemetry = data.get("telemetry", {})

    unavailable = []
    if not data.get("event_categories"):
        unavailable.append("event category")
    if not telemetry.get("altitude_m"):
        unavailable.append("irtifa")
    if not telemetry.get("velocity_mps"):
        unavailable.append("hız")
    if not telemetry.get("gimbal_pitch"):
        unavailable.append("gimbal pitch")
    note = (
        f'<div class="mvi-muted-note">Bu dataset için kullanılamıyor: {", ".join(unavailable)}</div>'
        if unavailable else ""
    )

    rows = []
    for field in schema.get("fields", []):
        bounds = field.get("bounds")
        if field.get("name") in PRIMARY_FILTER_FIELDS or not bounds:
            continue
        rows.append([
            field["name"], bounds[0], bounds[1], field.get("unit") or "",
            "wrap" if field.get("wrap") else "linear",
        ])
    has_is_night = any(field.get("name") == "is_night" for field in schema.get("fields", []))
    return (
        _dropdown_update(data.get("event_categories")),
        _dropdown_update(data.get("splits")),
        _dropdown_update(data.get("video_ids")),
        *_slider_updates(telemetry, "altitude_m"),
        *_slider_updates(telemetry, "velocity_mps"),
        *_slider_updates(telemetry, "gimbal_pitch"),
        data,
        components.filter_group_header("Filtreler", 0),
        note,
        gr.update(value=rows, visible=bool(rows)),
        gr.update(choices=[False, True], value=None, visible=has_is_night),
    )


def _count_active_filters(
    event_category, split, video_id,
    altitude_min, altitude_max, velocity_min, velocity_max, gimbal_min, gimbal_max,
    facets_state,
) -> int:
    count = sum(1 for value in (event_category, split, video_id) if value)
    telemetry = (facets_state or {}).get("telemetry", {})

    def changed(name: str, lo_val: Any, hi_val: Any) -> bool:
        bounds = telemetry.get(name)
        if not bounds or lo_val is None or hi_val is None:
            return False
        lo, hi = bounds
        return not (abs(float(lo_val) - lo) < 1e-6 and abs(float(hi_val) - hi) < 1e-6)

    count += sum([
        changed("altitude_m", altitude_min, altitude_max),
        changed("velocity_mps", velocity_min, velocity_max),
        changed("gimbal_pitch", gimbal_min, gimbal_max),
    ])
    return count


def update_filter_badge(*args) -> str:
    return components.filter_group_header("Filtreler", _count_active_filters(*args))


def clear_filters(facets_state):
    telemetry = (facets_state or {}).get("telemetry", {})

    def reset(name: str):
        bounds = telemetry.get(name)
        if not bounds:
            return gr.update(), gr.update()
        lo, hi = bounds
        return gr.update(value=lo), gr.update(value=hi)

    altitude = reset("altitude_m")
    velocity = reset("velocity_mps")
    gimbal = reset("gimbal_pitch")
    return (
        gr.update(value=None), gr.update(value=None), gr.update(value=None),
        *altitude, *velocity, *gimbal,
        components.filter_group_header("Filtreler", 0),
    )


def update_strategies(backend: str):
    choices = list(CAPABILITIES["strategies"].get(backend, ()))
    return gr.update(choices=choices, value=choices[0] if choices else None)


# ------------------------------------------------------------- top header --

def render_header(dataset_id: str | None):
    try:
        health = _get("/health", params={"dataset_id": dataset_id} if dataset_id else None)
    except Exception:
        health = {"pg": False, "ch": False, "qdrant": False, "embedding": {
            "level": "danger", "message": "API'ye ulaşılamıyor — backend health kontrolü başarısız.",
        }}
    embedding = health.get("embedding", {})
    top_html = components.top_bar(PRODUCT_NAME, dataset_id, health)
    badge_html = components.status_badge(embedding.get("message", ""), embedding.get("level", "info"))
    return top_html, badge_html


# ----------------------------------------------------------------- search --

def _payload(
    query: str,
    dataset_id: str,
    event_category: str | None,
    split: str | None,
    video_id: str | None,
    altitude_min: float | None,
    altitude_max: float | None,
    velocity_min: float | None,
    velocity_max: float | None,
    gimbal_min: float | None,
    gimbal_max: float | None,
    backend: str,
    strategy: str,
    dimension: int,
    adaptive: bool,
    base_dim: int,
    top_n: int,
    pattern: str,
    top_k: int,
    repeats: int,
    canonical_rows: list[list[Any]] | None = None,
    is_night: bool | None = None,
) -> dict[str, Any]:
    metadata_filters = {
        key: value
        for key, value in {
            "event_category": event_category,
            "split": split,
            "video_id": video_id,
        }.items()
        if value not in (None, "")
    }
    telemetry_filters = {
        key: bounds
        for key, bounds in {
            "altitude_m": _range(altitude_min, altitude_max),
            "velocity_mps": _range(velocity_min, velocity_max),
            "gimbal_pitch": _range(gimbal_min, gimbal_max),
        }.items()
        if bounds is not None
    }
    if is_night is not None:
        metadata_filters["is_night"] = bool(is_night)
    for row in canonical_rows or []:
        if not row or len(row) < 3 or row[1] is None or row[2] is None:
            continue
        name = str(row[0])
        if name in PRIMARY_FILTER_FIELDS or name == "is_night":
            continue
        target = metadata_filters if name in NUMERIC_METADATA_FIELDS else telemetry_filters
        target[name] = [float(row[1]), float(row[2])]
    return {
        "query": query,
        "dataset_id": dataset_id,
        "backend": backend,
        "strategy": strategy,
        "dimension": int(dimension),
        "adaptive_mrl": {"enabled": adaptive, "base_dim": int(base_dim), "top_n": int(top_n)},
        "metadata_filters": metadata_filters,
        "telemetry_filters": telemetry_filters,
        "pattern": pattern,
        "top_k": int(top_k),
        "repeats": int(repeats),
    }


def _sanitize_telemetry(
    lo: float | None, hi: float | None, name: str, facets_state: dict[str, Any] | None,
) -> tuple[float | None, float | None]:
    """A hidden telemetry slider still holds a numeric value (Slider can't be None), so a
    field the current dataset doesn't have (e.g. gimbal_pitch for AU-AIR) would otherwise leak
    into the payload as an active [0, 0] range and zero out every candidate. Only forward a
    slider's value once /facets confirmed the field has real bounds for this dataset."""
    if not (facets_state or {}).get("telemetry", {}).get(name):
        return None, None
    return lo, hi


def _detail_meta(response: dict[str, Any]) -> dict[str, Any]:
    diagnostics = response.get("diagnostics", {})
    return {
        "backend": response.get("backend"), "strategy": response.get("strategy"),
        "dimension": response.get("dimension"),
        "candidate_count": diagnostics.get("candidate_count"),
        "returned_count": diagnostics.get("returned_count"),
        "run_id": response.get("run_id"), "dataset_version": response.get("dataset_version"),
        "vector_provenance": response.get("vector_provenance"),
        "model_id": response.get("model_id"), "model_revision": response.get("model_revision"),
        "filter_execution_mode": response.get("filter_execution_mode"),
        "api_url": API_URL,
    }


def _media_for(result: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    try:
        return _get(
            f'/media/{result["segment_id"]}/info',
            params={"run_id": response.get("run_id")} if response.get("run_id") else None,
        )
    except Exception as exc:
        return {"available": False, "source_exists": False, "reason": f"media info unavailable: {exc}"}


def run_search(
    query, dataset_id, event_category, split, video_id,
    altitude_min, altitude_max, velocity_min, velocity_max, gimbal_min, gimbal_max,
    backend, strategy, dimension, adaptive, base_dim, top_n, pattern, top_k, repeats,
    canonical_rows, is_night, facets_state,
):
    try:
        cold_start = _get("/health").get("embedding_mode") == "hybrid_text"
    except Exception:
        cold_start = False
    yield (
        components.loading_state("Aranıyor…", cold_start=cold_start),
        gr.update(choices=[], value=None),
        "",
        components.loading_state("Ölçülüyor…"),
        components.loading_state("Ölçülüyor…"),
        None,
    )

    altitude_min, altitude_max = _sanitize_telemetry(altitude_min, altitude_max, "altitude_m", facets_state)
    velocity_min, velocity_max = _sanitize_telemetry(velocity_min, velocity_max, "velocity_mps", facets_state)
    gimbal_min, gimbal_max = _sanitize_telemetry(gimbal_min, gimbal_max, "gimbal_pitch", facets_state)
    payload = _payload(
        query, dataset_id, event_category, split, video_id,
        altitude_min, altitude_max, velocity_min, velocity_max, gimbal_min, gimbal_max,
        backend, strategy, dimension, adaptive, base_dim, top_n, pattern, top_k, repeats,
        canonical_rows, is_night,
    )
    try:
        response = _post("/search", payload)
    except httpx.HTTPStatusError as exc:
        detail = _error_detail(exc)
        if "cached mode has no real embedding" in detail:
            state_html = components.empty_state("cached_query_missing")
        else:
            state_html = components.error_state(
                f"Arama isteği reddedildi (HTTP {exc.response.status_code}).", detail,
            )
        yield state_html, gr.update(choices=[], value=None), "", "", "", None
        return
    except (httpx.ConnectError, httpx.ConnectTimeout):
        yield components.empty_state("backend_unavailable"), gr.update(choices=[], value=None), "", "", "", None
        return
    except Exception as exc:
        state_html = components.error_state("Beklenmeyen bir hata oluştu.", f"{type(exc).__name__}: {exc}")
        yield state_html, gr.update(choices=[], value=None), "", "", "", None
        return

    results = response.get("results", [])
    diagnostics = response.get("diagnostics", {})

    if not results:
        active_filters = _count_active_filters(
            event_category, split, video_id,
            altitude_min, altitude_max, velocity_min, velocity_max, gimbal_min, gimbal_max,
            facets_state,
        ) > 0
        if diagnostics.get("underfilled_reason") == "candidate_shortage" and active_filters:
            results_html = components.empty_state("filter_too_narrow")
        else:
            results_html = components.empty_state("no_results")
    else:
        results_html = components.result_list(results)
        if response.get("vector_provenance") == "synthetic":
            results_html = components.warning_banner(
                "SENTETİK EMBEDDING — bu sonuçlar sıralama kalitesi için anlamlı değildir, "
                "yalnızca sistem/gecikme doğrulaması.",
                "danger",
            ) + results_html

    detail_choices = [
        (f"#{index + 1} · {row.get('video_id')} · {row.get('t_start', 0):.1f}s–{row.get('t_end', 0):.1f}s", row["segment_id"])
        for index, row in enumerate(results)
    ]
    first_segment = results[0]["segment_id"] if results else None
    detail_html = (
        components.result_detail_panel(results[0], _detail_meta(response), _media_for(results[0], response))
        if results else ""
    )

    latency_html = components.latency_panel(response["timings_ms"], response["timings_stats"])
    diagnostics_html = components.diagnostics_panel(diagnostics, response["embedding_mode"])

    yield (
        results_html,
        gr.update(choices=detail_choices, value=first_segment),
        detail_html,
        latency_html,
        diagnostics_html,
        response,
    )


def show_detail(segment_id: str | None, raw_response: dict[str, Any] | None):
    if not raw_response or not segment_id:
        return ""
    match = next((row for row in raw_response.get("results", []) if row.get("segment_id") == segment_id), None)
    if match is None:
        return ""
    return components.result_detail_panel(match, _detail_meta(raw_response), _media_for(match, raw_response))


def export_csv(raw_response: dict[str, Any] | None):
    rows = (raw_response or {}).get("results", [])
    target = Path(tempfile.gettempdir()) / "faz9_search_results.csv"
    pd.DataFrame(rows, columns=RESULT_COLUMNS if rows else None).to_csv(target, index=False)
    return str(target)


# --------------------------------------------------------------- compare ---

def _run_comparison(query: str, dataset_id: str, backends: list[str], dimensions: list[int], repeats: int):
    rows = []
    for backend in backends or []:
        strategy = CAPABILITIES["strategies"][backend][0]
        for dimension in dimensions or []:
            payload = _payload(
                query, dataset_id, None, None, None, None, None, None, None, None, None,
                backend, strategy, int(dimension), False, 256, 100,
                "C" if backend == "pgvector" else "A", 10, int(repeats),
            )
            try:
                response = _post("/search", payload)
                rows.append({
                    "backend": backend, "strategy": strategy, "dimension": dimension,
                    "p50_ms": response["timings_stats"]["p50"],
                    "p95_ms": response["timings_stats"]["p95"],
                    "recall_vs_exact": response["diagnostics"].get("ann_recall_vs_exact"),
                    "returned_count": response["diagnostics"]["returned_count"],
                    "underfilled": response["diagnostics"]["underfilled"],
                    "embedding_mode": response["embedding_mode"],
                    "vector_provenance": response.get("vector_provenance"),
                })
            except Exception as exc:
                rows.append({"backend": backend, "strategy": strategy, "dimension": dimension, "error": str(exc)})
    return rows


def _render_comparison(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return components.empty_state("no_selection")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row.get("embedding_mode") or "bilinmiyor (hata)", []).append(row)
    parts = []
    if any(row.get("vector_provenance") == "synthetic" for row in rows):
        parts.append(components.warning_banner(
            "Bu karşılaştırmadaki sonuçlar sentetik embedding ile üretildi — sıralama kalitesi anlamlı değildir.",
            "danger",
        ))
    for mode, group_rows in groups.items():
        parts.append(components.comparison_group_header(f"embedding_mode: {mode}"))
        parts.append(components.comparison_grid([components.comparison_card(row) for row in group_rows]))
    return "".join(parts)


def run_compare(query: str, dataset_id: str, backends: list[str], dimensions: list[int], repeats: int):
    rows = _run_comparison(query, dataset_id, backends, dimensions, repeats)
    return _render_comparison(rows)


# --------------------------------------------------------------- layout ----

CAPABILITIES = _capabilities()
datasets = _datasets()
initial_backend = CAPABILITIES["backends"][0]
initial_dimension = CAPABILITIES["dimensions"][0]
initial_top_html, initial_badge_html = render_header(datasets[0] if datasets else None)

with gr.Blocks(title=f"{PRODUCT_NAME} — Faz 9") as demo:
    topbar_html = gr.HTML(initial_top_html, elem_id="mvi-topbar")
    status_badge_html = gr.HTML(initial_badge_html, elem_id="mvi-status-badge")

    with gr.Tabs():
        with gr.Tab("Ara"):
            with gr.Row():
                with gr.Column(scale=2):
                    dataset = gr.Dropdown(choices=datasets, value=datasets[0] if datasets else None, label="Dataset")
                    query = gr.Textbox(label="Serbest metin sorgusu", elem_id="mvi-query")
                    if SAMPLE_QUERIES:
                        with gr.Row(elem_classes=["chip-row"]):
                            for sample in SAMPLE_QUERIES:
                                gr.Button(
                                    sample["tr"], size="sm", elem_classes=["chip"],
                                ).click(
                                    lambda text=sample["tr"]: text, outputs=[query],
                                )
                    top_k = gr.Slider(1, 50, value=10, step=1, label="top_k")
                    facets_state = gr.State({})

                    with gr.Accordion("Filtreler", open=False, elem_id="mvi-filters"):
                        filter_badge_html = gr.HTML(components.filter_group_header("Filtreler", 0))
                        filter_note_html = gr.HTML("")
                        event_category = gr.Dropdown(label="Event category", visible=False)
                        split = gr.Dropdown(label="Split", visible=False)
                        video_id = gr.Dropdown(label="Video ID", visible=False)
                        with gr.Row():
                            altitude_min = gr.Slider(0, 1, label="İrtifa min (m)", visible=False)
                            altitude_max = gr.Slider(0, 1, label="İrtifa max (m)", visible=False)
                        with gr.Row():
                            velocity_min = gr.Slider(0, 1, label="Hız min (m/s)", visible=False)
                            velocity_max = gr.Slider(0, 1, label="Hız max (m/s)", visible=False)
                        with gr.Row():
                            gimbal_min = gr.Slider(0, 1, label="Gimbal pitch min", visible=False)
                            gimbal_max = gr.Slider(0, 1, label="Gimbal pitch max", visible=False)
                        is_night = gr.Dropdown([False, True], label="Night", visible=False)
                        canonical_filters = gr.Dataframe(
                            headers=["field", "min", "max", "unit", "semantics"],
                            datatype=["str", "number", "number", "str", "str"],
                            value=[], interactive=True, visible=False,
                            label="Diğer canonical filtreler (circular: min > max wrap uygular)",
                        )
                        clear_filters_button = gr.Button("Clear filters", size="sm")

                    with gr.Accordion("Advanced Search Settings", open=False, elem_id="mvi-advanced"):
                        backend = gr.Radio(CAPABILITIES["backends"], value=initial_backend, label="Backend")
                        initial_strategies = CAPABILITIES["strategies"][initial_backend]
                        strategy = gr.Dropdown(initial_strategies, value=initial_strategies[0], label="Strategy")
                        dimension = gr.Radio(CAPABILITIES["dimensions"], value=initial_dimension, label="Dimension")
                        adaptive = gr.Checkbox(False, label="Adaptive MRL")
                        base_dim = gr.Radio([256, 512], value=256, label="Base dimension")
                        top_n = gr.Slider(1, 200, value=100, step=1, label="Adaptive top_N")
                        with gr.Row():
                            pattern = gr.Radio(["A", "B", "C"], value="A", label="Pattern")
                            gr.HTML(components.pattern_not_implemented_badge())
                        repeats = gr.Slider(1, 20, value=1, step=1, label="Tekrar")

                    search_button = gr.Button("Search", variant="primary", elem_id="mvi-search-button")
                    ten_button = gr.Button("10 tekrar", size="sm")

                with gr.Column(scale=3):
                    gr.HTML(components.section_header("Arama sonuçları"))
                    results_html = gr.HTML(components.empty_state("no_query"), elem_id="mvi-results")

                    gr.HTML(components.section_header("Sonuç detayı"))
                    detail_selector = gr.Dropdown(
                        label="Detay için sonuç seç", choices=[], value=None, elem_id="mvi-detail-selector",
                    )
                    detail_html = gr.HTML("", elem_id="mvi-detail")

                    with gr.Row():
                        export_button = gr.Button("Sonucu CSV indir", size="sm")
                        download = gr.File(label="CSV")

                    raw_response = gr.State()

                    gr.HTML(components.section_header("Observability / Diagnostics", "Ana sonuçlardan ayrı — teşhis paneli"))
                    latency_html = gr.HTML(components.latency_panel({}, {}))
                    diagnostics_html = gr.HTML(components.diagnostics_panel({}, "—"))

            dataset.change(
                load_facets, inputs=[dataset],
                outputs=[
                    event_category, split, video_id,
                    altitude_min, altitude_max, velocity_min, velocity_max, gimbal_min, gimbal_max,
                    facets_state, filter_badge_html, filter_note_html,
                    canonical_filters, is_night,
                ],
            ).then(render_header, inputs=[dataset], outputs=[topbar_html, status_badge_html])

            filter_controls = [
                event_category, split, video_id,
                altitude_min, altitude_max, velocity_min, velocity_max, gimbal_min, gimbal_max,
            ]
            filter_sliders = [altitude_min, altitude_max, velocity_min, velocity_max, gimbal_min, gimbal_max]
            # Both .select (dropdown) and .release (slider) fire only on a genuine user
            # interaction, never on the programmatic gr.update() batch load_facets sends when a
            # dataset loads/changes. Using .change instead races that batch: a listener that
            # takes a slider as input gets invoked with the browser's stale pre-update value
            # while the server has already applied the new minimum/maximum, and Gradio's input
            # validation rejects it ("Value 0 is less than minimum value 2.8381").
            for control in (event_category, split, video_id):
                control.select(update_filter_badge, inputs=[*filter_controls, facets_state], outputs=[filter_badge_html])
            for control in filter_sliders:
                control.release(update_filter_badge, inputs=[*filter_controls, facets_state], outputs=[filter_badge_html])

            clear_filters_button.click(
                clear_filters, inputs=[facets_state],
                outputs=[
                    event_category, split, video_id,
                    altitude_min, altitude_max, velocity_min, velocity_max, gimbal_min, gimbal_max,
                    filter_badge_html,
                ],
            )

            backend.change(update_strategies, inputs=[backend], outputs=[strategy])

            search_inputs = [
                query, dataset, event_category, split, video_id,
                altitude_min, altitude_max, velocity_min, velocity_max, gimbal_min, gimbal_max,
                backend, strategy, dimension, adaptive, base_dim, top_n, pattern, top_k, repeats,
                canonical_filters, is_night, facets_state,
            ]
            search_outputs = [results_html, detail_selector, detail_html, latency_html, diagnostics_html, raw_response]
            search_button.click(run_search, inputs=search_inputs, outputs=search_outputs)
            ten_button.click(lambda: 10, outputs=[repeats]).then(run_search, inputs=search_inputs, outputs=search_outputs)
            detail_selector.change(show_detail, inputs=[detail_selector, raw_response], outputs=[detail_html])
            export_button.click(export_csv, inputs=[raw_response], outputs=[download])

            if datasets:
                demo.load(
                    load_facets, inputs=[dataset],
                    outputs=[
                        event_category, split, video_id,
                        altitude_min, altitude_max, velocity_min, velocity_max, gimbal_min, gimbal_max,
                        facets_state, filter_badge_html, filter_note_html,
                        canonical_filters, is_night,
                    ],
                )
            demo.load(render_header, inputs=[dataset], outputs=[topbar_html, status_badge_html])

        with gr.Tab("Karşılaştır", elem_id="mvi-comparison"):
            compare_query = gr.Textbox(label="Sorgu")
            compare_dataset = gr.Dropdown(choices=datasets, value=datasets[0] if datasets else None, label="Dataset")
            compare_backends = gr.CheckboxGroup(CAPABILITIES["backends"], value=CAPABILITIES["backends"], label="Backend'ler")
            compare_dimensions = gr.CheckboxGroup(CAPABILITIES["dimensions"], value=CAPABILITIES["dimensions"], label="Boyutlar")
            compare_repeats = gr.Slider(1, 20, value=3, step=1, label="Tekrar")
            compare_button = gr.Button("Karşılaştır", variant="primary")
            compare_output = gr.HTML(components.empty_state("no_selection"))
            compare_button.click(
                run_compare,
                inputs=[compare_query, compare_dataset, compare_backends, compare_dimensions, compare_repeats],
                outputs=[compare_output],
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True, css=CSS)
