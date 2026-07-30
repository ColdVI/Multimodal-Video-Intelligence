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


API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
RESULT_COLUMNS = [
    "video_id", "t_start", "t_end", "score", "caption", "file_path",
    "altitude_m", "velocity_mps", "gimbal_pitch",
]


def _get(path: str) -> dict[str, Any]:
    with httpx.Client(timeout=30.0) as client:
        response = client.get(f"{API_URL}{path}")
        response.raise_for_status()
        return response.json()


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    with httpx.Client(timeout=300.0) as client:
        response = client.post(f"{API_URL}{path}", json=payload)
        response.raise_for_status()
        return response.json()


def _datasets() -> list[str]:
    try:
        return [row["dataset_id"] for row in _get("/stats").get("datasets", [])]
    except Exception:
        return []


def _range(lo: Any, hi: Any) -> list[float] | None:
    if lo is None or hi is None:
        return None
    return [float(lo), float(hi)]


def load_facets(dataset_id: str):
    if not dataset_id:
        empty = gr.update(choices=[], value=None)
        hidden = gr.update(visible=False)
        return empty, empty, empty, hidden, hidden, hidden, hidden, hidden, hidden, "Dataset yüklenmedi."
    data = _get(f"/facets/{dataset_id}")
    telemetry = data.get("telemetry", {})

    def sliders(name: str):
        bounds = telemetry.get(name)
        if not bounds:
            return gr.update(visible=False), gr.update(visible=False)
        lo, hi = bounds
        return (
            gr.update(minimum=lo, maximum=hi, value=lo, visible=True),
            gr.update(minimum=lo, maximum=hi, value=hi, visible=True),
        )

    has_telemetry = any(telemetry.values())
    altitude_updates = sliders("altitude_m")
    velocity_updates = sliders("velocity_mps")
    gimbal_updates = sliders("gimbal_pitch")
    return (
        gr.update(choices=data.get("event_categories", []), value=None),
        gr.update(choices=data.get("splits", []), value=None),
        gr.update(choices=data.get("video_ids", []), value=None),
        *altitude_updates, *velocity_updates, *gimbal_updates,
        "Gerçek min/max telemetri filtreleri etkin." if has_telemetry else "Bu dataset telemetri içermiyor.",
    )


def update_strategies(backend: str):
    choices = list(SUPPORTED_STRATEGIES.get(backend, ()))
    return gr.update(choices=choices, value=choices[0] if choices else None)


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
) -> dict[str, Any]:
    return {
        "query": query,
        "dataset_id": dataset_id,
        "backend": backend,
        "strategy": strategy,
        "dimension": int(dimension),
        "adaptive_mrl": {"enabled": adaptive, "base_dim": int(base_dim), "top_n": int(top_n)},
        "metadata_filters": {"event_category": event_category, "split": split, "video_id": video_id},
        "telemetry_filters": {
            "altitude_m": _range(altitude_min, altitude_max),
            "velocity_mps": _range(velocity_min, velocity_max),
            "gimbal_pitch": _range(gimbal_min, gimbal_max),
        },
        "pattern": pattern,
        "top_k": int(top_k),
        "repeats": int(repeats),
    }


def run_search(*args):
    try:
        response = _post("/search", _payload(*args))
        timings = response["timings_ms"]
        stats = response["timings_stats"]
        latency = {
            "filter_ms": timings["filter"], "embed_ms": timings["embed"],
            "vector_search_ms": timings["vector_search"], "hydrate_ms": timings["hydrate"],
            "total_ms": timings["total"], "p50_ms": stats["p50"], "p95_ms": stats["p95"],
        }
        rows = [{column: result.get(column) for column in RESULT_COLUMNS} for result in response.get("results", [])]
        frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
        return latency, response["diagnostics"], frame, response
    except Exception as exc:
        error = {"error": f"{type(exc).__name__}: {exc}"}
        return error, error, pd.DataFrame(columns=RESULT_COLUMNS), error


def export_csv(raw_response: dict[str, Any] | None):
    rows = (raw_response or {}).get("results", [])
    target = Path(tempfile.gettempdir()) / "faz7_search_results.csv"
    pd.DataFrame(rows).to_csv(target, index=False)
    return str(target)


def compare(query: str, dataset_id: str, backends: list[str], dimensions: list[int], repeats: int):
    rows = []
    for backend in backends or []:
        strategy = SUPPORTED_STRATEGIES[backend][0]
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
                })
            except Exception as exc:
                rows.append({"backend": backend, "strategy": strategy, "dimension": dimension, "error": str(exc)})
    return pd.DataFrame(rows)


datasets = _datasets()
details = mode_details(datasets[0] if datasets else None)
banner_class = "faz7-danger" if details["level"] == "danger" else "faz7-success"

CSS = """
.faz7-banner {padding: 16px; border-radius: 10px; font-size: 17px; font-weight: 800; text-align: center; margin-bottom: 12px;}
.faz7-danger {background: #7f1d1d; color: #fff; border: 2px solid #ef4444;}
.faz7-success {background: #14532d; color: #fff; border: 2px solid #22c55e;}
"""

with gr.Blocks(title="Multimodal Video Intelligence — Faz 7") as demo:
    gr.HTML(f'<div class="faz7-banner {banner_class}">{details["message"]}</div>')
    gr.Markdown("# Faz 7 · Çoklu backend video arama ve gecikme laboratuvarı")
    with gr.Tabs():
        with gr.Tab("Ara"):
            with gr.Row():
                with gr.Column(scale=2):
                    dataset = gr.Dropdown(choices=datasets, value=datasets[0] if datasets else None, label="Dataset")
                    with gr.Accordion("Metadata filtreleri", open=False):
                        event_category = gr.Dropdown(label="Event category")
                        split = gr.Dropdown(label="Split")
                        video_id = gr.Dropdown(label="Video ID")
                    with gr.Accordion("Telemetri filtreleri", open=False):
                        telemetry_note = gr.Markdown("Dataset seçildiğinde gerçek min/max yüklenir.")
                        with gr.Row():
                            altitude_min = gr.Slider(0, 1, label="İrtifa min (m)", visible=False)
                            altitude_max = gr.Slider(0, 1, label="İrtifa max (m)", visible=False)
                        with gr.Row():
                            velocity_min = gr.Slider(0, 1, label="Hız min (m/s)", visible=False)
                            velocity_max = gr.Slider(0, 1, label="Hız max (m/s)", visible=False)
                        with gr.Row():
                            gimbal_min = gr.Slider(0, 1, label="Gimbal pitch min", visible=False)
                            gimbal_max = gr.Slider(0, 1, label="Gimbal pitch max", visible=False)
                    with gr.Accordion("Arama yöntemi", open=True):
                        backend = gr.Radio(["clickhouse", "qdrant", "pgvector"], value="clickhouse", label="Backend")
                        strategy = gr.Dropdown(list(SUPPORTED_STRATEGIES["clickhouse"]), value="prefilter", label="Strategy")
                        dimension = gr.Radio([2048, 1024, 512, 256], value=512, label="Dimension")
                        adaptive = gr.Checkbox(False, label="Adaptive MRL")
                        base_dim = gr.Radio([256, 512], value=256, label="Base dimension")
                        top_n = gr.Slider(1, 200, value=100, step=1, label="Adaptive top_N")
                        pattern = gr.Radio(["A", "B", "C"], value="A", label="Pattern")
                    top_k = gr.Slider(1, 50, value=10, step=1, label="top_k")
                    repeats = gr.Slider(1, 20, value=1, step=1, label="Tekrar")
                    search_button = gr.Button("Search", variant="primary")
                    ten_button = gr.Button("10 tekrar")
                with gr.Column(scale=3):
                    query = gr.Textbox(label="Serbest metin sorgusu", value="kalabalık trafik")
                    gr.Markdown("### Gecikme paneli")
                    latency = gr.JSON(label="filter / embed / vector_search / hydrate / total / p50 / p95")
                    gr.Markdown("### Diagnostics")
                    diagnostics = gr.JSON()
                    results = gr.Dataframe(headers=RESULT_COLUMNS, interactive=False, label="Sonuçlar")
                    raw_response = gr.State()
                    export_button = gr.Button("Sonucu CSV indir")
                    download = gr.File(label="CSV")

            dataset.change(
                load_facets, inputs=[dataset],
                outputs=[event_category, split, video_id, altitude_min, altitude_max, velocity_min, velocity_max, gimbal_min, gimbal_max, telemetry_note],
            )
            backend.change(update_strategies, inputs=[backend], outputs=[strategy])
            search_inputs = [
                query, dataset, event_category, split, video_id,
                altitude_min, altitude_max, velocity_min, velocity_max, gimbal_min, gimbal_max,
                backend, strategy, dimension, adaptive, base_dim, top_n, pattern, top_k, repeats,
            ]
            search_outputs = [latency, diagnostics, results, raw_response]
            search_button.click(run_search, inputs=search_inputs, outputs=search_outputs)
            ten_button.click(lambda: 10, outputs=[repeats]).then(run_search, inputs=search_inputs, outputs=search_outputs)
            export_button.click(export_csv, inputs=[raw_response], outputs=[download])
            if datasets:
                demo.load(
                    load_facets, inputs=[dataset],
                    outputs=[event_category, split, video_id, altitude_min, altitude_max, velocity_min, velocity_max, gimbal_min, gimbal_max, telemetry_note],
                )

        with gr.Tab("Karşılaştır"):
            compare_query = gr.Textbox(label="Sorgu", value="kalabalık trafik")
            compare_dataset = gr.Dropdown(choices=datasets, value=datasets[0] if datasets else None, label="Dataset")
            compare_backends = gr.CheckboxGroup(["clickhouse", "qdrant", "pgvector"], value=["clickhouse", "qdrant", "pgvector"], label="Backend'ler")
            compare_dimensions = gr.CheckboxGroup([2048, 1024, 512, 256], value=[512, 256], label="Boyutlar")
            compare_repeats = gr.Slider(1, 20, value=3, step=1, label="Tekrar")
            compare_button = gr.Button("Karşılaştır", variant="primary")
            compare_table = gr.Dataframe(interactive=False)
            compare_button.click(
                compare,
                inputs=[compare_query, compare_dataset, compare_backends, compare_dimensions, compare_repeats],
                outputs=[compare_table],
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True, css=CSS)
