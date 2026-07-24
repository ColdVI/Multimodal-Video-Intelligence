"""Google Colab icin butonlu VideoSearch POC arayuzu.

Bu modul ClickHouse'un yerini uretim iddiasi ile almaya calismaz. Colab'de
model/filtre kalitesini gorunur kilmak icin ayni embedding ve deterministik
filtreleri exact, bellek-ici cosine aramayla calistirir. Uretilen rapor backend
turunu ve metodolojik sinirlari acikca yazar.
"""
from __future__ import annotations

import base64
import csv
import datetime as dt
import html
import json
import math
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable

import numpy as np


MODEL_LABELS = {
    "xclip_hf_zeroshot": "Microsoft X-CLIP (512d)",
    "siglip2_frameavg": "SigLIP2 frame-average (1152d)",
}

# Ilk besli, mevcut smoke kosusunda nesne/sorgu kapsami icin secilen sekanslar.
REPORT_SEQUENCE_IDS = [
    "uav0000013_01073_v",
    "uav0000072_04488_v",
    "uav0000266_04830_v",
    "uav0000138_00000_v",
    "uav0000361_02323_v",
]

# Raporluk Colab paketi tam 8 GB veriyi tasimadan ayni gercek sekanslar
# uzerinde boru hattini calistirir. Bunlar sentetik veri degildir; resmi
# VisDrone-MOT train setinin dogrulanmis bir alt kumesidir.
SMOKE_FRAME_COUNTS = {
    "uav0000013_01073_v": 58,
    "uav0000072_04488_v": 85,
    "uav0000266_04830_v": 116,
    "uav0000138_00000_v": 213,
    "uav0000361_02323_v": 219,
}
SMOKE_TOTAL_FRAMES = sum(SMOKE_FRAME_COUNTS.values())

FILTER_LABELS = {
    "person_count": "insan",
    "car_count": "araba",
    "bus_count": "otobus",
    "truck_count": "kamyon",
    "is_night": "gece",
}

BACKEND_NAME = "exact_in_memory_cosine"


@dataclass
class QueryRun:
    query: str
    model: str
    use_filters: bool
    filters: list
    semantic: str
    top_k: int
    total_windows: int
    filtered_windows: int
    elapsed_ms: float
    results: list[dict] = field(default_factory=list)


def _read_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: pathlib.Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def format_seconds(value: float) -> str:
    value = max(0, int(round(value)))
    hours, rem = divmod(value, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def choose_sequences(all_names: Iterable[str], count: int) -> list[str]:
    """Raporluk besliyi once al, kalanlari alfabetik tamamla."""
    names = sorted(set(all_names))
    if count < 1:
        raise ValueError("sekans sayisi en az 1 olmali")
    preferred = [name for name in REPORT_SEQUENCE_IDS if name in names]
    remaining = [name for name in names if name not in preferred]
    return (preferred + remaining)[: min(count, len(names))]


def validate_report_subset(root: pathlib.Path) -> tuple[int, int, int]:
    """Cikarilmis 5-sekanslik resmi VisDrone alt kumesini dogrula."""
    root = pathlib.Path(root)
    sequence_root = root / "sequences"
    annotation_root = root / "annotations"
    actual_sequences = {path.name for path in sequence_root.glob("*/")}
    actual_annotations = {path.stem for path in annotation_root.glob("*.txt")}
    expected = set(SMOKE_FRAME_COUNTS)
    if actual_sequences != expected or actual_annotations != expected:
        raise RuntimeError(
            "smoke veri adlari yanlis: "
            f"sequences={sorted(actual_sequences)} annotations={sorted(actual_annotations)}"
        )
    actual_counts = {
        name: sum(1 for _ in (sequence_root / name).glob("*.jpg"))
        for name in expected
    }
    if actual_counts != SMOKE_FRAME_COUNTS:
        raise RuntimeError(
            f"smoke kare sayilari yanlis: {actual_counts} != {SMOKE_FRAME_COUNTS}"
        )
    return len(expected), len(expected), sum(actual_counts.values())


def validate_report_archive(path: pathlib.Path) -> tuple[int, int, int]:
    """5-sekanslik Colab ZIP'inin yol guvenligi ve veri sozlesmesi."""
    from scripts.download_visdrone import DATASET_NAME

    frame_counts = {name: 0 for name in SMOKE_FRAME_COUNTS}
    annotations = set()
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            member = pathlib.PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise RuntimeError(f"guvensiz ZIP yolu: {info.filename}")
            if info.is_dir():
                continue
            parts = member.parts
            if (
                len(parts) == 4
                and parts[0] == DATASET_NAME
                and parts[1] == "sequences"
                and parts[2] in frame_counts
                and member.suffix.lower() == ".jpg"
            ):
                frame_counts[parts[2]] += 1
            elif (
                len(parts) == 3
                and parts[0] == DATASET_NAME
                and parts[1] == "annotations"
                and member.suffix.lower() == ".txt"
                and member.stem in frame_counts
            ):
                annotations.add(member.stem)
            else:
                raise RuntimeError(f"smoke ZIP'inde beklenmeyen dosya: {info.filename}")
        damaged = archive.testzip()
        if damaged:
            raise RuntimeError(f"smoke ZIP CRC hatasi: {damaged}")
    if frame_counts != SMOKE_FRAME_COUNTS or annotations != set(SMOKE_FRAME_COUNTS):
        raise RuntimeError(
            f"smoke ZIP sozlesmesi yanlis: frames={frame_counts}, annotations={sorted(annotations)}"
        )
    return len(frame_counts), len(annotations), sum(frame_counts.values())


def load_records(repo_root: pathlib.Path, model_name: str) -> list[dict]:
    repo_root = pathlib.Path(repo_root)
    features_path = repo_root / "data" / "features.json"
    embeddings_path = repo_root / "data" / f"embeddings_{model_name}.json"
    if not features_path.exists():
        raise FileNotFoundError("data/features.json yok; once pipeline'i calistirin")
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"{embeddings_path.name} yok; {model_name} embedding adimini calistirin"
        )

    embeddings = {
        (item["video_id"], float(item["t_start"])): item["embedding"]
        for item in _read_json(embeddings_path)
    }
    records = []
    for feature in _read_json(features_path):
        key = (feature["video_id"], float(feature["t_start"]))
        embedding = embeddings.get(key)
        if embedding is None:
            continue
        record = dict(feature)
        record["embedding"] = embedding
        records.append(record)
    if not records:
        raise RuntimeError("feature ve embedding kayitlari eslesmedi")
    return records


def record_matches(record: dict, filters: Iterable[tuple]) -> bool:
    for column, operator, expected in filters:
        actual = record[column]
        if operator == ">=" and not actual >= expected:
            return False
        if operator == "<=" and not actual <= expected:
            return False
        if operator == ">" and not actual > expected:
            return False
        if operator == "<" and not actual < expected:
            return False
        if operator == "=" and not actual == expected:
            return False
        if operator not in {">=", "<=", ">", "<", "="}:
            raise ValueError(f"desteklenmeyen operator: {operator}")
    return True


def rank_records(
    records: list[dict], query_vector, filters=(), top_k: int = 200
) -> list[dict]:
    """Normalize edilmis olsun/olmasin exact cosine siralamasi."""
    if top_k < 1:
        raise ValueError("top_k en az 1 olmali")
    selected = [record for record in records if record_matches(record, filters)]
    if not selected:
        return []

    matrix = np.asarray([record["embedding"] for record in selected], dtype=np.float32)
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != query.shape[0]:
        raise ValueError(
            f"embedding boyutu uyusmuyor: rows={matrix.shape}, query={query.shape}"
        )
    matrix_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    query_norm = float(np.linalg.norm(query))
    matrix = matrix / np.maximum(matrix_norm, 1e-12)
    query = query / max(query_norm, 1e-12)
    distances = 1.0 - matrix @ query
    order = np.argsort(distances, kind="stable")[:top_k]

    ranked = []
    for index in order:
        item = dict(selected[int(index)])
        item.pop("embedding", None)
        item["distance"] = float(distances[int(index)])
        item["score"] = float(1.0 - distances[int(index)])
        ranked.append(item)
    return ranked


def _merge_ranked(ranked: list[dict], gap_tolerance: float) -> list[dict]:
    """Pencereleri video bazinda birlestir, en iyi pencerenin metadata'sini tut."""
    from search.merge import merge_intervals

    raw = [
        (row["video_id"], row["t_start"], row["t_end"], row["distance"])
        for row in ranked
    ]
    merged = merge_intervals(raw, gap_tol=gap_tolerance)
    results = []
    for rank, (video_id, t_start, t_end, score) in enumerate(merged, 1):
        candidates = [
            row
            for row in ranked
            if row["video_id"] == video_id
            and row["t_start"] < t_end
            and row["t_end"] > t_start
        ]
        best = max(candidates, key=lambda row: row["score"]) if candidates else {}
        results.append(
            {
                "rank": rank,
                "video_id": video_id,
                "t_start": float(t_start),
                "t_end": float(t_end),
                "score": float(score),
                "person_count": int(best.get("person_count", 0)),
                "car_count": int(best.get("car_count", 0)),
                "bus_count": int(best.get("bus_count", 0)),
                "truck_count": int(best.get("truck_count", 0)),
                "is_night": bool(best.get("is_night", False)),
                "camera_motion": float(best.get("camera_motion", 0.0)),
                "brightness": float(best.get("brightness", 0.0)),
            }
        )
    return results


class InMemorySearchSession:
    """Bir modeli bir kez yukleyip birden cok sorguda kullanan Colab oturumu."""

    def __init__(self, repo_root: pathlib.Path, model_name: str):
        self.repo_root = pathlib.Path(repo_root)
        self.model_name = model_name
        self.records = load_records(self.repo_root, model_name)
        from models import get_embedder

        self.embedder = get_embedder(model_name)

    def search(self, query: str, top_k: int = 10, use_filters: bool = True) -> QueryRun:
        from common import load_config
        from search.parser import parse

        started = time.perf_counter()
        parsed = parse(query)
        filters = parsed.filters if use_filters else []
        query_vector = self.embedder.embed_text(parsed.semantic)
        filtered_count = sum(record_matches(row, filters) for row in self.records)
        # Once genis aday havuzu, sonra interval merge, en son arayuz top-k.
        candidate_limit = min(len(self.records), max(200, top_k))
        ranked = rank_records(
            self.records, query_vector, filters=filters, top_k=candidate_limit
        )
        cfg = load_config()
        merged = _merge_ranked(ranked, cfg["merge"]["gap_tolerance_s"])
        elapsed_ms = (time.perf_counter() - started) * 1000
        return QueryRun(
            query=query,
            model=self.model_name,
            use_filters=use_filters,
            filters=[list(value) for value in filters],
            semantic=parsed.semantic,
            top_k=top_k,
            total_windows=len(self.records),
            filtered_windows=filtered_count,
            elapsed_ms=elapsed_ms,
            results=merged[:top_k],
        )


def evaluate_models(
    repo_root: pathlib.Path,
    model_names: Iterable[str],
    top_k: int,
    progress: Callable[[int, int, str], None] | None = None,
) -> list[dict]:
    from common import load_config
    from eval.metrics import evaluate
    from eval.run_eval import category_of

    repo_root = pathlib.Path(repo_root)
    gt_path = repo_root / load_config()["paths"]["groundtruth"]
    if not gt_path.exists():
        raise FileNotFoundError("ground truth yok; pipeline'i calistirin")
    groundtruth = _read_json(gt_path)
    models = list(model_names)
    total = len(models) * 2 * len(groundtruth)
    done = 0
    rows = []
    cfg = load_config()
    for model_name in models:
        session = InMemorySearchSession(repo_root, model_name)
        for use_filters in (True, False):
            for query, gt_by_video in groundtruth.items():
                run = session.search(query, top_k=top_k, use_filters=use_filters)
                pred = [
                    (r["video_id"], r["t_start"], r["t_end"], r["score"])
                    for r in run.results
                ]
                metrics = evaluate(
                    pred,
                    gt_by_video,
                    k=top_k,
                    iou_thr=cfg["eval"]["iou_threshold"],
                )
                rows.append(
                    {
                        "model": model_name,
                        "filter": use_filters,
                        "query": query,
                        "category": category_of(query),
                        "latency_ms": round(run.elapsed_ms, 2),
                        **metrics,
                    }
                )
                done += 1
                if progress:
                    progress(done, total, f"{MODEL_LABELS.get(model_name, model_name)} · {query}")
    return rows


def aggregate_metrics(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["model"], bool(row["filter"])), []).append(row)
    output = []
    for (model, use_filter), values in sorted(groups.items()):
        output.append(
            {
                "model": model,
                "filter": use_filter,
                "queries": len(values),
                "mean_precision@k": float(
                    np.mean([row["precision@k"] for row in values])
                ),
                "mean_recall@k": float(np.mean([row["recall@k"] for row in values])),
                "total_hits": int(sum(row["n_hits"] for row in values)),
                "total_gt": int(sum(row["n_gt"] for row in values)),
                "mean_latency_ms": float(np.mean([row["latency_ms"] for row in values])),
            }
        )
    return output


def report_warnings(metrics: list[dict], manifest: dict | None, top_k: int) -> list[str]:
    warnings = []
    query_count = len({row["query"] for row in metrics})
    if query_count < 30:
        warnings.append(
            f"Yalnizca {query_count} sorgu var; sonuc kesifseldir, istatistiksel model zaferi olarak sunulamaz."
        )
    selected_videos = int((manifest or {}).get("selected_videos", 0))
    if selected_videos and top_k >= selected_videos:
        warnings.append(
            f"top_k={top_k}, video sayisi={selected_videos}; adaylar doygunlasabilir ve model farklari metrikte gorunmeyebilir."
        )
    warnings.append(
        "Colab aramasi exact bellek-ici cosine kullanir; bu rapor ClickHouse gecikme benchmark'i degildir."
    )
    warnings.append(
        "Yurume ground truth'u piksel yer degistirmesinden turetilir; drone ego-motion yanlis pozitif uretebilir."
    )
    return warnings


def report_findings(metrics: list[dict]) -> list[str]:
    if not metrics:
        return ["Accuracy henuz calistirilmadi."]
    aggregates = aggregate_metrics(metrics)
    findings = []
    for row in aggregates:
        filter_label = "filtre acik" if row["filter"] else "filtre kapali"
        findings.append(
            f"{MODEL_LABELS.get(row['model'], row['model'])}, {filter_label}: "
            f"ortalama P@k={row['mean_precision@k']:.3f}, "
            f"R@k={row['mean_recall@k']:.3f} ({row['queries']} sorgu)."
        )

    by_model: dict[str, dict[bool, dict]] = {}
    for row in aggregates:
        by_model.setdefault(row["model"], {})[row["filter"]] = row
    for model, modes in by_model.items():
        if True in modes and False in modes:
            p_delta = modes[True]["mean_precision@k"] - modes[False]["mean_precision@k"]
            r_delta = modes[True]["mean_recall@k"] - modes[False]["mean_recall@k"]
            findings.append(
                f"{MODEL_LABELS.get(model, model)} icin filtre etkisi: "
                f"Δprecision={p_delta:+.3f}, Δrecall={r_delta:+.3f}."
            )
    return findings


def hardware_info() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "bilinmiyor",
        "cuda_available": False,
        "gpu": "GPU yok",
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 2)
    except Exception as exc:  # pragma: no cover - ortama bagli
        info["torch_error"] = str(exc)
    return info


def _run_command(
    args: list[str],
    cwd: pathlib.Path,
    on_line: Callable[[str], None] | None = None,
) -> None:
    env = os.environ.copy()
    env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    process = subprocess.Popen(
        args,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert process.stdout is not None
    tail = []
    for line in process.stdout:
        clean = line.rstrip()
        tail.append(clean)
        tail = tail[-30:]
        if on_line:
            on_line(clean)
    code = process.wait()
    if code:
        raise RuntimeError("adim basarisiz:\n" + "\n".join(tail))


def prepare_dataset(
    repo_root: pathlib.Path,
    archive_path: str | None = None,
    download_if_missing: bool = False,
    status: Callable[[str], None] | None = None,
) -> tuple[int, int, int]:
    """Mevcut veri, Drive ZIP'i veya resmi indirme ile VisDrone'u hazirla."""
    from scripts.download_visdrone import (
        DATASET_NAME,
        URL,
        validate_archive,
        validate_dataset,
    )

    repo_root = pathlib.Path(repo_root)
    dataset = repo_root / "data" / "raw" / DATASET_NAME

    def validate_available_dataset():
        try:
            return validate_dataset(dataset)
        except RuntimeError as full_error:
            try:
                return validate_report_subset(dataset)
            except RuntimeError as smoke_error:
                raise RuntimeError(
                    f"veri ne tam set ne raporluk smoke seti: {full_error}; {smoke_error}"
                ) from smoke_error

    if dataset.exists():
        counts = validate_available_dataset()
        if status:
            status(f"Veri hazir: {counts[0]} sekans, {counts[2]:,} kare")
        return counts

    source = pathlib.Path(archive_path).expanduser() if archive_path else None
    if source and source.exists():
        if status:
            status("ZIP boyutu ve SHA-256 dogrulaniyor…")
        try:
            validate_archive(source)
        except RuntimeError:
            validate_report_archive(source)
    elif download_if_missing:
        import gdown

        target = repo_root / "data" / "downloads" / f"{DATASET_NAME}.zip"
        target.parent.mkdir(parents=True, exist_ok=True)
        if status:
            status("Resmi Google Drive baglantisindan indiriliyor…")
        result = gdown.download(URL, str(target), quiet=False, resume=True)
        if not result:
            raise RuntimeError("Google Drive indirmesi tamamlanamadi")
        source = target
        validate_archive(source)
    else:
        raise FileNotFoundError(
            "VisDrone ZIP'i bulunamadi. Drive ZIP yolunu girin veya resmi indirmeyi secin."
        )

    if status:
        status("ZIP guvenli bicimde aciliyor…")
    raw_dir = repo_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        archive.extractall(raw_dir)
    counts = validate_available_dataset()
    if status:
        status(f"Veri hazir: {counts[0]} sekans, {counts[2]:,} kare")
    return counts


def run_pipeline(
    repo_root: pathlib.Path,
    sequence_count: int,
    model_names: Iterable[str],
    progress: Callable[[int, int, str, str], None] | None = None,
) -> dict:
    """Mevcut repo scriptlerini Colab UI arkasinda sirayla calistir."""
    repo_root = pathlib.Path(repo_root).resolve()
    sequence_root = repo_root / "data" / "raw" / "VisDrone2019-MOT-train" / "sequences"
    if not sequence_root.exists():
        raise FileNotFoundError("VisDrone verisi hazir degil")
    all_names = [path.name for path in sequence_root.iterdir() if path.is_dir()]
    if sequence_count > len(all_names):
        raise ValueError(
            f"Bu veri paketinde {len(all_names)} sekans var; kapsam olarak en fazla bunu secin."
        )
    selected = choose_sequences(all_names, sequence_count)
    models = list(model_names)
    if not models:
        raise ValueError("en az bir model secin")

    stages = [
        ("Videolar hazirlaniyor", [
            sys.executable,
            "ingest/01_frames_to_video.py",
            *[value for name in selected for value in ("--sequence", name)],
        ]),
        ("Pencereler olusturuluyor", [sys.executable, "ingest/02_windowing.py"]),
        ("YOLO filtre alanlari uretiliyor", [sys.executable, "ingest/04_detect.py"]),
    ]
    for model_name in models:
        stages.append(
            (
                f"Embedding: {MODEL_LABELS.get(model_name, model_name)}",
                [sys.executable, "ingest/03_embed.py", "--model", model_name],
            )
        )
    stages.append(("Ground truth uretiliyor", [sys.executable, "eval/make_groundtruth.py"]))

    durations = {}
    total = len(stages)
    last_line = ""
    for index, (label, args) in enumerate(stages, 1):
        if progress:
            progress(index - 1, total, label, "basladi")
        started = time.perf_counter()

        def on_line(line: str) -> None:
            nonlocal last_line
            last_line = line
            if progress:
                progress(index - 1, total, label, line)

        _run_command(args, repo_root, on_line=on_line)
        durations[label] = round(time.perf_counter() - started, 3)
        if progress:
            progress(index, total, label, last_line or "tamamlandi")

    windows = _read_json(repo_root / "data" / "windows.json")
    manifest = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "backend": BACKEND_NAME,
        "dataset": "VisDrone2019-MOT-train",
        "selected_videos": len(selected),
        "selected_sequence_ids": selected,
        "windows": len(windows),
        "models": models,
        "stage_seconds": durations,
        "total_seconds": round(sum(durations.values()), 3),
        "hardware": hardware_info(),
    }
    _write_json(repo_root / "artifacts" / "run_manifest.json", manifest)
    return manifest


def _preview_data_uri(video_path: pathlib.Path, t_start: float, t_end: float) -> str | None:
    """Sonuc araligindan uc karelik yatay onizleme."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    for value in np.linspace(t_start, max(t_start, t_end - 0.04), 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(value * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        height, width = frame.shape[:2]
        target_width = 320
        frame = cv2.resize(frame, (target_width, max(1, int(height * target_width / width))))
        cv2.putText(
            frame,
            f"{value:.1f}s",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        frames.append(frame)
    cap.release()
    if not frames:
        return None
    sheet = np.hstack(frames)
    ok, encoded = cv2.imencode(".jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def query_run_html(repo_root: pathlib.Path, run: QueryRun) -> str:
    filter_html = "".join(
        f'<span class="chip">{html.escape(FILTER_LABELS.get(c, c))} {html.escape(str(op))} {html.escape(str(v))}</span>'
        for c, op, v in run.filters
    ) or '<span class="chip muted">yapilandirilmis filtre yok</span>'
    cards = []
    videos_dir = pathlib.Path(repo_root) / "data" / "raw" / "videos"
    for item in run.results:
        preview = _preview_data_uri(
            videos_dir / f"{item['video_id']}.mp4", item["t_start"], item["t_end"]
        )
        image = f'<img src="{preview}" />' if preview else '<div class="noimg">onizleme yok</div>'
        cards.append(
            f"""
            <article class="result-card">
              <div class="rank">#{item['rank']}</div>
              {image}
              <div class="result-body">
                <strong>{html.escape(item['video_id'])}</strong>
                <div>{item['t_start']:.2f}s – {item['t_end']:.2f}s · skor {item['score']:.3f}</div>
                <small>insan {item['person_count']} · araba {item['car_count']} · otobüs {item['bus_count']} · kamyon {item['truck_count']}</small>
              </div>
            </article>"""
        )
    if not cards:
        cards.append('<div class="empty">Bu sorgu/filtre için sonuç bulunamadı.</div>')
    return f"""
    <style>
      .vs-wrap {{font-family:Inter,Arial,sans-serif;color:#172033}}
      .query-meta {{background:#f3f6fb;border:1px solid #dce4f2;border-radius:12px;padding:14px;margin:8px 0 14px}}
      .chip {{display:inline-block;background:#e4edff;color:#174a9c;padding:5px 9px;border-radius:999px;margin:4px;font-size:12px}}
      .muted {{background:#eceff3;color:#5e6673}}
      .result-card {{border:1px solid #dce4f2;border-radius:14px;margin:12px 0;overflow:hidden;background:white;position:relative}}
      .result-card img {{width:100%;display:block;background:#111}}
      .result-body {{padding:12px 16px;line-height:1.55}}
      .rank {{position:absolute;top:8px;left:8px;background:#172033;color:white;border-radius:8px;padding:4px 8px;z-index:2}}
      .empty,.noimg {{padding:28px;text-align:center;color:#6c7480;background:#f5f6f8}}
    </style>
    <div class="vs-wrap">
      <div class="query-meta">
        <strong>Sorgu:</strong> {html.escape(run.query)}<br/>
        <strong>Model:</strong> {html.escape(MODEL_LABELS.get(run.model, run.model))} ·
        <strong>Backend:</strong> exact in-memory cosine ·
        <strong>Süre:</strong> {run.elapsed_ms:.1f} ms<br/>
        <strong>Aday:</strong> {run.total_windows} → {run.filtered_windows} pencere
        <div>{filter_html}</div>
      </div>
      {''.join(cards)}
    </div>"""


def build_report_html(
    title: str,
    author: str,
    notes: str,
    manifest: dict | None,
    metrics: list[dict],
    query_history: list[dict],
    top_k: int,
) -> str:
    aggregates = aggregate_metrics(metrics) if metrics else []
    warnings = report_warnings(metrics, manifest, top_k)
    findings = report_findings(metrics)

    def rows_html(rows, columns):
        body = []
        for row in rows:
            cells = []
            for key, label in columns:
                value = row.get(key, "")
                if isinstance(value, float):
                    value = f"{value:.3f}"
                cells.append(f"<td>{html.escape(str(value))}</td>")
            body.append("<tr>" + "".join(cells) + "</tr>")
        head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
        return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"

    aggregate_table = rows_html(
        aggregates,
        [
            ("model", "Model"),
            ("filter", "Filtre"),
            ("queries", "Sorgu"),
            ("mean_precision@k", "Ort. P@k"),
            ("mean_recall@k", "Ort. R@k"),
            ("total_hits", "Hit"),
            ("total_gt", "GT"),
            ("mean_latency_ms", "Ort. ms"),
        ],
    )
    detail_table = rows_html(
        metrics,
        [
            ("model", "Model"),
            ("filter", "Filtre"),
            ("query", "Sorgu"),
            ("category", "Kategori"),
            ("precision@k", "P@k"),
            ("recall@k", "R@k"),
            ("n_hits", "Hit"),
            ("n_gt", "GT"),
        ],
    )
    manifest_json = html.escape(json.dumps(manifest or {}, ensure_ascii=False, indent=2))
    return f"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{html.escape(title)}</title>
<style>
body{{font-family:Inter,Arial,sans-serif;max-width:1180px;margin:38px auto;padding:0 24px;color:#172033;line-height:1.55}}
h1{{margin-bottom:4px}} .sub{{color:#667085}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}
.card{{border:1px solid #dde4ef;border-radius:14px;padding:16px;background:#fff}} .warn{{border-left:5px solid #f2a900;background:#fff9e8}}
table{{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow:auto}} th,td{{border-bottom:1px solid #e7ebf1;padding:9px;text-align:left;white-space:nowrap}} th{{background:#f3f6fb}}
code,pre{{background:#f3f6fb;border-radius:8px;padding:12px;overflow:auto}} .good{{color:#067647}}
</style></head><body>
<h1>{html.escape(title)}</h1><div class="sub">Hazırlayan: {html.escape(author or '—')} · {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
<p>{html.escape(notes)}</p>
<h2>Koşu özeti</h2><div class="grid">
<div class="card"><b>Backend</b><br>{html.escape((manifest or {}).get('backend', BACKEND_NAME))}</div>
<div class="card"><b>Video</b><br>{(manifest or {}).get('selected_videos', '—')}</div>
<div class="card"><b>Pencere</b><br>{(manifest or {}).get('windows', '—')}</div>
<div class="card"><b>Top-k</b><br>{top_k}</div></div>
<h2>Rapor cümleleri</h2><ul>{''.join(f'<li>{html.escape(item)}</li>' for item in findings)}</ul>
<h2>Metodolojik uyarılar</h2>{''.join(f'<div class="card warn">{html.escape(item)}</div>' for item in warnings)}
<h2>Toplu accuracy</h2>{aggregate_table or '<p>Henüz hesaplanmadı.</p>'}
<h2>Sorgu bazlı accuracy</h2>{detail_table or '<p>Henüz hesaplanmadı.</p>'}
<h2>İncelenen serbest sorgular</h2><p>{len(query_history)} sorgu çalıştırıldı. Ayrıntılar paketteki <code>query_history.json</code> dosyasındadır.</p>
<h2>Tekrarlanabilirlik manifesti</h2><pre>{manifest_json}</pre>
</body></html>"""


def export_report_bundle(
    repo_root: pathlib.Path,
    title: str,
    author: str,
    notes: str,
    manifest: dict | None,
    metrics: list[dict],
    query_history: list[dict],
    top_k: int,
) -> pathlib.Path:
    repo_root = pathlib.Path(repo_root)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = repo_root / "artifacts" / f"report_{stamp}"
    report_dir.mkdir(parents=True, exist_ok=False)
    (report_dir / "report.html").write_text(
        build_report_html(title, author, notes, manifest, metrics, query_history, top_k),
        encoding="utf-8",
    )
    _write_json(report_dir / "run_manifest.json", manifest or {})
    _write_json(report_dir / "metrics.json", metrics)
    _write_json(report_dir / "query_history.json", query_history)
    for name in ("clickhouse_search_report.html", "clickhouse_search_report.json"):
        source = repo_root / "artifacts" / name
        if source.exists():
            shutil.copy2(source, report_dir / name)

    if metrics:
        with (report_dir / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
    with (report_dir / "README.txt").open("w", encoding="utf-8") as handle:
        handle.write(
            "report.html dosyasini tarayicida acin.\n"
            "Bu paket Colab exact in-memory arama sonucudur; ClickHouse latency benchmark'i degildir.\n"
        )

    archive = report_dir.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in report_dir.rglob("*"):
            if path.is_file():
                bundle.write(path, path.relative_to(report_dir.parent))
    return archive


def create_colab_app(repo_root: str | pathlib.Path):
    """Ipywidgets tabanli uygulamayi kur ve goster."""
    import pandas as pd
    import ipywidgets as widgets
    from IPython.display import HTML, clear_output, display

    repo_root = pathlib.Path(repo_root).resolve()
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    state = {
        "manifest": None,
        "metrics": [],
        "query_history": [],
        "sessions": {},
        "top_k": 10,
    }
    manifest_path = repo_root / "artifacts" / "run_manifest.json"
    if manifest_path.exists():
        state["manifest"] = _read_json(manifest_path)

    header = widgets.HTML(
        """<div style="padding:22px;border-radius:18px;background:linear-gradient(120deg,#102a56,#2458a6);color:white">
        <h2 style="margin:0 0 5px">VideoSearch · Colab Control Room</h2>
        <div>GPU pipeline, görsel sorgu sonuçları, accuracy ve rapor paketi</div></div>"""
    )

    # Veri / GPU sekmesi
    hw = hardware_info()
    gpu_color = "#067647" if hw["cuda_available"] else "#b42318"
    hardware_card = widgets.HTML(
        f"""<div style="padding:14px;border:1px solid #dde4ef;border-radius:12px">
        <b>Çalışma ortamı</b><br>GPU: <span style="color:{gpu_color}">{html.escape(str(hw['gpu']))}</span><br>
        CUDA: {hw['cuda_available']} · Python {hw['python']} · Torch {html.escape(str(hw.get('torch','—')))}</div>"""
    )
    archive_path = widgets.Text(
        description="ZIP yolu",
        placeholder="/content/drive/MyDrive/VideoSearch/VisDrone2019-MOT-train.zip",
        layout=widgets.Layout(width="95%"),
        style={"description_width": "90px"},
    )
    official_download = widgets.Checkbox(value=False, description="ZIP yoksa resmi bağlantıdan indir")
    prepare_button = widgets.Button(description="Veriyi doğrula / hazırla", button_style="primary", icon="database")
    drive_button = widgets.Button(description="Google Drive'a bağlan", icon="folder-open")
    data_status = widgets.HTML("<span style='color:#667085'>Veri henüz doğrulanmadı.</span>")
    data_output = widgets.Output()

    def mount_drive(_):
        with data_output:
            clear_output(wait=True)
            try:
                from google.colab import drive

                drive.mount("/content/drive")
                display(HTML("<b style='color:#067647'>Google Drive bağlandı.</b>"))
            except Exception as exc:
                display(HTML(f"<b style='color:#b42318'>{html.escape(str(exc))}</b>"))

    def prepare_data(_):
        prepare_button.disabled = True
        data_status.value = "<b>Kontrol ediliyor…</b>"
        with data_output:
            clear_output(wait=True)
            try:
                counts = prepare_dataset(
                    repo_root,
                    archive_path=archive_path.value.strip() or None,
                    download_if_missing=official_download.value,
                    status=lambda msg: setattr(data_status, "value", f"<b>{html.escape(msg)}</b>"),
                )
                display(
                    HTML(
                        f"<div style='color:#067647'>✓ {counts[0]} sekans, {counts[1]} annotation, {counts[2]:,} kare doğrulandı.</div>"
                    )
                )
            except Exception as exc:
                data_status.value = "<b style='color:#b42318'>Veri hazırlama başarısız.</b>"
                display(HTML(f"<pre>{html.escape(str(exc))}</pre>"))
            finally:
                prepare_button.disabled = False

    drive_button.on_click(mount_drive)
    prepare_button.on_click(prepare_data)
    data_tab = widgets.VBox(
        [hardware_card, widgets.HTML("<h3>VisDrone-MOT train</h3>"), archive_path,
         official_download, widgets.HBox([drive_button, prepare_button]), data_status, data_output]
    )

    # Pipeline sekmesi
    sequence_count = widgets.SelectionSlider(
        options=[("Raporluk 5", 5), ("10 video", 10), ("20 video", 20), ("Tam set 56", 56)],
        value=5,
        description="Kapsam",
        continuous_update=False,
        style={"description_width": "70px"},
        layout=widgets.Layout(width="90%"),
    )
    model_select = widgets.SelectMultiple(
        options=[(label, name) for name, label in MODEL_LABELS.items()],
        value=tuple(MODEL_LABELS),
        description="Modeller",
        style={"description_width": "70px"},
        layout=widgets.Layout(width="90%", height="90px"),
    )
    run_button = widgets.Button(description="GPU pipeline'i çalıştır", button_style="success", icon="play")
    pipeline_progress = widgets.IntProgress(value=0, min=0, max=100, description="İlerleme")
    pipeline_stage = widgets.HTML("Hazır")
    pipeline_output = widgets.Output()

    def run_clicked(_):
        run_button.disabled = True
        pipeline_progress.value = 0
        with pipeline_output:
            clear_output(wait=True)
        try:
            def update(done, total, stage, line):
                pipeline_progress.value = int(100 * done / max(total, 1))
                pipeline_stage.value = (
                    f"<b>{html.escape(stage)}</b><br><small>{html.escape(line[-240:])}</small>"
                )

            manifest = run_pipeline(
                repo_root,
                sequence_count=sequence_count.value,
                model_names=model_select.value,
                progress=update,
            )
            state["manifest"] = manifest
            state["sessions"].clear()
            pipeline_progress.value = 100
            pipeline_stage.value = (
                f"<b style='color:#067647'>✓ Tamamlandı:</b> {manifest['selected_videos']} video, "
                f"{manifest['windows']} pencere, {format_seconds(manifest['total_seconds'])}"
            )
            with pipeline_output:
                display(pd.DataFrame(
                    [{"adım": key, "süre_sn": value} for key, value in manifest["stage_seconds"].items()]
                ))
        except Exception as exc:
            pipeline_stage.value = "<b style='color:#b42318'>Pipeline başarısız.</b>"
            with pipeline_output:
                display(HTML(f"<pre>{html.escape(str(exc))}</pre>"))
        finally:
            run_button.disabled = False

    run_button.on_click(run_clicked)
    pipeline_tab = widgets.VBox(
        [widgets.HTML("<h3>GPU inference pipeline</h3><p>Komut yazmadan kapsamı ve modelleri seçin.</p>"),
         sequence_count, model_select, run_button, pipeline_progress, pipeline_stage, pipeline_output]
    )

    # Sorgu sekmesi
    query_presets = widgets.Dropdown(
        options=[
            "otobüsü göster", "kamyonu göster", "arabaları göster",
            "yürüyen adamı göster", "otobüs ve yürüyen adam", "kamyon ve yaya birlikte",
        ],
        description="Örnek",
        style={"description_width": "70px"},
        layout=widgets.Layout(width="80%"),
    )
    query_text = widgets.Textarea(
        value=query_presets.value,
        description="Sorgu",
        style={"description_width": "70px"},
        layout=widgets.Layout(width="90%", height="70px"),
    )
    query_presets.observe(lambda change: setattr(query_text, "value", change["new"]), names="value")
    query_model = widgets.Dropdown(
        options=[(label, name) for name, label in MODEL_LABELS.items()],
        description="Model",
        style={"description_width": "70px"},
        layout=widgets.Layout(width="80%"),
    )
    query_filters = widgets.Checkbox(value=True, description="Yapılandırılmış filtreler açık")
    query_topk = widgets.IntSlider(value=10, min=1, max=50, step=1, description="Top-k")
    search_button = widgets.Button(description="Ara ve sonuçları göster", button_style="primary", icon="search")
    query_output = widgets.Output()

    def search_clicked(_):
        search_button.disabled = True
        with query_output:
            clear_output(wait=True)
            display(HTML("<b>Model ve sorgu hazırlanıyor…</b>"))
            try:
                key = query_model.value
                session = state["sessions"].get(key)
                if session is None:
                    session = InMemorySearchSession(repo_root, key)
                    state["sessions"][key] = session
                run = session.search(
                    query_text.value.strip(), top_k=query_topk.value, use_filters=query_filters.value
                )
                state["top_k"] = query_topk.value
                state["query_history"].append(asdict(run))
                clear_output(wait=True)
                display(HTML(query_run_html(repo_root, run)))
            except Exception as exc:
                clear_output(wait=True)
                display(HTML(f"<pre>{html.escape(str(exc))}</pre>"))
            finally:
                search_button.disabled = False

    search_button.on_click(search_clicked)
    query_tab = widgets.VBox(
        [widgets.HTML("<h3>Görsel sorgu laboratuvarı</h3>"), query_presets, query_text,
         query_model, widgets.HBox([query_filters, query_topk]), search_button, query_output]
    )

    # Accuracy sekmesi
    eval_models = widgets.SelectMultiple(
        options=[(label, name) for name, label in MODEL_LABELS.items()],
        value=tuple(MODEL_LABELS),
        description="Modeller",
        style={"description_width": "70px"},
        layout=widgets.Layout(width="90%", height="90px"),
    )
    eval_topk = widgets.IntSlider(value=10, min=1, max=50, description="Top-k")
    eval_button = widgets.Button(description="Accuracy hesapla", button_style="warning", icon="bar-chart")
    eval_progress = widgets.IntProgress(value=0, min=0, max=100, description="İlerleme")
    eval_status = widgets.HTML("Hazır")
    eval_output = widgets.Output()

    def eval_clicked(_):
        eval_button.disabled = True
        eval_progress.value = 0
        with eval_output:
            clear_output(wait=True)
        try:
            def update(done, total, label):
                eval_progress.value = int(100 * done / max(total, 1))
                eval_status.value = html.escape(label)

            metrics = evaluate_models(
                repo_root, eval_models.value, top_k=eval_topk.value, progress=update
            )
            state["metrics"] = metrics
            state["top_k"] = eval_topk.value
            aggregates = aggregate_metrics(metrics)
            eval_progress.value = 100
            eval_status.value = "<b style='color:#067647'>✓ Accuracy hazır</b>"
            with eval_output:
                display(HTML("<h4>Toplu sonuç</h4>"))
                aggregate_df = pd.DataFrame(aggregates)
                display(aggregate_df.style.format({
                    "mean_precision@k": "{:.3f}", "mean_recall@k": "{:.3f}",
                    "mean_latency_ms": "{:.1f}",
                }))
                try:
                    import matplotlib.pyplot as plt

                    plot_df = aggregate_df.copy()
                    plot_df["deney"] = plot_df.apply(
                        lambda row: f"{row['model']}\nfiltre={'açık' if row['filter'] else 'kapalı'}", axis=1
                    )
                    ax = plot_df.set_index("deney")[["mean_precision@k", "mean_recall@k"]].plot(
                        kind="bar", figsize=(11, 4), ylim=(0, 1.05), rot=20,
                        color=["#2458a6", "#37a07f"]
                    )
                    ax.set_ylabel("Skor")
                    ax.set_title("Model × filtre accuracy karşılaştırması")
                    ax.grid(axis="y", alpha=.2)
                    plt.tight_layout()
                    plt.show()
                except Exception as plot_exc:
                    display(HTML(f"<small>Grafik üretilemedi: {html.escape(str(plot_exc))}</small>"))
                display(HTML("<h4>Sorgu bazlı sonuç</h4>"))
                display(pd.DataFrame(metrics))
                warnings = report_warnings(metrics, state["manifest"], eval_topk.value)
                display(HTML("".join(
                    f"<div style='padding:10px;margin:6px 0;background:#fff4d6;border-left:4px solid #e5a000'>{html.escape(item)}</div>"
                    for item in warnings
                )))
        except Exception as exc:
            eval_status.value = "<b style='color:#b42318'>Accuracy başarısız.</b>"
            with eval_output:
                display(HTML(f"<pre>{html.escape(str(exc))}</pre>"))
        finally:
            eval_button.disabled = False

    eval_button.on_click(eval_clicked)
    accuracy_tab = widgets.VBox(
        [widgets.HTML("<h3>Accuracy dashboard</h3><p>Otomatik VisDrone ground truth ile model × filtre ölçümü.</p>"),
         eval_models, eval_topk, eval_button, eval_progress, eval_status, eval_output]
    )

    # Rapor sekmesi
    report_title = widgets.Text(value="Hybrid Video Search POC — Colab Evaluation", description="Başlık", layout=widgets.Layout(width="90%"), style={"description_width": "70px"})
    report_author = widgets.Text(value="", description="Hazırlayan", layout=widgets.Layout(width="90%"), style={"description_width": "70px"})
    report_notes = widgets.Textarea(value="VisDrone-MOT üzerinde GPU destekli POC koşusu.", description="Not", layout=widgets.Layout(width="90%", height="80px"), style={"description_width": "70px"})
    drive_report_dir = widgets.Text(value="/content/drive/MyDrive/VideoSearch/reports", description="Drive", layout=widgets.Layout(width="90%"), style={"description_width": "70px"})
    copy_to_drive = widgets.Checkbox(value=True, description="Rapor ZIP'ini Drive'a kopyala")
    export_button = widgets.Button(description="Rapor paketini üret", button_style="info", icon="download")
    report_output = widgets.Output()

    def export_clicked(_):
        export_button.disabled = True
        with report_output:
            clear_output(wait=True)
            try:
                archive = export_report_bundle(
                    repo_root,
                    report_title.value,
                    report_author.value,
                    report_notes.value,
                    state["manifest"],
                    state["metrics"],
                    state["query_history"],
                    state["top_k"],
                )
                drive_copy = None
                if copy_to_drive.value:
                    destination = pathlib.Path(drive_report_dir.value)
                    if destination.parent.exists() or pathlib.Path("/content/drive").exists():
                        destination.mkdir(parents=True, exist_ok=True)
                        drive_copy = destination / archive.name
                        shutil.copy2(archive, drive_copy)
                display(HTML(
                    f"<div style='color:#067647'><b>✓ Rapor hazır:</b> {html.escape(str(archive))}</div>"
                    + (f"<div>Drive kopyası: {html.escape(str(drive_copy))}</div>" if drive_copy else "")
                ))
                try:
                    from google.colab import files

                    files.download(str(archive))
                except Exception:
                    from IPython.display import FileLink

                    display(FileLink(str(archive)))
            except Exception as exc:
                display(HTML(f"<pre>{html.escape(str(exc))}</pre>"))
            finally:
                export_button.disabled = False

    export_button.on_click(export_clicked)
    report_tab = widgets.VBox(
        [widgets.HTML("<h3>Rapor paketi</h3><p>HTML yönetici özeti + CSV/JSON kanıt dosyaları + koşu manifesti.</p>"),
         report_title, report_author, report_notes, drive_report_dir, copy_to_drive,
         export_button, report_output]
    )

    tabs = widgets.Tab(children=[data_tab, pipeline_tab, query_tab, accuracy_tab, report_tab])
    for index, title in enumerate(["1 · Veri/GPU", "2 · Pipeline", "3 · Query", "4 · Accuracy", "5 · Rapor"]):
        tabs.set_title(index, title)
    display(header)
    display(tabs)
    return tabs, state


def create_gradio_app(repo_root: str | pathlib.Path):
    """Colab'de ipywidgets yerine guvenilir bicimde render olan Gradio UI."""
    import gradio as gr
    import pandas as pd

    repo_root = pathlib.Path(repo_root).resolve()
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    state = {
        "manifest": None,
        "metrics": [],
        "query_history": [],
        "sessions": {},
        "top_k": 10,
    }
    manifest_path = repo_root / "artifacts" / "run_manifest.json"
    if manifest_path.exists():
        state["manifest"] = _read_json(manifest_path)

    def prepare_ui(uploaded_zip, drive_zip, official_download, progress=gr.Progress()):
        source = uploaded_zip or (drive_zip.strip() if drive_zip else None)
        progress(0.05, desc="Veri kaynagi kontrol ediliyor")

        def status(message):
            progress(0.35, desc=message)

        counts = prepare_dataset(
            repo_root,
            archive_path=source,
            download_if_missing=official_download,
            status=status,
        )
        progress(1.0, desc="VisDrone hazir")
        return (
            f"### ✅ Veri hazır\n\n"
            f"- Sekans: **{counts[0]}**\n"
            f"- Annotation: **{counts[1]}**\n"
            f"- Kare: **{counts[2]:,}**"
        )

    def pipeline_ui(scope, models, progress=gr.Progress()):
        if not models:
            raise gr.Error("En az bir model seçin.")

        def update(done, total, stage, line):
            progress(done / max(total, 1), desc=f"{stage}: {line[-90:]}")

        manifest = run_pipeline(
            repo_root,
            sequence_count=int(scope),
            model_names=models,
            progress=update,
        )
        state["manifest"] = manifest
        state["sessions"].clear()
        timing = pd.DataFrame(
            [{"Adım": key, "Süre (sn)": value} for key, value in manifest["stage_seconds"].items()]
        )
        status = (
            f"### ✅ Pipeline tamamlandı\n\n"
            f"**{manifest['selected_videos']} video · {manifest['windows']} pencere · "
            f"{format_seconds(manifest['total_seconds'])}**"
        )
        return status, timing, manifest

    def search_ui(query, model_name, use_filters, top_k, progress=gr.Progress()):
        if not query or not query.strip():
            raise gr.Error("Bir sorgu yazın.")
        progress(0.1, desc="Model ve embedding hazirlaniyor")
        session = state["sessions"].get(model_name)
        if session is None:
            session = InMemorySearchSession(repo_root, model_name)
            state["sessions"][model_name] = session
        progress(0.65, desc="Exact cosine arama calisiyor")
        run = session.search(query.strip(), top_k=int(top_k), use_filters=use_filters)
        state["top_k"] = int(top_k)
        state["query_history"].append(asdict(run))
        progress(1.0, desc="Sonuclar hazir")
        return query_run_html(repo_root, run)

    def accuracy_ui(models, top_k, progress=gr.Progress()):
        if not models:
            raise gr.Error("En az bir model seçin.")

        def update(done, total, label):
            progress(done / max(total, 1), desc=label)

        metrics = evaluate_models(repo_root, models, int(top_k), progress=update)
        state["metrics"] = metrics
        state["top_k"] = int(top_k)
        aggregates = aggregate_metrics(metrics)
        aggregate_df = pd.DataFrame(aggregates)
        detail_df = pd.DataFrame(metrics)

        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(11, 4.5))
        plot_df = aggregate_df.copy()
        plot_df["Deney"] = plot_df.apply(
            lambda row: f"{MODEL_LABELS.get(row['model'], row['model'])}\n"
                        f"filtre={'açık' if row['filter'] else 'kapalı'}",
            axis=1,
        )
        positions = np.arange(len(plot_df))
        width = 0.36
        axis.bar(
            positions - width / 2,
            plot_df["mean_precision@k"],
            width,
            label="Precision@k",
            color="#2458a6",
        )
        axis.bar(
            positions + width / 2,
            plot_df["mean_recall@k"],
            width,
            label="Recall@k",
            color="#37a07f",
        )
        axis.set_xticks(positions, plot_df["Deney"], rotation=15, ha="right")
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Skor")
        axis.set_title("Model × filtre accuracy karşılaştırması")
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
        figure.tight_layout()

        warnings = report_warnings(metrics, state["manifest"], int(top_k))
        warning_md = "### Metodolojik uyarılar\n\n" + "\n".join(
            f"- ⚠️ {item}" for item in warnings
        )
        return aggregate_df, detail_df, figure, warning_md

    def report_ui(title, author, notes, copy_to_drive, drive_dir):
        archive = export_report_bundle(
            repo_root,
            title,
            author,
            notes,
            state["manifest"],
            state["metrics"],
            state["query_history"],
            state["top_k"],
        )
        message = f"### ✅ Rapor hazır\n\n`{archive.name}`"
        if copy_to_drive:
            destination = pathlib.Path(drive_dir)
            if not pathlib.Path("/content/drive").exists():
                message += "\n\n⚠️ Drive bağlı değil; rapor aşağıdaki dosya düğmesinden indirilebilir."
            else:
                destination.mkdir(parents=True, exist_ok=True)
                drive_copy = destination / archive.name
                shutil.copy2(archive, drive_copy)
                message += f"\n\nDrive kopyası: `{drive_copy}`"
        return message, str(archive)

    hardware = hardware_info()
    gpu_state = "✅" if hardware["cuda_available"] else "❌"
    hardware_md = (
        f"### Çalışma ortamı\n\n"
        f"- GPU: **{hardware['gpu']}** {gpu_state}\n"
        f"- CUDA: **{hardware['cuda_available']}**\n"
        f"- Python: **{hardware['python']}**\n"
        f"- Torch: **{hardware.get('torch', '—')}**"
    )
    model_choices = [(label, name) for name, label in MODEL_LABELS.items()]

    with gr.Blocks(title="VideoSearch Colab Control Room") as demo:
        gr.HTML(
            "<div style='background:linear-gradient(120deg,#102a56,#2458a6);color:white;"
            "border-radius:18px;padding:22px 26px;margin-bottom:12px'>"
            "<h1 style='margin:0 0 5px;color:white'>VideoSearch · Colab Control Room</h1>"
            "<div>GPU pipeline · görsel sorgu sonuçları · accuracy · rapor paketi</div></div>"
        )
        gr.Markdown(
            "Sekmeleri soldan sağa kullanın: **Veri/GPU → Pipeline → Query → Accuracy → Rapor**"
        )
        with gr.Tabs():
            with gr.Tab("1 · Veri/GPU"):
                gr.Markdown(hardware_md)
                gr.Markdown(
                    "Raporluk 5-sekans ZIP'ini veya tam VisDrone ZIP'ini yükleyin; "
                    "isterseniz Drive yolunu girin ya da resmi bağlantıdan indirin. "
                    "Veri zaten hazırsa düğme yalnızca sözleşmeyi doğrular."
                )
                uploaded_zip = gr.File(
                    label="VisDrone smoke-5 veya tam train ZIP'i (isteğe bağlı)", type="filepath"
                )
                drive_zip = gr.Textbox(
                    label="Drive / Colab ZIP yolu (isteğe bağlı)",
                    placeholder="/content/drive/MyDrive/VideoSearch/VisDrone2019-MOT-train.zip",
                )
                official_download = gr.Checkbox(
                    value=True, label="ZIP bulunamazsa resmi Google Drive bağlantısından indir"
                )
                prepare_button = gr.Button("Veriyi doğrula / hazırla", variant="primary")
                data_status = gr.Markdown("Veri henüz doğrulanmadı.")
                prepare_button.click(
                    prepare_ui,
                    inputs=[uploaded_zip, drive_zip, official_download],
                    outputs=data_status,
                )

            with gr.Tab("2 · Pipeline"):
                scope = gr.Dropdown(
                    choices=[("Raporluk 5", 5), ("10 video", 10), ("20 video", 20), ("Tam set 56", 56)],
                    value=5,
                    label="Kapsam",
                )
                pipeline_models = gr.CheckboxGroup(
                    choices=model_choices,
                    value=list(MODEL_LABELS),
                    label="Embedding modelleri",
                )
                pipeline_button = gr.Button("GPU pipeline'i çalıştır", variant="primary")
                pipeline_status = gr.Markdown("Hazır.")
                timing_table = gr.Dataframe(label="Adım süreleri", interactive=False)
                run_manifest = gr.JSON(label="Tekrarlanabilirlik manifesti")
                pipeline_button.click(
                    pipeline_ui,
                    inputs=[scope, pipeline_models],
                    outputs=[pipeline_status, timing_table, run_manifest],
                )

            with gr.Tab("3 · Query"):
                query = gr.Textbox(
                    value="otobüsü göster",
                    label="Türkçe sorgu",
                    lines=2,
                    placeholder="Örn. kamyon ve yaya birlikte",
                )
                with gr.Row():
                    query_model = gr.Dropdown(
                        choices=model_choices,
                        value="xclip_hf_zeroshot",
                        label="Model",
                    )
                    query_filters = gr.Checkbox(value=True, label="Yapılandırılmış filtreler açık")
                    query_top_k = gr.Slider(1, 50, value=10, step=1, label="Top-k")
                search_button = gr.Button("Ara ve görsel sonuçları göster", variant="primary")
                search_results = gr.HTML()
                search_button.click(
                    search_ui,
                    inputs=[query, query_model, query_filters, query_top_k],
                    outputs=search_results,
                )

            with gr.Tab("4 · Accuracy"):
                accuracy_models = gr.CheckboxGroup(
                    choices=model_choices,
                    value=list(MODEL_LABELS),
                    label="Modeller",
                )
                accuracy_top_k = gr.Slider(1, 50, value=10, step=1, label="Top-k")
                accuracy_button = gr.Button("Accuracy hesapla", variant="primary")
                aggregate_table = gr.Dataframe(label="Toplu sonuç", interactive=False)
                detail_table = gr.Dataframe(label="Sorgu bazlı sonuç", interactive=False)
                accuracy_plot = gr.Plot(label="Model × filtre karşılaştırması")
                accuracy_warnings = gr.Markdown()
                accuracy_button.click(
                    accuracy_ui,
                    inputs=[accuracy_models, accuracy_top_k],
                    outputs=[aggregate_table, detail_table, accuracy_plot, accuracy_warnings],
                )

            with gr.Tab("5 · Rapor"):
                report_title = gr.Textbox(
                    value="Hybrid Video Search POC — Colab Evaluation", label="Başlık"
                )
                report_author = gr.Textbox(label="Hazırlayan")
                report_notes = gr.Textbox(
                    value="VisDrone-MOT üzerinde GPU destekli POC koşusu.",
                    label="Rapor notu",
                    lines=3,
                )
                report_drive = gr.Textbox(
                    value="/content/drive/MyDrive/VideoSearch/reports", label="Drive rapor klasörü"
                )
                report_copy = gr.Checkbox(value=False, label="Drive'a da kopyala")
                report_button = gr.Button("Rapor paketini üret", variant="primary")
                report_status = gr.Markdown()
                report_file = gr.File(label="Rapor ZIP'i")
                report_button.click(
                    report_ui,
                    inputs=[report_title, report_author, report_notes, report_copy, report_drive],
                    outputs=[report_status, report_file],
                )
    return demo


__all__ = [
    "BACKEND_NAME",
    "InMemorySearchSession",
    "QueryRun",
    "aggregate_metrics",
    "build_report_html",
    "choose_sequences",
    "create_colab_app",
    "create_gradio_app",
    "evaluate_models",
    "export_report_bundle",
    "load_records",
    "prepare_dataset",
    "rank_records",
    "record_matches",
    "report_findings",
    "report_warnings",
    "run_pipeline",
]
