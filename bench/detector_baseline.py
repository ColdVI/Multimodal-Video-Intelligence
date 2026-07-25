"""Faz 3: dedektor bake-off. window_features()'in ornekledigi AYNI kare
indekslerinde VisDrone annotation'indan gercek sayimi turetir (count MAE +
esik dogrulugu icin), ve model basina inference gecikmesini olcer.

window_features() ile ayni ornekleme semasi kullanilir: t =
np.linspace(t0, t1, n_sample, endpoint=False), kare indeksi int(t*fps)
(0-index, MP4); VisDrone annotation'i 1-index oldugu icin +1 uygulanir."""
import importlib.util
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.make_groundtruth import CAT, load_annotations

_DETECT_MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "ingest" / "04_detect.py"


def _load_detect_module():
    spec = importlib.util.spec_from_file_location("ingest_04_detect", _DETECT_MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# VisDrone annotation kategorisi -> bizim filtre kavramimiz. Detektor
# class_map'iyle (config.yaml: detector.variants.*.class_map) semantik
# olarak ayni esleme; GT tarafinda annotation kategori adiyla ifade edilir.
GT_CONCEPT_MAP = {
    "pedestrian": "person", "people": "person",
    "car": "car", "truck": "truck", "bus": "bus",
}


def gt_counts_for_window(frames: dict, t0: float, t1: float, fps: float, n_sample: int) -> dict:
    concepts = sorted(set(GT_CONCEPT_MAP.values()))
    per_frame = {c: [] for c in concepts}
    for t in np.linspace(t0, t1, n_sample, endpoint=False):
        frame_no = int(t * fps) + 1  # 0-index (mp4) -> 1-index (VisDrone)
        objs = frames.get(frame_no, [])
        frame_counts = {c: 0 for c in concepts}
        for _tid, cat_id, _cx, _cy in objs:
            for cat_name, cid in CAT.items():
                if cid == cat_id and cat_name in GT_CONCEPT_MAP:
                    frame_counts[GT_CONCEPT_MAP[cat_name]] += 1
        for c in concepts:
            per_frame[c].append(frame_counts[c])
    return {c: float(np.median(v)) if v else 0.0 for c, v in per_frame.items()}


def evaluate_variant(variant_name: str, windows: list, manifest: dict, ann_dir: pathlib.Path,
                     videos_dir: pathlib.Path, checkpoint: str, class_map: dict,
                     n_sample: int) -> dict:
    detect_mod = _load_detect_module()
    ann_cache = {}
    errors = defaultdict(list)
    threshold_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    latencies_s = []

    for w in windows:
        vid = w["video_id"]
        if vid not in manifest:
            continue
        fps = manifest[vid]["fps"]
        if vid not in ann_cache:
            ann_path = ann_dir / f"{vid}.txt"
            ann_cache[vid] = load_annotations(ann_path) if ann_path.exists() else {}
        frames = ann_cache[vid]

        video_path = videos_dir / f"{vid}.mp4"
        t0 = time.perf_counter()
        pred = detect_mod.window_features(
            str(video_path), w["t_start"], w["t_end"],
            n_sample=n_sample, checkpoint=checkpoint, class_map=class_map)
        latencies_s.append((time.perf_counter() - t0) / n_sample)

        gt = gt_counts_for_window(frames, w["t_start"], w["t_end"], fps, n_sample)
        for concept in ("person", "car", "bus", "truck"):
            pred_count = pred[f"{concept}_count"]
            gt_count = gt[concept]
            errors[concept].append(abs(pred_count - gt_count))

            pred_pos, gt_pos = pred_count >= 1, gt_count >= 1
            stats = threshold_stats[concept]
            if pred_pos and gt_pos:
                stats["tp"] += 1
            elif pred_pos and not gt_pos:
                stats["fp"] += 1
            elif not pred_pos and gt_pos:
                stats["fn"] += 1
            else:
                stats["tn"] += 1

    mae = {c: (sum(v) / len(v) if v else 0.0) for c, v in errors.items()}
    threshold_pr = {}
    for c, s in threshold_stats.items():
        precision = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else None
        recall = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else None
        threshold_pr[c] = {"precision": precision, "recall": recall, **s}

    return {
        "variant": variant_name,
        "n_windows": len(windows),
        "mae": mae,
        "threshold_pr": threshold_pr,
        "mean_inference_s_per_frame": (sum(latencies_s) / len(latencies_s)) if latencies_s else 0.0,
        "fps": (1.0 / (sum(latencies_s) / len(latencies_s))) if latencies_s else 0.0,
    }
