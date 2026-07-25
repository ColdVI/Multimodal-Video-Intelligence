"""YOLO tabanli filtre kolonu ureticisi - uretimdeki telemetri kolonlarinin
POC vekili. window_features() saf mantik; main() bunu tum pencereler icin
cagirip data/features.json'a yazar (onceki surumde bu adim eksikti)."""
import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import configure_runtime_environment, load_config

configure_runtime_environment()

# Geriye-uyum: variant verilmezse COCO sinif haritasi (eski COCO_MAP ile
# birebir ayni deger). Faz 3: dedektor varyantlari config.yaml: detector'de
# tanimlanir (checkpoint + class_map); yeni varyant eklemek kod degil
# config degisikligidir.
COCO_MAP = {0: "person", 2: "car", 5: "bus", 7: "truck"}
_models = {}


def _resolve_variant(cfg, variant: str = None):
    detector_cfg = cfg.get("detector", {})
    variant = variant or detector_cfg.get("default_variant")
    variants = detector_cfg.get("variants", {})
    if variant and variant in variants:
        spec = variants[variant]
        return spec["checkpoint"], {int(k): v for k, v in spec["class_map"].items()}
    return "yolo26x.pt", COCO_MAP


def _get_model(checkpoint: str):
    if checkpoint not in _models:
        from ultralytics import YOLO
        try:
            _models[checkpoint] = YOLO(checkpoint)
        except Exception:
            print(f"uyari: {checkpoint} bulunamadi, yolo11x.pt'ye dusuluyor")
            _models[checkpoint] = YOLO("yolo11x.pt")
    return _models[checkpoint]


def window_features(video_path, t0, t1, n_sample=6, checkpoint="yolo26x.pt",
                    class_map=None):
    class_map = class_map or COCO_MAP
    model = _get_model(checkpoint)
    concepts = sorted(set(class_map.values()))
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    counts = {c: [] for c in concepts}
    brightness, motions = [], []
    prev_gray = None

    for t in np.linspace(t0, t1, n_sample, endpoint=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        res = model(frame, verbose=False)[0]
        cls = res.boxes.cls.cpu().numpy().astype(int)
        frame_counts = {c: 0 for c in concepts}
        for cid, cname in class_map.items():
            frame_counts[cname] += int((cls == cid).sum())
        for cname, n in frame_counts.items():
            counts[cname].append(n)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness.append(float(gray.mean()))
        small = cv2.resize(gray, (160, 90))
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            motions.append(float(np.linalg.norm(flow, axis=2).mean()))
        prev_gray = small
    cap.release()

    return {
        "person_count": int(np.median(counts.get("person") or [0])),
        "car_count": int(np.median(counts.get("car") or [0])),
        "bus_count": int(np.median(counts.get("bus") or [0])),
        "truck_count": int(np.median(counts.get("truck") or [0])),
        "brightness": float(np.mean(brightness or [0])),
        "is_night": bool(np.mean(brightness or [255]) < 60),
        "camera_motion": float(np.mean(motions or [0])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None,
                    help="config.yaml: detector.variants altindaki bir ad; "
                         "verilmezse detector.default_variant kullanilir")
    args = ap.parse_args()

    cfg = load_config()
    windows_path = pathlib.Path(cfg["paths"]["windows"])
    if not windows_path.exists():
        print(f"HATA: {windows_path} yok. Once 02_windowing.py calistirin.")
        raise SystemExit(1)

    checkpoint, class_map = _resolve_variant(cfg, args.variant)
    n_sample = cfg.get("detector", {}).get("n_sample", 6)

    windows = json.load(open(windows_path))
    videos_dir = pathlib.Path(cfg["paths"]["videos_dir"])
    out = []
    for w in windows:
        video_path = videos_dir / f"{w['video_id']}.mp4"
        feats = window_features(str(video_path), w["t_start"], w["t_end"],
                                n_sample=n_sample, checkpoint=checkpoint,
                                class_map=class_map)
        out.append({**w, **feats})

    out_path = pathlib.Path(cfg["paths"]["features"])
    json.dump(out, open(out_path, "w"))
    print(f"{len(out)} ozellik satiri -> {out_path} (variant={args.variant or cfg.get('detector', {}).get('default_variant')})")


if __name__ == "__main__":
    main()
