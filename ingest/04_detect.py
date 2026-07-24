"""YOLO tabanli filtre kolonu ureticisi - uretimdeki telemetri kolonlarinin
POC vekili. window_features() saf mantik; main() bunu tum pencereler icin
cagirip data/features.json'a yazar (onceki surumde bu adim eksikti)."""
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import configure_runtime_environment, load_config

configure_runtime_environment()

COCO_MAP = {0: "person", 2: "car", 5: "bus", 7: "truck"}
_model = None


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        try:
            _model = YOLO("yolo26x.pt")
        except Exception:
            print("uyari: yolo26x.pt bulunamadi, yolo11x.pt'ye dusuluyor")
            _model = YOLO("yolo11x.pt")
    return _model


def window_features(video_path, t0, t1, n_sample=6):
    model = _get_model()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    counts = {v: [] for v in COCO_MAP.values()}
    brightness, motions = [], []
    prev_gray = None

    for t in np.linspace(t0, t1, n_sample, endpoint=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        res = model(frame, verbose=False)[0]
        cls = res.boxes.cls.cpu().numpy().astype(int)
        for cid, cname in COCO_MAP.items():
            counts[cname].append(int((cls == cid).sum()))

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
        "person_count": int(np.median(counts["person"] or [0])),
        "car_count": int(np.median(counts["car"] or [0])),
        "bus_count": int(np.median(counts["bus"] or [0])),
        "truck_count": int(np.median(counts["truck"] or [0])),
        "brightness": float(np.mean(brightness or [0])),
        "is_night": bool(np.mean(brightness or [255]) < 60),
        "camera_motion": float(np.mean(motions or [0])),
    }


def main():
    cfg = load_config()
    windows_path = pathlib.Path(cfg["paths"]["windows"])
    if not windows_path.exists():
        print(f"HATA: {windows_path} yok. Once 02_windowing.py calistirin.")
        raise SystemExit(1)

    windows = json.load(open(windows_path))
    videos_dir = pathlib.Path(cfg["paths"]["videos_dir"])
    out = []
    for w in windows:
        video_path = videos_dir / f"{w['video_id']}.mp4"
        feats = window_features(str(video_path), w["t_start"], w["t_end"])
        out.append({**w, **feats})

    out_path = pathlib.Path(cfg["paths"]["features"])
    json.dump(out, open(out_path, "w"))
    print(f"{len(out)} ozellik satiri -> {out_path}")


if __name__ == "__main__":
    main()
