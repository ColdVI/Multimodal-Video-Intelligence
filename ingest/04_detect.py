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
from ingest.frame_io import read_frames_sequential, sample_frame_indices

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
                    class_map=None, frames=None):
    """frames verilirse (RGB ndarray listesi, kronolojik sirali) video hic
    acilmaz - cagiran taraf zaten decode etmis kareleri paylasiyor demektir
    (bkz. scripts/colab_gpu_bench.py: embedding icin okunan kareler
    dedektore de veriliyor, ayni videoyu iki kez decode etmemek icin).
    frames=None ise eskisi gibi video_path'ten TEK seek + sequential
    okuma ile n_sample kare cekilir (eskiden n_sample AYRI seek yapiliyordu -
    H.264'te her seek en yakin keyframe'e geri donup ileri decode ediyordu,
    bkz. docs/codex/06_NIHAI_RAPOR.md)."""
    class_map = class_map or COCO_MAP
    model = _get_model(checkpoint)
    concepts = sorted(set(class_map.values()))
    counts = {c: [] for c in concepts}
    brightness, motions = [], []
    prev_gray = None

    if frames is None:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        indices = sample_frame_indices(t0, t1, fps, n_sample)
        frame_map = read_frames_sequential(video_path, indices)
        sampled = [frame_map[i] for i in indices if i in frame_map]
    else:
        idx = np.linspace(0, len(frames) - 1, min(n_sample, len(frames))).astype(int)
        sampled = [frames[i] for i in idx]

    for frame in sampled:
        # frame_io RGB donduruyor (embedding modelleri - PIL/HF - RGB
        # bekliyor). Ultralytics ise ham numpy array'i HER ZAMAN BGR sanip
        # kendi icinde ters ceviriyor (engine/predictor.py: "im[..., ::-1]
        # # BGR to RGB") - RGB'yi oldugu gibi versek YOLO onu BGR sanip
        # tekrar cevirir, kanallar bozulur. O yuzden model'e vermeden once
        # BGR'ye geri ceviriyoruz; gri/parlaklik/optik akis icin bu adima
        # gerek yok (RGB2GRAY zaten dogru agirliklarla hesapliyor).
        res = model(frame[..., ::-1], verbose=False)[0]
        cls = res.boxes.cls.cpu().numpy().astype(int)
        frame_counts = {c: 0 for c in concepts}
        for cid, cname in class_map.items():
            frame_counts[cname] += int((cls == cid).sum())
        for cname, n in frame_counts.items():
            counts[cname].append(n)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        brightness.append(float(gray.mean()))
        small = cv2.resize(gray, (160, 90))
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            motions.append(float(np.linalg.norm(flow, axis=2).mean()))
        prev_gray = small

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
