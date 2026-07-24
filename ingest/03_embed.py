"""Pencereler + video -> embedding vektorleri. Model-agnostik: models/
registry'sindeki herhangi bir adapter kullanilabilir."""
import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_config
from models import available_models, get_embedder


def read_window(video_path, t0, t1, n=32):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    for t in np.linspace(t0, t1, n, endpoint=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=available_models())
    args = ap.parse_args()

    cfg = load_config()
    windows_path = pathlib.Path(cfg["paths"]["windows"])
    if not windows_path.exists():
        print(f"HATA: {windows_path} yok. Once 02_windowing.py calistirin.")
        raise SystemExit(1)

    emb = get_embedder(args.model)
    videos_dir = pathlib.Path(cfg["paths"]["videos_dir"])
    windows = json.load(open(windows_path))

    out = []
    for i, w in enumerate(windows):
        video_path = videos_dir / f"{w['video_id']}.mp4"
        frames = read_window(str(video_path), w["t_start"], w["t_end"])
        if not frames:
            continue
        vec = emb.embed_video(frames)
        out.append({**w, "embedding": vec.tolist()})
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(windows)}")

    out_path = f"data/embeddings_{emb.name}.json"
    json.dump(out, open(out_path, "w"))
    print(f"{len(out)} embedding -> {out_path}")


if __name__ == "__main__":
    main()
