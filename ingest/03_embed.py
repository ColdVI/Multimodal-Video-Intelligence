"""Pencereler + video -> embedding vektorleri. Model-agnostik: models/
registry'sindeki herhangi bir adapter kullanilabilir."""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from bench.gpu_gate import require_gpu_for_qwen_windows
from common import load_config
from ingest.frame_io import read_window_frames
from models import available_models, get_embedder


def read_window(video_path, t0, t1, n=32):
    return read_window_frames(video_path, t0, t1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=available_models())
    args = ap.parse_args()

    cfg = load_config()
    windows_path = pathlib.Path(cfg["paths"]["windows"])
    if not windows_path.exists():
        print(f"HATA: {windows_path} yok. Once 02_windowing.py calistirin.")
        raise SystemExit(1)

    videos_dir = pathlib.Path(cfg["paths"]["videos_dir"])
    windows = json.load(open(windows_path))

    if args.model.startswith("qwen3vl_emb"):
        require_gpu_for_qwen_windows(f"ingest/03_embed.py --model {args.model}", len(windows))
    emb = get_embedder(args.model)

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
