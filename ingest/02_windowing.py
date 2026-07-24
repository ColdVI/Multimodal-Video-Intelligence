"""Videolari 8s/4s (config.yaml) kayan pencerelere boler -> data/windows.json"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_config


def main():
    cfg = load_config()
    win = cfg["window"]["size_s"]
    stride = cfg["window"]["stride_s"]
    min_win = cfg["window"]["min_window_s"]

    manifest_path = pathlib.Path(cfg["paths"]["manifest"])
    if not manifest_path.exists():
        print(f"HATA: {manifest_path} yok. Once 01_frames_to_video.py calistirin.")
        raise SystemExit(1)

    manifest = json.load(open(manifest_path))
    windows = []
    for vid, m in manifest.items():
        t = 0.0
        while t < m["duration_s"]:
            t_end = min(t + win, m["duration_s"])
            if t_end - t >= min_win:
                windows.append({"video_id": vid, "t_start": round(t, 2),
                                 "t_end": round(t_end, 2)})
            t += stride

    out_path = pathlib.Path(cfg["paths"]["windows"])
    json.dump(windows, open(out_path, "w"))
    print(f"{len(windows)} pencere -> {out_path}")


if __name__ == "__main__":
    main()
