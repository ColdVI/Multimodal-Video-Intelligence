"""results_detail.json'i FiftyOne'da inceler: sorgu + tahmin edilen ve
gercek zaman araliklari timeline'da yan yana."""
import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import configure_runtime_environment

configure_runtime_environment()


def interval_to_support(a, b, fps, n_frames):
    """[a,b) saniye araligini FiftyOne'in 1-index inclusive frame destegine cevir."""
    start = max(1, int(math.floor(a * fps)) + 1)
    end = min(n_frames, max(start, int(math.ceil(b * fps))))
    return [start, end]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-launch", action="store_true",
                    help="dataseti olustur/dogrula fakat tarayici acma")
    args = ap.parse_args()

    import fiftyone as fo

    detail = json.load(open("results_detail.json"))
    manifest = json.load(open("data/raw/manifest.json"))
    ds = fo.Dataset("poc-results", overwrite=True)

    for r in detail:
        meta = manifest[r["video_id"]]
        fps, n_frames = meta["fps"], meta["n_frames"]
        sample = fo.Sample(filepath=f"data/raw/videos/{r['video_id']}.mp4")
        sample["query"] = r["query"]
        sample["model"] = r["model"]
        sample["filter_on"] = r["filter"]
        sample["pred"] = fo.TemporalDetections(detections=[
            fo.TemporalDetection(label=f"pred ({score:.2f})",
                                  support=interval_to_support(
                                      a, b, fps, n_frames))
            for a, b, score in r["pred"]
        ])
        sample["gt"] = fo.TemporalDetections(detections=[
            fo.TemporalDetection(label="gt",
                                  support=interval_to_support(
                                      a, b, fps, n_frames))
            for a, b in r["gt"]
        ])
        ds.add_sample(sample)

    print(f"{len(ds)} FiftyOne sample hazir")
    if args.no_launch:
        return
    session = fo.launch_app(ds)
    session.wait()


if __name__ == "__main__":
    main()
