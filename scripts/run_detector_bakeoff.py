"""Faz 3: dedektor bake-off'unu config.yaml: detector.variants'taki tum
varyantlar icin calistirir. count MAE, esik P/R, hiz - artifacts/
detector_bakeoff.json'a yazar."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bench.detector_baseline import evaluate_variant
from common import load_config


def main():
    cfg = load_config()
    windows = json.load(open(cfg["paths"]["windows"]))
    manifest = json.load(open(cfg["paths"]["manifest"]))
    ann_dir = pathlib.Path(cfg["paths"]["annotations_dir"])
    videos_dir = pathlib.Path(cfg["paths"]["videos_dir"])
    n_sample = cfg.get("detector", {}).get("n_sample", 6)

    results = []
    for name, spec in cfg["detector"]["variants"].items():
        print(f"--- {name} ({spec['checkpoint']}) ---")
        class_map = {int(k): v for k, v in spec["class_map"].items()}
        result = evaluate_variant(
            name, windows, manifest, ann_dir, videos_dir,
            checkpoint=spec["checkpoint"], class_map=class_map, n_sample=n_sample)
        results.append(result)
        print(f"  MAE: {result['mae']}")
        print(f"  fps (single-frame, CPU): {result['fps']:.2f}")
        for concept, pr in result["threshold_pr"].items():
            print(f"  {concept}: precision={pr['precision']} recall={pr['recall']} "
                 f"(tp={pr['tp']} fp={pr['fp']} fn={pr['fn']} tn={pr['tn']})")

    out_path = pathlib.Path("artifacts/detector_bakeoff.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON kanit: {out_path}")


if __name__ == "__main__":
    main()
