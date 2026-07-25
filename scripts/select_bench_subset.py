"""56 VisDrone-MOT sekansindan, Faz 1 bench subset'i icin temsili bir 15-20
sekanslik liste secer: trafik yogunlugu (kisi/arac sayisi), otobus varligi
ve gun/gece (ortalama kare parlakligi proxy'si) cesitliligini kapsar.

Tek seferlik analiz scripti - config.yaml: bench.subset listesini elle
doldurmak icin kanit uretir; secimi otomatik yazmaz (insan/agent son
listeyi config.yaml'a bilincli olarak koyar)."""
import pathlib
import statistics
import sys
from collections import defaultdict

import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

CAT = {"pedestrian": 1, "people": 2, "car": 4, "van": 5, "truck": 6, "bus": 9}


def load_stats(ann_path):
    per_frame_person = defaultdict(int)
    per_frame_car = defaultdict(int)
    has_bus = False
    has_truck = False
    n_frames = 0
    for line in open(ann_path):
        parts = line.strip().split(",")
        if len(parts) < 8:
            continue
        f, tid, x, y, w, h, score, cat = map(float, parts[:8])
        if score == 0:
            continue
        f = int(f)
        cat = int(cat)
        n_frames = max(n_frames, f)
        if cat in (CAT["pedestrian"], CAT["people"]):
            per_frame_person[f] += 1
        if cat in (CAT["car"], CAT["van"]):
            per_frame_car[f] += 1
        if cat == CAT["bus"]:
            has_bus = True
        if cat == CAT["truck"]:
            has_truck = True
    mean_person = statistics.mean(per_frame_person.values()) if per_frame_person else 0.0
    mean_car = statistics.mean(per_frame_car.values()) if per_frame_car else 0.0
    return {
        "n_frames": n_frames,
        "mean_person": mean_person,
        "mean_car": mean_car,
        "has_bus": has_bus,
        "has_truck": has_truck,
    }


def mean_brightness(seq_dir):
    frames = sorted(seq_dir.glob("*.jpg"))
    if not frames:
        return None
    mid = frames[len(frames) // 2]
    img = cv2.imread(str(mid))
    if img is None:
        return None
    return float(img.mean())


def main():
    repo = pathlib.Path(__file__).resolve().parents[1]
    root = repo / "data" / "raw" / "VisDrone2019-MOT-train"
    seq_dirs = sorted((root / "sequences").glob("*/"))

    rows = []
    for seq_dir in seq_dirs:
        name = seq_dir.name
        ann_path = root / "annotations" / f"{name}.txt"
        if not ann_path.exists():
            continue
        stats = load_stats(ann_path)
        stats["brightness"] = mean_brightness(seq_dir)
        stats["name"] = name
        rows.append(stats)

    rows.sort(key=lambda r: -(r["mean_person"] + r["mean_car"]))
    brightness_vals = [r["brightness"] for r in rows if r["brightness"] is not None]
    dark_thr = statistics.median(brightness_vals) if brightness_vals else None

    print(f"{'sekans':<22} {'kare':>5} {'kisi_ort':>9} {'arac_ort':>9} "
          f"{'bus':>4} {'truck':>6} {'parlaklik':>10}")
    for r in rows:
        print(f"{r['name']:<22} {r['n_frames']:>5} {r['mean_person']:>9.2f} "
              f"{r['mean_car']:>9.2f} {str(r['has_bus']):>4} {str(r['has_truck']):>6} "
              f"{r['brightness'] if r['brightness'] is not None else -1:>10.1f}")

    print(f"\nmedyan parlaklik (gun/gece esigi proxy'si): {dark_thr:.1f}" if dark_thr else "")
    n_dark = sum(1 for r in rows if dark_thr and r["brightness"] is not None and r["brightness"] < dark_thr)
    print(f"toplam sekans: {len(rows)}, medyan-alti (karanlik) sayisi: {n_dark}")


if __name__ == "__main__":
    main()
