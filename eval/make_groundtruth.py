"""VisDrone MOT anotasyonlarindan, sorgu bazli ground-truth zaman araliklari
uretir. Manuel etiketleme YOK - sinir kutusu + track ID'den turetilir."""
import json
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_config

CAT = {"pedestrian": 1, "people": 2, "car": 4, "van": 5, "truck": 6, "bus": 9}
WALK_PX_PER_S = 15.0  # track merkez yer degistirme esigi; 5-10 sekans gozle
                      # kalibre edilmeden uretime guvenilmemeli (bkz. CONTEXT.md)


def load_annotations(path):
    """VisDrone MOT satiri: frame,track_id,x,y,w,h,score,cat,trunc,occl
    Donus: {frame_no: [(track_id, cat, cx, cy), ...]}"""
    frames = defaultdict(list)
    for line in open(path):
        parts = line.strip().split(",")
        if len(parts) < 8:
            continue
        f, tid, x, y, w, h, score, cat = map(float, parts[:8])
        if score == 0:  # ignore bolgesi
            continue
        frames[int(f)].append((int(tid), int(cat), x + w / 2, y + h / 2))
    return frames


def frames_to_intervals(flags, fps, min_dur=1.0, gap_tol_s=2.0):
    """[kare basina bool] -> birlesik (t0, t1) araliklari."""
    gap_frames = int(gap_tol_s * fps)
    intervals = []
    start = None
    last_true = -10 ** 9
    for i, flag in enumerate(flags):
        if flag:
            if start is None:
                start = i
            last_true = i
        elif start is not None and i - last_true > gap_frames:
            # last_true'nun kendisi de bir kare islgal eder: kare i, zaman
            # [i/fps, (i+1)/fps) araligini kapsar. +1 olmadan N ardisik True
            # kare (N-1)/fps sureye dusuyor - N=fps icin tam 1.0sn yerine
            # 0.96sn cikiyor, bu da min_dur esigini yanlislikla kirpiyordu
            # (gercek pytest calistirmasinda yakalandi).
            intervals.append((start / fps, (last_true + 1) / fps))
            start = None
    if start is not None:
        intervals.append((start / fps, (last_true + 1) / fps))
    return [(a, b) for a, b in intervals if b - a >= min_dur]


def gt_object(frames, n_frames, cat_name, fps):
    cid = CAT[cat_name]
    flags = [any(c == cid for _, c, _, _ in frames.get(i, []))
             for i in range(1, n_frames + 1)]
    return frames_to_intervals(flags, fps)


def gt_walking(frames, n_frames, fps, px_per_s=WALK_PX_PER_S):
    """Yaya track'inin ~1sn penceredeki merkez yer degistirmesi esik ustu mu.
    NOT: kamera hareketinden arindirilmamis (ego-motion). Duran bir yaya,
    drone hareket ederken piksel uzayinda kayabilir -> yanlis pozitif riski.
    Uretime almadan once 5-10 sekansi FiftyOne'da gozle dogrulayin."""
    flags = [False] * n_frames
    tracks = defaultdict(dict)
    for f, objs in frames.items():
        for tid, cat, cx, cy in objs:
            if cat == CAT["pedestrian"]:
                tracks[tid][f] = (cx, cy)
    step = int(round(fps))
    for tid, pos in tracks.items():
        for f in pos:
            f2 = f + step
            if f2 in pos:
                dx = pos[f2][0] - pos[f][0]
                dy = pos[f2][1] - pos[f][1]
                if (dx * dx + dy * dy) ** 0.5 >= px_per_s:
                    for k in range(f, min(f2, n_frames)):
                        flags[k] = True
    return frames_to_intervals(flags, fps)


def intersect(iv_a, iv_b, min_overlap=1.0):
    out = []
    for a0, a1 in iv_a:
        for b0, b1 in iv_b:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi - lo >= min_overlap:
                out.append((lo, hi))
    return out


def build_queries():
    """Sorgu -> (frames, n_frames, fps) -> [(t0,t1),...] fonksiyonu.
    Genelden ozele kademeli zorluk: tekli -> hareket -> bilesik."""
    return {
        "otobüsü göster": lambda F, N, fps: gt_object(F, N, "bus", fps),
        "kamyonu göster": lambda F, N, fps: gt_object(F, N, "truck", fps),
        "arabaları göster": lambda F, N, fps: gt_object(F, N, "car", fps),
        "yürüyen adamı göster": lambda F, N, fps: gt_walking(F, N, fps),
        "otobüs ve yürüyen adam": lambda F, N, fps: intersect(
            gt_object(F, N, "bus", fps), gt_walking(F, N, fps)),
        "kamyon ve yaya birlikte": lambda F, N, fps: intersect(
            gt_object(F, N, "truck", fps), gt_object(F, N, "pedestrian", fps)),
    }


def main():
    cfg = load_config()
    manifest = json.load(open(cfg["paths"]["manifest"]))
    ann_dir = pathlib.Path(cfg["paths"]["annotations_dir"])
    queries = build_queries()

    gt = defaultdict(dict)
    for vid, m in manifest.items():
        ann_path = ann_dir / f"{vid}.txt"
        if not ann_path.exists():
            print(f"uyari: {ann_path} yok, atlaniyor")
            continue
        frames = load_annotations(ann_path)
        for q, fn in queries.items():
            iv = fn(frames, m["n_frames"], m["fps"])
            if iv:
                gt[q][vid] = iv

    out_path = pathlib.Path(cfg["paths"]["groundtruth"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(gt, open(out_path, "w"), indent=1)
    for q, by_vid in gt.items():
        print(f"{q!r}: {len(by_vid)} video")


if __name__ == "__main__":
    main()
