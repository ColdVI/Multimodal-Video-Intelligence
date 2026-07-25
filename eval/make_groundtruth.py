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


def gt_count_at_least(frames, n_frames, cat_name, fps, min_count):
    """Ayni karede >=min_count kategori-instance'i var mi (sayisal sorgular
    icin, ör. 'en az 3 araba')."""
    cid = CAT[cat_name]
    flags = [sum(1 for _, c, _, _ in frames.get(i, []) if c == cid) >= min_count
             for i in range(1, n_frames + 1)]
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
    Her kavram TR+EN cifti olarak tanimlanir (ayni GT fonksiyonu, farkli
    sorgu metni) - modelin coğu Ingilizce egitildigi icin dil farkinin
    kendi olcum satiri olmasi, sessiz bir bias olmamasi icin (bkz.
    docs/codex/05_CODEX_BENCHMARK_VE_OPTIMIZASYON_PLANI.md Faz 1 madde 3).
    Kategoriler: tekli, hareket, sayisal, bileşik, negatif-kontrol (corpus
    disi kavram, bos GT donmeli)."""
    def bus(F, N, fps):
        return gt_object(F, N, "bus", fps)

    def truck(F, N, fps):
        return gt_object(F, N, "truck", fps)

    def car(F, N, fps):
        return gt_object(F, N, "car", fps)

    def pedestrian(F, N, fps):
        return gt_object(F, N, "pedestrian", fps)

    def van(F, N, fps):
        return gt_object(F, N, "van", fps)

    def walking(F, N, fps):
        return gt_walking(F, N, fps)

    def empty(F, N, fps):
        return []

    def car_at_least_3(F, N, fps):
        return gt_count_at_least(F, N, "car", fps, 3)

    def person_at_least_5(F, N, fps):
        return gt_count_at_least(F, N, "pedestrian", fps, 5)

    def crowd_at_least_10(F, N, fps):
        return gt_count_at_least(F, N, "pedestrian", fps, 10)

    def bus_and_walking(F, N, fps):
        return intersect(bus(F, N, fps), walking(F, N, fps))

    def truck_and_pedestrian(F, N, fps):
        return intersect(truck(F, N, fps), pedestrian(F, N, fps))

    def car_and_pedestrian(F, N, fps):
        return intersect(car(F, N, fps), pedestrian(F, N, fps))

    return {
        # tekli (5 kavram x TR/EN)
        "otobüsü göster": bus, "show the bus": bus,
        "kamyonu göster": truck, "show the truck": truck,
        "arabaları göster": car, "show the cars": car,
        "yayaları göster": pedestrian, "show the pedestrians": pedestrian,
        "vanı göster": van, "show the van": van,  # van_count kolonu yok - filtresiz kavram ornegi
        # hareket
        "yürüyen adamı göster": walking, "show the walking person": walking,
        # sayisal (esik CONCEPT_MAP'teki filtre esigiyle hizali)
        "en az 3 araba göster": car_at_least_3, "show at least 3 cars": car_at_least_3,
        "en az 5 kişi göster": person_at_least_5, "show at least 5 people": person_at_least_5,
        "kalabalık bir sahne göster": crowd_at_least_10, "show a crowded scene": crowd_at_least_10,
        # bileşik (nesne ∩ hareket, nesne ∩ nesne)
        "otobüs ve yürüyen adam": bus_and_walking, "bus and walking person": bus_and_walking,
        "kamyon ve yaya birlikte": truck_and_pedestrian,
        "truck and pedestrian together": truck_and_pedestrian,
        "araba ve yaya birlikte": car_and_pedestrian,
        "car and pedestrian together": car_and_pedestrian,
        # negatif-kontrol: corpus'ta olmayan kavram, bos donmeli
        "tren göster": empty, "show the train": empty,
        "bisiklet göster": empty, "show the bicycle": empty,
    }


def build_query_metadata():
    """Sorgu metni -> {category, lang, concept}. Kategori/dil'i sorgu
    metninden regex ile tahmin etmek (İngilizce "and"/"walking" gibi
    kelimeler Türkçe kalıplarla örtüşmez) kırılgan olurdu; bu yüzden GT
    üretimiyle aynı yerde, elle ve açıkça tanımlanır. bench/ ve
    eval/run_eval.py::category_of bu sözlüğü referans alabilir."""
    meta = {}
    for concept, tr, en in [
        ("bus", "otobüsü göster", "show the bus"),
        ("truck", "kamyonu göster", "show the truck"),
        ("car", "arabaları göster", "show the cars"),
        ("pedestrian", "yayaları göster", "show the pedestrians"),
        ("van", "vanı göster", "show the van"),
    ]:
        meta[tr] = {"category": "tekli", "lang": "tr", "concept": concept}
        meta[en] = {"category": "tekli", "lang": "en", "concept": concept}
    for concept, tr, en in [
        ("walking", "yürüyen adamı göster", "show the walking person"),
    ]:
        meta[tr] = {"category": "hareket", "lang": "tr", "concept": concept}
        meta[en] = {"category": "hareket", "lang": "en", "concept": concept}
    for concept, tr, en in [
        ("car_at_least_3", "en az 3 araba göster", "show at least 3 cars"),
        ("person_at_least_5", "en az 5 kişi göster", "show at least 5 people"),
        ("crowd_at_least_10", "kalabalık bir sahne göster", "show a crowded scene"),
    ]:
        meta[tr] = {"category": "sayısal", "lang": "tr", "concept": concept}
        meta[en] = {"category": "sayısal", "lang": "en", "concept": concept}
    for concept, tr, en in [
        ("bus_and_walking", "otobüs ve yürüyen adam", "bus and walking person"),
        ("truck_and_pedestrian", "kamyon ve yaya birlikte",
         "truck and pedestrian together"),
        ("car_and_pedestrian", "araba ve yaya birlikte",
         "car and pedestrian together"),
    ]:
        meta[tr] = {"category": "bileşik", "lang": "tr", "concept": concept}
        meta[en] = {"category": "bileşik", "lang": "en", "concept": concept}
    for concept, tr, en in [
        ("train", "tren göster", "show the train"),
        ("bicycle", "bisiklet göster", "show the bicycle"),
    ]:
        meta[tr] = {"category": "negatif-kontrol", "lang": "tr", "concept": concept}
        meta[en] = {"category": "negatif-kontrol", "lang": "en", "concept": concept}
    assert set(meta) == set(build_queries()), "metadata ve sorgu seti senkron degil"
    return meta


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
