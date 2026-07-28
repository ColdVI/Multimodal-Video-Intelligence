"""MSR-VTT 1k-A boru hatti dogrulamasi - TEK SEFERLIK sanity check, KALICI
ingest hattina eklenmez, ClickHouse'a yazmaz.

AMAC: MSR-VTT'de tum aday modellerin yayinlanmis R@1/R@5/R@10/MedR
sayilari var. Bizim olctugumuz sayi yayinlanan sayiya yakin CIKMIYORSA,
sorun MODELDE degil bizim boru hattimizdadir (kare ornekleme, normalize,
pooling zinciri...) - bu script'in birincil degeri budur.

PROTOKOL FARKI (VisDrone benchmarkindan): burasi standart text-to-video
retrieval - video BASINA tek pencere (tum klip), IoU/temporal esleme YOK,
merge_intervals YOK. GT dogrudan video_id<->caption 1-1 eslesmesi. Bu
yuzden eval/metrics.py'nin zaman-araligi tabanli evaluate()'ini
KULLANMIYOR - compute_retrieval_metrics() standart rank-tabanli
R@K/MedR/MeanR hesabi.

UYARI: MVEB liderlik tablosu bu setin FILTRELENMIS bir alt kumesini
(~879 klip, ses cikarma basarisiz olanlar elenmis) kullaniyor olabilir.
Burada TAM 1000 klipin standart 1k-A protokolu kullaniliyor - klasik
yayinlanmis CLIP4Clip-tarzi R@1/R@5/R@10 ile karsilastirilabilir, MVEB
skoruyla DEGIL. Ikisini karistirmayin.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ingest.frame_io import read_window_frames
from models import get_embedder

# CLIP4Clip (ViT-B/32), standart 1k-A t2v protokolu - literatur referansi.
#
# GERCEK KOSUM SONRASI ONEMLI DUZELTME (28 Temmuz 2026): xclip_hf_zeroshot
# ile tam 1000 videoluk kosum R@1=21.5 verdi, bu baseline'dan (44.5) 23 puan
# uzak - KIRMIZI BAYRAK tetiklendi. AMA arastirinca CLIP4Clip'in bu sayisi
# GECERSIZ bir karsilastirma noktasi: CLIP4Clip, MSR-VTT ciftleri uzerinde
# UCTAN UCA FINE-TUNE EDILMIS bir retrieval modeli (ozel benzerlik-baglantisi
# katmani + video-metin ciftleriyle egitim). microsoft/xclip-base-patch16-
# zero-shot ise (dogrulandi: HF model karti) YALNIZCA video SINIFLANDIRMA
# (HMDB-51/UCF-101/Kinetics-600) icin zero-shot degerlendirilmis; MODEL
# KARTINDA MSR-VTT RETRIEVAL SAYISI HIC YOK. Yani bu KIRMIZI BAYRAK muhtemelen
# "boru hattimiz bozuk" degil "yanlis/uygunsuz referans sectim" anlamina
# geliyor - fine-tune edilmis bir modelin sayisini fine-tune EDILMEMIS bir
# modelle karsilastirmak bastan adil degildi. Bunu net bir "pipeline dogru"
# kanitina cevirmek icin CLIP4Clip'in KENDI makalesindeki "CLIP zero-shot/
# straight" (fine-tune edilmemis) ablasyon satirini PDF'ten bulup buraya
# eklemek gerekiyor - bu oturumda paperswithcode.com'un tamamen baska bir
# sayfaya yonlendirmesi ve arXiv ozetinin tablo icermemesi yuzunden
# BULUNAMADI. Sonuc: bu kosum ne "pipeline bozuk" ne "pipeline dogru"
# kanitlar - baseline'in kendisi elden gecirilmeden yorumlanamaz.
PUBLISHED_BASELINES = {
    "CLIP4Clip (ViT-B/32), t2v - DOGRULANMAMIS + xclip_hf_zeroshot icin "
    "UYGUNSUZ (fine-tuned vs zero-shot karsilastirmasi, yukaridaki notu okuyun)": {
        "R@1": 44.5, "R@5": 71.4, "R@10": 81.6, "MedR": 2.0},
}
FLAG_THRESHOLD_PCT = 10.0  # bu kadar sapma KIRMIZI BAYRAK


def load_test_split(json_path: str) -> list:
    return json.load(open(json_path, encoding="utf-8"))


def probe_video_duration(video_path: str) -> float:
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n_frames / fps if fps else 0.0


def embed_all_videos(entries: list, videos_dir: pathlib.Path, model_name: str,
                     n_frames: int = 32, limit: int = None) -> dict:
    """video_id -> L2-normalize embedding. Her klip TEK pencere (0, sure) -
    VisDrone'daki 8s/4s kayan pencereleme burada YOK, standart t2v
    protokolu tum klibi tek birim olarak ele alir."""
    emb = get_embedder(model_name)
    out = {}
    items = entries[:limit] if limit else entries
    for i, e in enumerate(items):
        video_path = videos_dir / e["video"]
        if not video_path.exists():
            continue
        duration = probe_video_duration(str(video_path))
        if duration <= 0:
            continue
        frames = read_window_frames(str(video_path), 0.0, duration, n=n_frames)
        if not frames:
            continue
        vec = emb.embed_video(frames)
        out[e["video_id"]] = vec
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(items)} video embed edildi")
    return out


def embed_all_captions(entries: list, model_name: str, limit: int = None) -> dict:
    emb = get_embedder(model_name)
    items = entries[:limit] if limit else entries
    return {e["video_id"]: emb.embed_text(e["caption"]) for e in items}


def compute_retrieval_metrics(sim_matrix: np.ndarray) -> dict:
    """sim_matrix[i,j] = caption i ile video j arasindaki cosine benzerlik.
    GT kosulu: caption i'nin dogru eslesmesi video i (satir/sutun
    index'leri hizali - cagiran taraf garanti eder). Standart t2v
    retrieval: R@1/R@5/R@10/MedR/MeanR (yuzde olarak R@K)."""
    n = sim_matrix.shape[0]
    ranks = np.zeros(n)
    for i in range(n):
        order = np.argsort(-sim_matrix[i])
        rank = int(np.where(order == i)[0][0]) + 1
        ranks[i] = rank
    return {
        "R@1": float(np.mean(ranks <= 1) * 100),
        "R@5": float(np.mean(ranks <= 5) * 100),
        "R@10": float(np.mean(ranks <= 10) * 100),
        "MedR": float(np.median(ranks)),
        "MeanR": float(np.mean(ranks)),
        "n": n,
    }


def red_flag_check(measured: dict, baseline: dict) -> list:
    flags = []
    for key in ("R@1", "R@5", "R@10"):
        if key not in baseline:
            continue
        diff_pct = abs(measured[key] - baseline[key])
        if diff_pct > FLAG_THRESHOLD_PCT:
            flags.append(f"{key}: olculen={measured[key]:.1f} yayinlanan={baseline[key]:.1f} "
                        f"(fark {diff_pct:.1f} puan > {FLAG_THRESHOLD_PCT} esigi) - "
                        "olasi sebep: kare sayisi, cozunurluk, normalize, prompt sablonu, split farki")
    return flags


def run_validation(model_name: str, entries: list, videos_dir: pathlib.Path,
                   n_frames: int, limit: int) -> dict:
    print(f"=== {model_name} ===")
    t0 = time.perf_counter()
    video_embs = embed_all_videos(entries, videos_dir, model_name, n_frames, limit)
    video_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    caption_embs = embed_all_captions(entries, model_name, limit)
    text_s = time.perf_counter() - t0

    common_ids = [vid for vid in video_embs if vid in caption_embs]
    if len(common_ids) < len(entries[:limit] if limit else entries):
        print(f"  UYARI: {len(entries[:limit] if limit else entries) - len(common_ids)} "
             "video/caption ciftinden biri eksik (video acilmadi?), onlar disarida birakildi")

    video_mat = np.stack([video_embs[vid] for vid in common_ids])
    caption_mat = np.stack([caption_embs[vid] for vid in common_ids])
    sim = caption_mat @ video_mat.T  # L2-normalize varsayimi - modeller zaten normalize donuyor

    metrics = compute_retrieval_metrics(sim)
    metrics["video_embed_total_s"] = round(video_s, 1)
    metrics["text_embed_total_s"] = round(text_s, 1)
    metrics["n_videos_embedded"] = len(video_embs)
    metrics["n_pairs_evaluated"] = len(common_ids)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/downloads/msrvtt/msrvtt_test_1k.json")
    ap.add_argument("--videos-dir", default="data/downloads/msrvtt/videos")
    ap.add_argument("--models", nargs="+", default=["xclip_hf_zeroshot", "qwen3vl_emb_2048"])
    ap.add_argument("--n-frames", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None,
                    help="hizli sanity-check icin ilk N klip - tam 1000 icin verilmesin")
    ap.add_argument("--out", default="artifacts/pipeline_validation.json")
    args = ap.parse_args()

    entries = load_test_split(args.split)
    videos_dir = pathlib.Path(args.videos_dir)

    results = {}
    for model_name in args.models:
        measured = run_validation(model_name, entries, videos_dir, args.n_frames, args.limit)
        red_flags = []
        for baseline_name, baseline in PUBLISHED_BASELINES.items():
            flags = red_flag_check(measured, baseline)
            if flags:
                red_flags.append({"baseline": baseline_name, "flags": flags})
        results[model_name] = {"measured": measured, "red_flags": red_flags}
        print(f"  R@1={measured['R@1']:.1f} R@5={measured['R@5']:.1f} "
             f"R@10={measured['R@10']:.1f} MedR={measured['MedR']:.1f} "
             f"(n={measured['n_pairs_evaluated']})")
        for rf in red_flags:
            for f in rf["flags"]:
                print(f"  [!] {f}")

    out = {
        "protocol": "MSR-VTT 1k-A, standart t2v retrieval (video basina tek pencere, "
                    "IoU/merge_intervals YOK) - MVEB'in filtrelenmis ~879 klip alt "
                    "kumesiyle DEGIL, tam 1000 klipin klasik protokolüyle karsilastirilabilir.",
        "n_frames_per_video": args.n_frames,
        "limit": args.limit,
        "published_baselines": PUBLISHED_BASELINES,
        "results": results,
    }
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nJSON kanit: {out_path}")


if __name__ == "__main__":
    main()
