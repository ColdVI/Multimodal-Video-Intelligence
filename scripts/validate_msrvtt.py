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

from bench.gpu_gate import require_gpu_for_qwen_windows
from bench.manifest import torch_info
from dataset_adapters.msrvtt import MSRVTTAdapter
from ingest.frame_io import read_window_frames
from models import get_embedder
from scripts.mrl_truncate_embeddings import truncate_and_renormalize
from scripts.msrvtt_embedding_cache import append_cached, cache_key, cache_path, load_cached

FRAME_SAMPLING_VERSION = "read_window_frames_v1"  # ingest/frame_io.py'nin ornekleme mantigi degisirse artir

# DUZELTME (28 Temmuz 2026): onceki surumde kirmizi bayrak CLIP4Clip'in
# FINE-TUNE EDILMIS sayisina (44.5/71.4/81.6) karsi kosuluyordu - bu
# checkpoint'imiz (zero-shot) icin adil bir kiyas degildi (bkz. asagidaki
# REFERENCE_ONLY_BASELINES notu). Yerine, GERCEK zero-shot CLIP baseline'i
# iki BAGIMSIZ kaynaktan dogrulanarak bulundu:
#   - Birincil kaynak: Portillo-Quintero ve ark. 2021, arXiv:2102.12443
#     ("A Straightforward Framework For Video Retrieval Using CLIP"), Tablo 2
#   - Capraz dogrulama: CLIP4Clip (Luo ve ark. 2021, arXiv:2104.08860),
#     Tablo 1(b) AYNI sayiyi "CLIP-straight" adiyla tekrar yayinliyor
#
# ONEMLI - Portillo-Quintero Tablo 2 IKI YON rapor eder, KARISTIRILMAMALI:
#   T2V (metin sorgu -> video ariyor):  R@1=31.2 R@5=53.7 R@10=64.2 MdR=4
#   V2T (video sorgu -> metin ariyor):  R@1=27.2 R@5=51.7 R@10=62.6 MdR=5
# Asagidaki compute_retrieval_metrics()'te sim[i,j] = caption_i . video_j;
# HER CAPTION SATIRI icin videolar siralaniyor - yani olcumumuz T2V'dir.
# T2V satiri kullanilmali (V2T ~4 puan farkli, yanlis yon olurdu).
#
# CHECKPOINT SOYU NOTU: microsoft/xclip-base-patch16-zero-shot, Ni ve ark.
# (video SINIFLANDIRMA icin on-egitilmis) X-CLIP soyundandir - Ma ve ark.
# (2022, ACM MM, retrieval icin uctan uca egitilmis) X-CLIP'i DEGIL (bkz.
# models/xclip_hf.py docstring'i). Zero-shot CLIP-straight'e gore de biraz
# geride kalmasi KISMEN BEKLENEN: siniflandirma-soylu on-egitim, "duz" CLIP
# gorsel-metin hizalamasindan farkli bir hedefe optimize eder. Bu bug degil,
# checkpoint secimi. Ma ve ark. X-CLIP'i (github.com/xuguohai/X-CLIP) HF Hub
# ustunden AutoModel ile YUKLENEMIYOR (dogrulandi) - yalniz kendi ozel model
# koduyla (custom checkpoint format, pip paketi yok) calisiyor; entegrasyonu
# bu scriptin kapsamini asan ayri bir is oldugundan simdilik atlandi.
ZERO_SHOT_BASELINE_NAME = (
    "CLIP-straight zero-shot (Portillo-Quintero ve ark. 2021, T2V, MSR-VTT 1k-A, "
    "fine-tune yok) - iki bagimsiz kaynaktan dogrulandi, kirmizi bayrak buna karsi kosulur")
ZERO_SHOT_BASELINE = {"R@1": 31.2, "R@5": 53.7, "R@10": 64.2, "MedR": 4.0}

# Yalniz REFERANS icin JSON ciktisinda tutulur (fine-tuning ile ne kadar
# yukselinebilecegine dair ust sinir gostergesi) - fine-tune edilmis bir
# retrieval modelini zero-shot checkpoint'imizle DOGRUDAN KIYASLAMAK adil
# degil, bu yuzden kirmizi bayrak kontrolu BUNA KARSI KOSULMAZ.
REFERENCE_ONLY_BASELINES = {
    "CLIP4Clip (ViT-B/32), fine-tuned t2v - ust sinir referansi, red bayrak icin KULLANILMIYOR": {
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
                     n_frames: int = 32, limit: int = None,
                     cache_file: pathlib.Path = None) -> dict:
    """video_id -> L2-normalize embedding. Her klip TEK pencere (0, sure) -
    VisDrone'daki 8s/4s kayan pencereleme burada YOK, standart t2v
    protokolu tum klibi tek birim olarak ele alir.

    cache_file verilirse (Faz 6 GPU-hazirlik): onceden tamamlanmis
    video_id'ler ATLANIR (resume - yarim kalan bir kosumda bastan
    baslanmaz), her yeni video TAMAMLANDIKCA diske yazilir (crash sonrasi
    ilerleme kaybolmaz). cache_file=None ise mevcut davranis birebir ayni
    (bellek-ici, kalici yazma yok) - X-CLIP/SigLIP2 gibi ucuz CPU
    modellerinde onbellek gereksiz karmasiklik olurdu."""
    emb = get_embedder(model_name)
    out = dict(load_cached(cache_file)) if cache_file else {}
    if out:
        print(f"  onbellekten devam: {len(out)} video zaten tamamlanmis ({cache_file})")
    items = entries[:limit] if limit else entries
    for i, e in enumerate(items):
        vid = e["video_id"]
        if vid in out:
            continue
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
        out[vid] = vec
        if cache_file:
            append_cached(cache_file, vid, vec.tolist())
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


def mean_rank_vs_chance(mean_r: float, n: int) -> str:
    """MeanR'yi rastgele siralamanin beklenen ortalama sirasina (n+1)/2 karsi
    ifade eder. Hangi baseline'in "adil" oldugu tartismali olsa bile, bu
    satir HER ZAMAN hesaplanabilir ve boru hattinin en azindan rastgele
    oneriden anlamli farkli oldugunu SAYISAL olarak dogrular - model
    kalitesinden bagimsiz bir asgari saglamlik kontrolu (GOREV maddesi 3,
    28 Temmuz 2026)."""
    chance = (n + 1) / 2
    ratio = chance / mean_r if mean_r > 0 else float("inf")
    return f"MeanR={mean_r:.1f}, rastgele sans ~{chance:.1f} -> {ratio:.1f}x daha iyi"


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


def qwen_cache_file(model_name: str, split_path: str, n_frames: int,
                    cache_dir: pathlib.Path = pathlib.Path("data/downloads/msrvtt/embedding_cache")) -> pathlib.Path:
    """Faz 6 GPU-hazirlik: qwen3vl_emb ailesi icin dataset+model kimligine
    baglanmis onbellek yolu. Herhangi bir alan (dataset hash, split, kare
    sayisi...) degisirse FARKLI dosya - eski/gecersiz sonuclarla sessizce
    karismaz."""
    dataset_manifest = MSRVTTAdapter().manifest()
    key = cache_key(
        dataset_id=dataset_manifest.dataset_id, split=dataset_manifest.split,
        dataset_hash=dataset_manifest.source_hash, model_id="Qwen/Qwen3-VL-Embedding-2B",
        model_revision="unknown", n_sample=n_frames,
        frame_sampling_version=FRAME_SAMPLING_VERSION, dtype="unknown",
        dimension_source="native" if model_name == "qwen3vl_emb_2048" else "truncated_from_2048",
    )
    return cache_path(cache_dir, model_name, key)


def run_validation(model_name: str, entries: list, videos_dir: pathlib.Path,
                   n_frames: int, limit: int, split_path: str = None) -> dict:
    print(f"=== {model_name} ===")
    cache_file = None
    if model_name.startswith("qwen3vl_emb"):
        n_items = len(entries[:limit] if limit else entries)
        # VisDrone pencere hizindan turetilen kaba tahmin - MSR-VTT klip
        # uzunlugu/kare sayisi farkli olabilir, sadece kapiyi tetiklemeye yeter.
        require_gpu_for_qwen_windows(f"validate_msrvtt.py --models {model_name}", n_items)
        cache_file = qwen_cache_file(model_name, split_path, n_frames)
    t0 = time.perf_counter()
    video_embs = embed_all_videos(entries, videos_dir, model_name, n_frames, limit,
                                  cache_file=cache_file)
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
    metrics["mean_rank_vs_chance"] = mean_rank_vs_chance(metrics["MeanR"], metrics["n"])
    metrics["video_embed_total_s"] = round(video_s, 1)
    metrics["text_embed_total_s"] = round(text_s, 1)
    metrics["n_videos_embedded"] = len(video_embs)
    metrics["n_pairs_evaluated"] = len(common_ids)
    metrics["compute_environment"] = torch_info()  # GPU adi/VRAM, cuda_available - artifact'ta nerede uretildigi belli olsun
    metrics["embedding_cache_file"] = str(cache_file) if cache_file else None
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
        measured = run_validation(model_name, entries, videos_dir, args.n_frames, args.limit,
                                  split_path=args.split)
        red_flags = []
        flags = red_flag_check(measured, ZERO_SHOT_BASELINE)
        if flags:
            red_flags.append({"baseline": ZERO_SHOT_BASELINE_NAME, "flags": flags})
        results[model_name] = {"measured": measured, "red_flags": red_flags}
        print(f"  R@1={measured['R@1']:.1f} R@5={measured['R@5']:.1f} "
             f"R@10={measured['R@10']:.1f} MedR={measured['MedR']:.1f} "
             f"(n={measured['n_pairs_evaluated']})")
        print(f"  {measured['mean_rank_vs_chance']}")
        for rf in red_flags:
            for f in rf["flags"]:
                print(f"  [!] {f}")

    out = {
        "protocol": "MSR-VTT 1k-A, standart t2v retrieval (video basina tek pencere, "
                    "IoU/merge_intervals YOK) - MVEB'in filtrelenmis ~879 klip alt "
                    "kumesiyle DEGIL, tam 1000 klipin klasik protokolüyle karsilastirilabilir.",
        "n_frames_per_video": args.n_frames,
        "limit": args.limit,
        "zero_shot_baseline": {"name": ZERO_SHOT_BASELINE_NAME, "values": ZERO_SHOT_BASELINE},
        "reference_only_baselines": REFERENCE_ONLY_BASELINES,
        "results": results,
    }
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nJSON kanit: {out_path}")


if __name__ == "__main__":
    main()
