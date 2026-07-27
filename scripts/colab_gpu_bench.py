"""Faz 4/GPU eki: Colab'de (T4 veya baska bir CUDA GPU) embedding + YOLO
dedeksiyon HIZINI olcer ve hardware_profile='colab_t4' ile etiketler.

NEDEN AYRI BIR SCRIPT: mevcut bench/ paketi ClickHouse'a bagli (strateji
matrisi, arama gecikmesi); Colab'de gecici bir ClickHouse kurmak kirilgan
olurdu. Kalite (Recall/Precision) zaten cihazdan bagimsizdir - sadece HIZ
olcmeye deger. Bu script bu yuzden yalnizca embed_video/embed_text ve YOLO
window_features() sure olcumlerini yapar; ClickHouse gerekmez.

DURUM: Bu script CPU'lu gelistirme makinesinde YAZILDI ama gercek bir
GPU'da CALISTIRILMADI - bench/timing.py ve models.get_embedder() gibi
zaten dogrulanmis parcalari yeniden kullanir, ama uctan uca kosum
kanitlanmadi. Sonuclari STATUS.md/TASKS.md Faz 4'e elle eklenmeli.

Kullanim (Colab, GPU runtime):
    1. Bu repoyu (veya en azindan gerekli dosyalari) Colab'e getir.
    2. `data/raw/videos/*.mp4`, `data/windows.json`, `data/features.json`,
       `yolo26x.pt`, `weights/yolo_visdrone/best.pt`,
       `weights/yolo_visdrone_s/best.pt` mevcut olmali (bkz.
       scripts/package_colab_gpu_bundle.py - bu dosyalari CPU makinesinde
       tek bir zip'e toplar).
    3. !pip install -q sentence-transformers qwen-vl-utils
    4. !python scripts/colab_gpu_bench.py
    5. Ciktiyi (artifacts/colab_gpu_bench.json) indirip bu depoya geri
       getir, STATUS.md/TASKS.md Faz 4'e gercek sayilarla ekle.
"""
import importlib.util
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import load_config
from bench.timing import StageTimer
from eval.make_groundtruth import build_queries
from models import get_embedder

HARDWARE_PROFILE = "colab_t4"  # gercek GPU adiyla degistirin (ör. colab_a100)
EMBED_MODELS = ["xclip_hf_zeroshot", "siglip2_frameavg", "qwen3vl_emb_2048",
                "qwen3vl_emb_1024", "qwen3vl_emb_512", "qwen3vl_emb_256"]


def _read_window(video_path, t0, t1, n=32):
    import cv2
    import numpy as np
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


def bench_embedding_speed(cfg, windows, queries, n_windows_sample=10):
    """Tum pencereleri degil, ilk n_windows_sample'i kullanir - amac hiz
    olcmek, tam re-embed degil (CPU tarafinda zaten tam koşum var)."""
    videos_dir = pathlib.Path(cfg["paths"]["videos_dir"])
    sample_windows = windows[:n_windows_sample]
    results = []

    for model_name in EMBED_MODELS:
        timer = StageTimer()
        try:
            t0 = time.perf_counter()
            emb = get_embedder(model_name, device="cuda")
            load_s = time.perf_counter() - t0
        except Exception as exc:
            results.append({"model": model_name, "error": f"load basarisiz: {exc}"})
            continue

        for w in sample_windows:
            video_path = videos_dir / f"{w['video_id']}.mp4"
            frames = _read_window(str(video_path), w["t_start"], w["t_end"])
            if not frames:
                continue
            with timer.measure("embed_video"):
                emb.embed_video(frames)

        for q in list(queries)[:10]:
            with timer.measure("embed_text"):
                emb.embed_text(q)

        results.append({
            "model": model_name, "hardware_profile": HARDWARE_PROFILE,
            "load_s": round(load_s, 2), "n_windows_sampled": len(sample_windows),
            "timing": timer.summary(),
        })
        print(f"{model_name}: load={load_s:.1f}s "
             f"embed_video_mean={timer.summary().get('embed_video', {}).get('mean_s', 0):.2f}s "
             f"embed_text_mean={timer.summary().get('embed_text', {}).get('mean_s', 0):.2f}s")
    return results


def bench_detector_speed(cfg, windows, n_windows_sample=10):
    detect_path = pathlib.Path(__file__).resolve().parents[1] / "ingest" / "04_detect.py"
    spec = importlib.util.spec_from_file_location("ingest_04_detect", detect_path)
    detect_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(detect_mod)

    videos_dir = pathlib.Path(cfg["paths"]["videos_dir"])
    n_sample = cfg.get("detector", {}).get("n_sample", 6)
    sample_windows = windows[:n_windows_sample]
    results = []

    for variant_name, variant in cfg["detector"]["variants"].items():
        class_map = {int(k): v for k, v in variant["class_map"].items()}
        timer = StageTimer()
        for w in sample_windows:
            video_path = videos_dir / f"{w['video_id']}.mp4"
            with timer.measure("detect_window"):
                detect_mod.window_features(
                    str(video_path), w["t_start"], w["t_end"],
                    n_sample=n_sample, checkpoint=variant["checkpoint"], class_map=class_map)
        results.append({
            "variant": variant_name, "hardware_profile": HARDWARE_PROFILE,
            "n_windows_sampled": len(sample_windows), "timing": timer.summary(),
        })
        print(f"{variant_name}: detect_window_mean="
             f"{timer.summary().get('detect_window', {}).get('mean_s', 0):.2f}s")
    return results


def main():
    if not torch.cuda.is_available():
        print("UYARI: torch.cuda.is_available() False - bu script GPU olcumu icin "
             "tasarlandi, CPU'da anlamli sonuc uretmez. Colab'de Calisma zamani "
             "turunu GPU yapin.")

    cfg = load_config()
    windows = json.load(open(cfg["paths"]["windows"]))
    queries = build_queries()

    embed_results = bench_embedding_speed(cfg, windows, queries)
    detector_results = bench_detector_speed(cfg, windows)

    out = {
        "hardware_profile": HARDWARE_PROFILE,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "embedding_speed": embed_results,
        "detector_speed": detector_results,
    }
    out_path = pathlib.Path("artifacts/colab_gpu_bench.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON kanit: {out_path}")


if __name__ == "__main__":
    main()
