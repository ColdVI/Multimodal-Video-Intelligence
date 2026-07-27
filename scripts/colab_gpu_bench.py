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
    1. `colab_gpu_bundle.zip`'i BIR KERE Google Drive'a yukleyin (Colab'in
       yerel /content diski her runtime kopusunda/sifirlanmasinda silinir;
       Drive kalici). Sonra her oturumda:
       from google.colab import drive; drive.mount('/content/drive')
       !cp /content/drive/MyDrive/colab_gpu_bundle.zip /content/ && \
         cd /content && unzip -q colab_gpu_bundle.zip
       (Drive-mount'lu yoldan DOGRUDAN calistirmayin - cv2 frame-seek FUSE
       uzerinden cok yavas; zip'i her seferinde yerel /content diskine acin,
       bu saniyeler surer.)
    2. !pip install -q sentence-transformers qwen-vl-utils
    3. Ciktiyi da Drive'a yazdirin ki runtime kopunca kaybolmasin:
       !python scripts/colab_gpu_bench.py \
         --out /content/drive/MyDrive/colab_gpu_bench_result.json
       Script her model/varyant bittikce dosyayi YENIDEN YAZAR (tum is
       bitmesini beklemez) - kopma olursa o ana kadarki sonuclar Drive'da
       durur, script tekrar calistirilinca hangi model/varyantlarin zaten
       bitmis oldugunu okuyup atlar.
    4. Sonucu (Drive'daki JSON) indirip bu depoya geri getirin, STATUS.md/
       TASKS.md Faz 4'e gercek sayilarla ekleyin.
"""
import argparse
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

HARDWARE_PROFILE = "colab_t4"  # main() --hardware-profile ile degistirilebilir
EMBED_MODELS = ["xclip_hf_zeroshot", "siglip2_frameavg", "qwen3vl_emb_2048",
                "qwen3vl_emb_1024", "qwen3vl_emb_512", "qwen3vl_emb_256"]
QWEN_MRL_GROUP = ["qwen3vl_emb_2048", "qwen3vl_emb_1024", "qwen3vl_emb_512", "qwen3vl_emb_256"]


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


def _read_all_sample_windows(cfg, windows, n_windows_sample=10):
    """Ornek pencereleri BIR KEZ diskten okur. cv2 rastgele kare-atlama
    (seek) VisDrone videolarinda cok yavas ve pencere ilerledikce buyuyor
    (gozlemlendi: ~7s -> ~190s) - muhtemelen video'da seyrek keyframe/
    index eksikligi, OpenCV her seek'te en yakin keyframe'den itibaren
    yeniden decode ediyor. Bu maliyet TEK BASINA kabul edilebilir ama 6
    embedding modeli icin ayni pencereleri 6 kez yeniden okumak (aslinda
    olan buydu) o maliyeti 6 katina cikariyordu - burada bir kez okuyup
    bellekte tutuyoruz, tum modeller ayni kareleri yeniden kullanir."""
    videos_dir = pathlib.Path(cfg["paths"]["videos_dir"])
    sample_windows = windows[:n_windows_sample]
    cached = []
    total_read_s = 0.0
    for i, w in enumerate(sample_windows):
        video_path = videos_dir / f"{w['video_id']}.mp4"
        t0 = time.perf_counter()
        frames = _read_window(str(video_path), w["t_start"], w["t_end"])
        read_s = time.perf_counter() - t0
        total_read_s += read_s
        cached.append({"window": w, "frames": frames, "read_s": round(read_s, 2)})
        print(f"  [kare-okuma] pencere {i+1}/{len(sample_windows)}: "
             f"{video_path.name} - {len(frames)} kare, {read_s:.1f}s "
             f"(kumulatif {total_read_s:.1f}s)", flush=True)
    return cached, total_read_s


def bench_embedding_speed(cached_windows, queries, out_path, out):
    """cached_windows: _read_all_sample_windows()'un ciktisi - kareler
    onceden bir kez okunmus, burada sadece embed_video/embed_text SURESI
    olculur (video I/O suresi karisimiyor).

    Her model bittikce out["embedding_speed"]'e eklenip diske yazilir -
    Drive'daki --out yoluna isaret ederseniz, runtime kopmasi olsa bile
    o ana kadar biten modellerin sonucu kaybolmaz. Zaten out icinde
    bulunan (onceki kosumdan kalma) modeller atlanir."""
    done = {r["model"] for r in out["embedding_speed"] if "error" not in r}

    for model_name in EMBED_MODELS:
        if model_name in done:
            print(f"{model_name}: atlaniyor (onceki kosumdan sonuc mevcut)")
            continue

        # Qwen MRL boyut varyantlari (2048/1024/512/256) ayni checkpoint'i
        # ayni sekilde forward eder - truncate_dim SADECE cikti vektorunu
        # kirpiyor (bkz. models/qwen3vl_emb.py: SentenceTransformer(...,
        # truncate_dim=self.dim)), transformer'in kendisini degil. Yani
        # 1024/512/256 icin ayni modeli tekrar yukleyip 10 pencereyi tekrar
        # kodlamak (~300sn/pencere x 10 x 3 = ~2.5 saat GPU zamani) hicbir
        # yeni bilgi vermez - qwen3vl_emb_2048'in olculmus zamanini kopyalar.
        if model_name in QWEN_MRL_GROUP and model_name != QWEN_MRL_GROUP[0]:
            base = next((r for r in out["embedding_speed"]
                        if r["model"] == QWEN_MRL_GROUP[0] and "error" not in r), None)
            if base is not None:
                derived = dict(base)
                derived["model"] = model_name
                derived["note"] = (f"{QWEN_MRL_GROUP[0]} ile ayni olculmus timing - MRL "
                                  "truncate_dim sadece cikti vektorunu kirpiyor, "
                                  "transformer forward-pass'ini tekrar calistirmaya "
                                  "gerek yok (bkz. models/qwen3vl_emb.py)")
                out["embedding_speed"].append(derived)
                _save(out_path, out)
                print(f"{model_name}: {QWEN_MRL_GROUP[0]} ile ayni forward-pass, "
                     f"timing kopyalandi (model tekrar calistirilmadi) "
                     f"[kaydedildi -> {out_path}]")
                continue

        timer = StageTimer()
        try:
            t0 = time.perf_counter()
            emb = get_embedder(model_name, device="cuda")
            load_s = time.perf_counter() - t0
        except Exception as exc:
            out["embedding_speed"].append({"model": model_name, "error": f"load basarisiz: {exc}"})
            _save(out_path, out)
            continue

        for i, item in enumerate(cached_windows):
            frames = item["frames"]
            if not frames:
                print(f"  [{model_name}] pencere {i+1}/{len(cached_windows)}: "
                     f"0 kare (onceki okumada bulunamadi), atlaniyor", flush=True)
                continue
            with timer.measure("embed_video"):
                emb.embed_video(frames)
            print(f"  [{model_name}] pencere {i+1}/{len(cached_windows)}: "
                 f"{len(frames)} kare, "
                 f"embed={timer.summary()['embed_video']['mean_s']:.2f}s (kumulatif ort.)",
                 flush=True)

        for q in list(queries)[:10]:
            with timer.measure("embed_text"):
                emb.embed_text(q)

        out["embedding_speed"].append({
            "model": model_name, "hardware_profile": HARDWARE_PROFILE,
            "load_s": round(load_s, 2), "n_windows_sampled": len(cached_windows),
            "timing": timer.summary(),
        })
        _save(out_path, out)
        print(f"{model_name}: load={load_s:.1f}s "
             f"embed_video_mean={timer.summary().get('embed_video', {}).get('mean_s', 0):.2f}s "
             f"embed_text_mean={timer.summary().get('embed_text', {}).get('mean_s', 0):.2f}s "
             f"[kaydedildi -> {out_path}]")


def bench_detector_speed(cfg, windows, out_path, out, n_windows_sample=10):
    detect_path = pathlib.Path(__file__).resolve().parents[1] / "ingest" / "04_detect.py"
    spec = importlib.util.spec_from_file_location("ingest_04_detect", detect_path)
    detect_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(detect_mod)

    videos_dir = pathlib.Path(cfg["paths"]["videos_dir"])
    n_sample = cfg.get("detector", {}).get("n_sample", 6)
    sample_windows = windows[:n_windows_sample]
    done = {r["variant"] for r in out["detector_speed"]}

    for variant_name, variant in cfg["detector"]["variants"].items():
        if variant_name in done:
            print(f"{variant_name}: atlaniyor (onceki kosumdan sonuc mevcut)")
            continue
        class_map = {int(k): v for k, v in variant["class_map"].items()}
        timer = StageTimer()
        for w in sample_windows:
            video_path = videos_dir / f"{w['video_id']}.mp4"
            with timer.measure("detect_window"):
                detect_mod.window_features(
                    str(video_path), w["t_start"], w["t_end"],
                    n_sample=n_sample, checkpoint=variant["checkpoint"], class_map=class_map)
        out["detector_speed"].append({
            "variant": variant_name, "hardware_profile": HARDWARE_PROFILE,
            "n_windows_sampled": len(sample_windows), "timing": timer.summary(),
        })
        _save(out_path, out)
        print(f"{variant_name}: detect_window_mean="
             f"{timer.summary().get('detect_window', {}).get('mean_s', 0):.2f}s "
             f"[kaydedildi -> {out_path}]")


def _save(out_path, out):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(out_path)  # atomik yazma - yaziyorken kopma yarim dosya birakmasin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/colab_gpu_bench.json",
                       help="Sonuc JSON yolu - Drive-mount'lu bir yol verin "
                            "(ör. /content/drive/MyDrive/colab_gpu_bench.json) "
                            "ki runtime kopunca kismi sonuclar kaybolmasin. FARKLI "
                            "bir GPU ile kosarken FARKLI bir --out verin (ör. "
                            "colab_gpu_bench_l4.json) - ayni dosyaya devam edilirse "
                            "modeller 'zaten kayitli' diye atlanip yeni GPU'da hic "
                            "olculmez.")
    parser.add_argument("--hardware-profile", default="colab_t4",
                       help="Ciktiya etiket olarak yazilacak profil adi "
                            "(ör. colab_l4, colab_a100). --out'taki dosyayla "
                            "tutarli olsun diye ayri tutuldu.")
    args = parser.parse_args()
    out_path = pathlib.Path(args.out)

    global HARDWARE_PROFILE
    HARDWARE_PROFILE = args.hardware_profile

    if not torch.cuda.is_available():
        print("UYARI: torch.cuda.is_available() False - bu script GPU olcumu icin "
             "tasarlandi, CPU'da anlamli sonuc uretmez. Colab'de Calisma zamani "
             "turunu GPU yapin.")

    if out_path.exists():
        out = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"Onceki kismi sonuc bulundu ({out_path}), kaldigi yerden devam ediliyor.")
    else:
        out = {
            "hardware_profile": HARDWARE_PROFILE,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "embedding_speed": [],
            "detector_speed": [],
        }

    cfg = load_config()
    windows = json.load(open(cfg["paths"]["windows"]))
    queries = build_queries()

    if all(m in {r["model"] for r in out["embedding_speed"] if "error" not in r} for m in EMBED_MODELS):
        print("Tum embedding modelleri zaten bitmis, kare okuma atlaniyor.")
        cached_windows = []
    else:
        print("Ornek pencereler bir kez okunuyor (tum modeller bu kareleri paylasacak)...")
        cached_windows, total_read_s = _read_all_sample_windows(cfg, windows)
        out["frame_read_total_s"] = round(total_read_s, 2)
        _save(out_path, out)

    bench_embedding_speed(cached_windows, queries, out_path, out)
    bench_detector_speed(cfg, windows, out_path, out)

    print(f"Tamamlandi. JSON kanit: {out_path}")


if __name__ == "__main__":
    main()
