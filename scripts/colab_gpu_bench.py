"""Faz 4/GPU eki: Colab'de (T4 veya baska bir CUDA GPU) embedding + YOLO
dedeksiyon HIZINI olcer ve --hardware-profile ile etiketler.

NEDEN AYRI BIR SCRIPT: mevcut bench/ paketi ClickHouse'a bagli (strateji
matrisi, arama gecikmesi); Colab'de gecici bir ClickHouse kurmak kirilgan
olurdu. Kalite (Recall/Precision) zaten cihazdan bagimsizdir - sadece HIZ
olcmeye deger. Bu script bu yuzden yalnizca embed_video/embed_text ve YOLO
window_features() sure olcumlerini yapar; ClickHouse gerekmez.

SEMA v2 (onceki calistirmalardan ders): ilk versiyon canli bir Colab
kosumunda su sorunlari cikardi (detay: docs/codex/06_NIHAI_RAPOR.md,
chunking_embedding_spec.md):
- Kare okuma (video decode) her pencerede n_sample kez AYRI seek yapiyordu;
  H.264'te bu decoder'i en yakin keyframe'e geri gonderip yeniden decode
  ettiriyor - pencere basina 7-205s olculdu. ingest/frame_io.py'deki TEK
  seek + sequential okuma bunu duzeltti (gercek videoda 12.6x, ayni
  sonucla dogrulandi).
- detect_window suresi decode+YOLO'yu birlestiriyordu; uc farkli boyuttaki
  YOLO modeli arasinda GPU'da yalniz %3 fark olculdu - bu YOLO'nun kendisi
  degil, ayni seek-agirlikli decode'un tekrarlanmasiydi. Artik dedektor,
  embedding icin zaten okunmus kareleri PAYLASIYOR (window_features'a
  frames= veriliyor, video tekrar acilmiyor).
- qwen3vl_emb_1024/512/256, 2048 ile AYNI transformer forward-pass'ini
  calistirir (truncate_dim yalniz cikti vektorunu kirpar) - eskiden 4 ayri
  JSON satiri olarak yazilip "6 model olculdu" izlenimi veriyordu. Artik
  TEK "qwen3vl_emb" satiri + truncate_dims listesi.
- Ortam bilgisi (compute capability, torch/cuda surumu, cpu/ram) hic
  kaydedilmiyordu - iki farkli GPU/oturum sonucu bu olmadan guvenilir
  karsilastirilamaz.
- Ilk 1-2 pencere CUDA/cudnn "warm-up" maliyeti tasir (kernel derleme,
  ilk tahsis) - ortalamaya karisirsa yavas gorunur. Artik ilk WARMUP_WINDOWS
  pencere calistirilir ama zamana dahil edilmez.

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
    3. Ciktiyi da Drive'a yazdirin ki runtime kopunca kaybolmasin. FARKLI
       bir GPU ile kosarken FARKLI bir --out ve --hardware-profile verin:
       !python scripts/colab_gpu_bench.py \
         --out /content/drive/MyDrive/colab_gpu_bench_l4.json \
         --hardware-profile colab_l4
       Script her model/varyant bittikce dosyayi YENIDEN YAZAR (tum is
       bitmesini beklemez) - kopma olursa o ana kadarki sonuclar Drive'da
       durur, script tekrar calistirilinca hangi model/varyantlarin zaten
       bitmis oldugunu okuyup atlar.
    4. Sonucu (Drive'daki JSON) indirip bu depoya geri getirin.

Eski (v1) semali bir JSON dosyasi verirseniz migrate_legacy_schema() onu
otomatik v2'ye cevirir (gecmis olcumler kaybolmaz).
"""
import argparse
import json
import pathlib
import platform
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import load_config
from bench.timing import StageTimer
from eval.make_groundtruth import build_queries
from ingest.frame_io import read_window_frames
from models import get_embedder

SCHEMA_VERSION = 2
HARDWARE_PROFILE = "colab_t4"  # main() --hardware-profile ile degistirilir
WARMUP_WINDOWS = 2  # ilk N pencere CUDA/cudnn isinma maliyeti tasir, zamana dahil edilmez
QWEN_FAMILY_MODEL = "qwen3vl_emb_2048"  # gercekten calistirilacak registry adi
QWEN_TRUNCATE_DIMS = [2048, 1024, 512, 256]
EMBED_MODELS = ["xclip_hf_zeroshot", "siglip2_frameavg", QWEN_FAMILY_MODEL]


def migrate_legacy_schema(old: dict) -> dict:
    """v1 (schema_version alani yok) -> v2. v1'de qwen3vl_emb_2048/1024/512/256
    4 ayri satirdi (1024/512/256 icinde 'note' alani vardi, 2048 ile ayni
    timing'i tasiyordu) - v2'de tek 'qwen3vl_emb' + truncate_dims olur.
    v1'de environment/warmup alanlari yoktu - bos/None olarak eklenir."""
    if old.get("schema_version") == SCHEMA_VERSION:
        return old

    new = {
        "schema_version": SCHEMA_VERSION,
        "hardware_profile": old.get("hardware_profile"),
        "gpu_name": old.get("gpu_name"),
        "environment": old.get("environment"),  # v1'de yoktu -> None, bilgi kaybi acikca isaretli
        "warmup_windows": None,  # v1'de warm-up yoktu, bilinmiyor
        "embedding_speed": [],
        "detector_speed": old.get("detector_speed", []),
        "frame_read_total_s": old.get("frame_read_total_s"),
    }
    qwen_entries = [r for r in old.get("embedding_speed", [])
                    if r.get("model", "").startswith("qwen3vl_emb")]
    for r in old.get("embedding_speed", []):
        if r.get("model", "").startswith("qwen3vl_emb"):
            continue
        new["embedding_speed"].append(r)
    base_qwen = next((r for r in qwen_entries if r.get("model") == QWEN_FAMILY_MODEL and "error" not in r), None)
    if base_qwen is not None:
        merged = dict(base_qwen)
        merged["model"] = "qwen3vl_emb"
        merged["truncate_dims"] = QWEN_TRUNCATE_DIMS
        merged.pop("note", None)
        new["embedding_speed"].append(merged)
    elif qwen_entries:
        # sadece hatali/eksik qwen kaydi vardi - oldugu gibi tasi, kaybetme
        new["embedding_speed"].extend(qwen_entries)
    return new


def _environment_manifest() -> dict:
    manifest = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cpu_count": os_cpu_count(),
    }
    if torch.cuda.is_available():
        manifest["cuda_version"] = torch.version.cuda
        manifest["compute_capability"] = list(torch.cuda.get_device_capability(0))
        manifest["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        manifest["gpu_total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
    try:
        import psutil
        manifest["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        manifest["ram_gb"] = None  # psutil Colab'de her zaman kurulu degil - sessizce gecme, None yaz
    return manifest


def os_cpu_count():
    import os
    return os.cpu_count()


def _model_dtype_and_attn(embedder) -> dict:
    """Model nesnesinin gercek dtype/attn_implementation'ini okumaya calisir
    (best-effort - VideoTextEmbedder alt siniflari farkli ic yapida olabilir,
    hepsi .model tasimayabilir). Bulunamazsa None yazar, sessizce atlamaz."""
    info = {"dtype": None, "attn_implementation": None}
    inner = getattr(embedder, "model", None)
    if inner is None:
        return info
    try:
        params = getattr(inner, "parameters", None)
        if callable(params):
            info["dtype"] = str(next(inner.parameters()).dtype)
    except Exception:
        pass
    config = getattr(inner, "config", None)
    if config is not None:
        info["attn_implementation"] = getattr(config, "_attn_implementation", None)
    return info


def _read_all_sample_windows(cfg, windows, n_windows_sample):
    """Ornek pencereleri BIR KEZ diskten okur (ingest/frame_io.py: tek
    seek + sequential okuma, eskiden pencere basina n_sample AYRI seek
    yapiliyordu). Tum embedding modelleri VE dedektor varyantlari ayni
    kareleri paylasir - bkz. modul docstring'i."""
    videos_dir = pathlib.Path(cfg["paths"]["videos_dir"])
    sample_windows = windows[:n_windows_sample]
    cached = []
    total_read_s = 0.0
    for i, w in enumerate(sample_windows):
        video_path = videos_dir / f"{w['video_id']}.mp4"
        t0 = time.perf_counter()
        frames = read_window_frames(str(video_path), w["t_start"], w["t_end"], n=32)
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
    olculur (video I/O suresi karisimiyor). Ilk WARMUP_WINDOWS pencere
    calistirilir ama timer'a dahil edilmez.

    Her model bittikce out["embedding_speed"]'e eklenip diske yazilir.
    Qwen ailesi icin yalnizca QWEN_FAMILY_MODEL gercekten calistirilir;
    MRL boyut varyantlari ayni forward-pass'i paylastigi icin (bkz.
    models/qwen3vl_emb.py: truncate_dim yalniz cikti vektorunu kirpar)
    tek satirda truncate_dims listesiyle raporlanir."""
    done = {r["model"] for r in out["embedding_speed"] if "error" not in r}

    for model_name in EMBED_MODELS:
        report_name = "qwen3vl_emb" if model_name == QWEN_FAMILY_MODEL else model_name
        if report_name in done:
            print(f"{report_name}: atlaniyor (onceki kosumdan sonuc mevcut)")
            continue

        timer = StageTimer()
        try:
            t0 = time.perf_counter()
            emb = get_embedder(model_name, device="cuda")
            load_s = time.perf_counter() - t0
        except Exception as exc:
            out["embedding_speed"].append({"model": report_name, "error": f"load basarisiz: {exc}"})
            _save(out_path, out)
            continue

        model_meta = _model_dtype_and_attn(emb)

        for i, item in enumerate(cached_windows):
            frames = item["frames"]
            if not frames:
                print(f"  [{report_name}] pencere {i+1}/{len(cached_windows)}: "
                     f"0 kare (onceki okumada bulunamadi), atlaniyor", flush=True)
                continue
            is_warmup = i < WARMUP_WINDOWS
            if is_warmup:
                emb.embed_video(frames)  # CUDA/cudnn isinma - zamanlanmaz
            else:
                with timer.measure("embed_video"):
                    emb.embed_video(frames)
            label = "isinma" if is_warmup else f"embed={timer.summary()['embed_video']['mean_s']:.2f}s (kumulatif ort.)"
            print(f"  [{report_name}] pencere {i+1}/{len(cached_windows)}: "
                 f"{len(frames)} kare, {label}", flush=True)

        text_queries = list(queries)[:10]
        for i, q in enumerate(text_queries):
            if i < WARMUP_WINDOWS:
                emb.embed_text(q)
            else:
                with timer.measure("embed_text"):
                    emb.embed_text(q)

        entry = {
            "model": report_name, "hardware_profile": HARDWARE_PROFILE,
            "load_s": round(load_s, 2),
            "n_windows_sampled": len(cached_windows),
            "warmup_windows": WARMUP_WINDOWS,
            **model_meta,
            "timing": timer.summary(),
        }
        if model_name == QWEN_FAMILY_MODEL:
            entry["truncate_dims"] = QWEN_TRUNCATE_DIMS
            entry["note"] = ("Tum MRL boyutlari (2048/1024/512/256) ayni transformer "
                            "forward-pass'ini paylasir - truncate_dim yalniz cikti "
                            "vektorunu kirpar (bkz. models/qwen3vl_emb.py). Bu yuzden "
                            "yalniz bir kez olculur.")
        out["embedding_speed"].append(entry)
        _save(out_path, out)
        print(f"{report_name}: load={load_s:.1f}s "
             f"embed_video_mean={timer.summary().get('embed_video', {}).get('mean_s', 0):.2f}s "
             f"embed_text_mean={timer.summary().get('embed_text', {}).get('mean_s', 0):.2f}s "
             f"dtype={model_meta['dtype']} attn={model_meta['attn_implementation']} "
             f"[kaydedildi -> {out_path}]")


def bench_detector_speed(cfg, cached_windows, out_path, out):
    """Dedektor artik VIDEO'YU TEKRAR ACMIYOR - embedding icin zaten
    okunmus kareleri (cached_windows) window_features(frames=...) ile
    paylasiyor. Eskiden her varyant kendi seek-agirlikli decode'unu
    yapiyordu; GPU'da 3 farkli buyuklukteki YOLO arasinda yalniz %3 fark
    olculmesinin sebebi buydu (decode zamani model farkini gomuyordu)."""
    detect_path = pathlib.Path(__file__).resolve().parents[1] / "ingest" / "04_detect.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("ingest_04_detect", detect_path)
    detect_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(detect_mod)

    n_sample = cfg.get("detector", {}).get("n_sample", 6)
    done = {r["variant"] for r in out["detector_speed"] if isinstance(r, dict) and "variant" in r}

    if not cached_windows:
        print("detector_speed: atlaniyor - paylasilacak kare yok (embedding zaten tamamlanmis olabilir)")
        return

    for variant_name, variant in cfg["detector"]["variants"].items():
        if variant_name in done:
            print(f"{variant_name}: atlaniyor (onceki kosumdan sonuc mevcut)")
            continue
        class_map = {int(k): v for k, v in variant["class_map"].items()}
        timer = StageTimer()
        for i, item in enumerate(cached_windows):
            w, frames = item["window"], item["frames"]
            if not frames:
                continue
            is_warmup = i < WARMUP_WINDOWS
            if is_warmup:
                detect_mod.window_features(
                    None, w["t_start"], w["t_end"], n_sample=n_sample,
                    checkpoint=variant["checkpoint"], class_map=class_map, frames=frames)
            else:
                with timer.measure("detect_window"):
                    detect_mod.window_features(
                        None, w["t_start"], w["t_end"], n_sample=n_sample,
                        checkpoint=variant["checkpoint"], class_map=class_map, frames=frames)
        out["detector_speed"].append({
            "variant": variant_name, "hardware_profile": HARDWARE_PROFILE,
            "n_windows_sampled": len(cached_windows), "warmup_windows": WARMUP_WINDOWS,
            "decode_shared_with_embedding": True,
            "timing": timer.summary(),
        })
        _save(out_path, out)
        print(f"{variant_name}: detect_window_mean="
             f"{timer.summary().get('detect_window', {}).get('mean_s', 0):.3f}s "
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
    parser.add_argument("--n-windows", type=int, default=30,
                       help="Ornek pencere sayisi (varsayilan 30 - istatistiksel "
                            "olarak 10'dan daha savunulabilir; ilk WARMUP_WINDOWS "
                            "zamanlamaya dahil edilmez).")
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
        out = migrate_legacy_schema(out)
        print(f"Onceki kismi sonuc bulundu ({out_path}), kaldigi yerden devam ediliyor.")
    else:
        out = {
            "schema_version": SCHEMA_VERSION,
            "hardware_profile": HARDWARE_PROFILE,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "environment": _environment_manifest(),
            "warmup_windows": WARMUP_WINDOWS,
            "embedding_speed": [],
            "detector_speed": [],
        }
    _save(out_path, out)

    cfg = load_config()
    windows = json.load(open(cfg["paths"]["windows"]))
    queries = build_queries()

    expected_models = {"xclip_hf_zeroshot", "siglip2_frameavg", "qwen3vl_emb"}
    already_done = {r["model"] for r in out["embedding_speed"] if "error" not in r}
    if expected_models <= already_done:
        print("Tum embedding modelleri zaten bitmis, kare okuma atlaniyor.")
        cached_windows = []
    else:
        print("Ornek pencereler bir kez okunuyor (tum modeller VE dedektor bu kareleri paylasacak)...")
        cached_windows, total_read_s = _read_all_sample_windows(cfg, windows, args.n_windows)
        out["frame_read_total_s"] = round(total_read_s, 2)
        out["frame_read_mean_per_window_s"] = round(total_read_s / max(len(cached_windows), 1), 2)
        _save(out_path, out)

    bench_embedding_speed(cached_windows, queries, out_path, out)
    bench_detector_speed(cfg, cached_windows, out_path, out)

    print(f"Tamamlandi. JSON kanit: {out_path}")


if __name__ == "__main__":
    main()
