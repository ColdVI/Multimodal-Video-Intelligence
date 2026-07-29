"""Bilinmeyen bir dataset dizinini analiz edip insan-okur ozet + bir
dataset_adapters/<isim>.py adapter iskeleti uretir.

NEDEN: ekip arkadaslari farkli veri yukleyecek (bkz. dataset_prompt_paketi.md).
Her biri icin elle adapter yazmak yerine, once veriyi TANIYAN bir arac -
dosya tipleri, video/kare-dizini yapisi, anotasyon formati tahmini, caption
varligi, ve KIRMIZI BAYRAKLAR (fps manifesti yok, keyframe araligi genis,
lisans dosyasi yok, isim uyusmazligi, tek sayili cozunurluk).

SALT OKUNUR: girdi dizinindeki hicbir dosyayi degistirmez/silmez. Buyuk
dizinlerde HER dosyayi derinlemesine incelemez - uzanti sayimi tam dizin
taramasi (ucuz, sadece stat), derin inceleme (video metadata, anotasyon
icerik sniffing) ilk 100 + rastgele 100 dosyalik bir orneklemle sinirlidir.

KEYFRAME ARALIGI TAHMINI: ffprobe kurulu degilse (bu depoda degil - yalniz
ffmpeg var, bkz. common ffmpeg cozumleme deseni) kesin deger olculemez.
Bunun yerine bir ORAN tahmini yapilir: rastgele bir noktaya seek maliyeti /
sequential kare decode maliyeti = yaklasik "kac kare geriye gidildigi".
Bu KESIN degil, acikca "tahmin" etiketlenir - ffprobe kuruluysa gercek
deger onceliklidir.
"""
import argparse
import hashlib
import json
import pathlib
import random
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
ANNOTATION_EXTS = {".json", ".txt", ".csv", ".xml"}
LICENSE_NAMES = {"license", "license.txt", "license.md", "copying", "copying.txt"}


def scan_extensions(root: pathlib.Path) -> Counter:
    """Tam dizin taramasi ama yalniz stat - dosya icerigi okunmaz, buyuk
    dizinlerde de ucuzdur."""
    counts = Counter()
    for p in root.rglob("*"):
        if p.is_file():
            counts[p.suffix.lower()] += 1
    return counts


def classify_structure(ext_counts: Counter) -> str:
    has_video = any(ext_counts.get(e, 0) > 0 for e in VIDEO_EXTS)
    has_images = any(ext_counts.get(e, 0) > 0 for e in IMAGE_EXTS)
    if has_video and has_images:
        return "video+kare-dizini (ikisi de var)"
    if has_video:
        return "video dosyalari"
    if has_images:
        return "kare dizinleri (video yok)"
    return "bilinmiyor (ne video ne kare bulundu)"


def sample_paths(paths: list, first_n: int = 100, random_n: int = 100, seed: int = 0) -> list:
    """Ilk first_n (sirali) + geri kalandan rastgele random_n - buyuk
    dizinlerde hepsini derinlemesine incelemeyi onler."""
    paths = sorted(paths)
    first = paths[:first_n]
    remaining = paths[first_n:]
    rng = random.Random(seed)
    extra = rng.sample(remaining, min(random_n, len(remaining))) if remaining else []
    return first + extra


def probe_video(path: pathlib.Path) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"path": str(path), "error": "acilamadi"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")

    # sequential decode maliyeti (kalibrasyon icin)
    t0 = time.perf_counter()
    for _ in range(min(10, n_frames)):
        cap.read()
    seq_s = (time.perf_counter() - t0) / max(min(10, n_frames), 1)

    # rastgele seek maliyeti (n_frames yeterince buyukse)
    keyframe_estimate = None
    if n_frames > 20:
        target = n_frames // 2
        t0 = time.perf_counter()
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        cap.read()
        seek_s = time.perf_counter() - t0
        if seq_s > 0:
            keyframe_estimate = {
                "estimate_note": "KESIN DEGIL - ffprobe kurulu degil, seek/sequential "
                                "oran tahmini. Gercek deger icin ffprobe kurup "
                                "'-show_frames -show_entries frame=pict_type' kullanin.",
                "single_seek_cost_s": round(seek_s, 4),
                "sequential_frame_cost_s": round(seq_s, 4),
                "estimated_frames_reseeked": round(seek_s / seq_s, 1),
                "estimated_keyframe_interval_s": round((seek_s / seq_s) / fps, 2) if fps else None,
            }
    cap.release()

    return {
        "path": str(path), "fps": fps, "n_frames": n_frames,
        "width": w, "height": h, "codec": codec,
        "duration_s": round(n_frames / fps, 2) if fps else None,
        "odd_height": bool(h % 2),
        "keyframe_estimate": keyframe_estimate,
    }


def guess_annotation_format(path: pathlib.Path) -> str:
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore")[:200_000])
            if isinstance(data, dict) and {"images", "annotations", "categories"} <= set(data.keys()):
                return "COCO-json"
            if isinstance(data, list) and data and isinstance(data[0], dict):
                keys = set(data[0].keys())
                if keys & {"caption", "captions", "text", "description"}:
                    return "caption-json (liste)"
                return "json (bilinmeyen liste semasi)"
            if isinstance(data, dict):
                sample_val = next(iter(data.values()), None)
                if isinstance(sample_val, list) and sample_val and isinstance(sample_val[0], str):
                    return "caption-json (id->liste)"
                return "json (bilinmeyen dict semasi)"
            return "json (tanimlanamadi)"

        if path.suffix.lower() == ".xml":
            root = ET.parse(path).getroot()
            if root.tag == "annotation" or root.find(".//object") is not None:
                return "PascalVOC-xml"
            return "xml (bilinmeyen sema)"

        if path.suffix.lower() in (".txt", ".csv"):
            first_line = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
            if "," in first_line:
                n_fields = len(first_line.split(","))
                if n_fields in (9, 10):
                    return "MOT-csv (frame,id,x,y,w,h,score,cat,...)"
                return f"csv ({n_fields} alan, bilinmeyen sema)"
            n_fields = len(first_line.split())
            if n_fields == 5:
                return "YOLO-txt (class x_center y_center w h)"
            return f"txt ({n_fields} alan, bilinmeyen sema)"
    except Exception as exc:
        return f"okunamadi: {type(exc).__name__}"
    return "bilinmiyor"


def detect_captions(annotation_samples: list) -> dict:
    caption_like = [p for p in annotation_samples if "caption" in guess_annotation_format(p).lower()]
    return {
        "has_captions": bool(caption_like),
        "caption_like_files_in_sample": len(caption_like),
        "sample_size": len(annotation_samples),
    }


def compute_manifest(root: pathlib.Path, ext_counts: Counter) -> dict:
    total_size = 0
    fingerprint_parts = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            size = p.stat().st_size
            total_size += size
            fingerprint_parts.append(f"{p.relative_to(root)}:{size}")
    fingerprint = hashlib.sha256("\n".join(fingerprint_parts).encode("utf-8")).hexdigest()
    return {
        "total_size_bytes": total_size,
        "total_size_gb": round(total_size / (1024 ** 3), 3),
        "file_count": sum(ext_counts.values()),
        "ext_counts": dict(ext_counts),
        "structure_fingerprint_sha256": fingerprint,
    }


def find_license_file(root: pathlib.Path):
    for p in root.iterdir():
        if p.is_file() and p.name.lower() in LICENSE_NAMES:
            return p
    return None


def find_fps_manifest(root: pathlib.Path, sample_json: list) -> bool:
    for p in sample_json:
        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore")[:50_000])
        except Exception:
            continue
        if isinstance(data, dict):
            if "fps" in data:
                return True
            first_val = next(iter(data.values()), None)
            if isinstance(first_val, dict) and "fps" in first_val:
                return True
    return False


def find_frame_directories(root: pathlib.Path) -> list:
    """Video dosyasi yerine kare dizinleri kullanan datasetlerde (VisDrone
    gibi) 'sekans' asil olarak dogrudan .jpg/.png iceren en-alt dizindir -
    ust duzey klasor adlari (ör. 'sequences/', 'annotations/') bir sekans
    DEGIL, sadece organizasyon klasoru. Isim-uyusmazligi kontrolunu bunlarla
    degil, gercek kare-dizini yapraklariyla yapiyoruz."""
    dirs_with_images = set()
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            dirs_with_images.add(p.parent)
    return sorted(dirs_with_images)


def check_name_mismatch(video_or_frame_names: list, annotation_names: list) -> list:
    video_stems = {pathlib.Path(n).stem for n in video_or_frame_names}
    ann_stems = {pathlib.Path(n).stem for n in annotation_names}
    if not video_stems or not ann_stems:
        return []
    mismatched = video_stems - ann_stems
    return sorted(mismatched)[:10]  # ilk 10 ornek yeterli, hepsini basma


def red_flags(report: dict) -> list:
    flags = []
    if not report.get("fps_manifest_found"):
        flags.append("fps manifestte YOK - sabit 25 fps varsaymak yasak, "
                     "gercek fps kaynagi bulunmali")
    for v in report.get("video_samples", []):
        ke = v.get("keyframe_estimate")
        if ke and ke.get("estimated_keyframe_interval_s") and ke["estimated_keyframe_interval_s"] > 5.0:
            flags.append(f"{v['path']}: tahmini keyframe araligi "
                         f"{ke['estimated_keyframe_interval_s']}s (>5s) - decode pahali olacak")
        if v.get("odd_height"):
            flags.append(f"{v['path']}: yukseklik tek sayi ({v['height']}) - "
                         "libx264/yuv420p kirilir (VisDrone'da 1904x1071 ile yasandi)")
    if not report.get("license_file_found"):
        flags.append("Kokte LICENSE dosyasi YOK - TICARI KULLANIM BILINMIYOR, "
                     "model karti/repo'dan teyit edilmeden kullanilmamali")
    mismatch = report.get("name_mismatch", [])
    if mismatch:
        flags.append(f"{len(mismatch)}+ video/kare dizini icin anotasyon eslesmiyor "
                     f"(orn: {mismatch[:3]})")
    return flags


def generate_adapter_skeleton(dataset_name: str, report: dict) -> str:
    class_name = "".join(w.capitalize() for w in dataset_name.replace("-", "_").split("_"))
    fps_note = ("manifestten okunuyor" if report.get("fps_manifest_found")
               else "BULUNAMADI - TODO: gercek fps kaynagini belirleyin, 25 varsaymayin")
    has_captions = report.get("caption_detection", {}).get("has_captions", False)
    return f'''"""scripts/inspect_dataset.py tarafindan uretilen ISKELET - calisir
degil, TODO'lari doldurun. Kaynak dizin analizi:
  yapi: {report.get("structure")}
  fps: {fps_note}
  anotasyon formati tahmini: {report.get("annotation_format_guess")}
  caption tespit edildi: {has_captions}
"""
from pathlib import Path

from dataset_adapters.base import DatasetAdapter


class {class_name}Adapter(DatasetAdapter):
    name = "{dataset_name}"
    has_captions = {has_captions}

    def list_sequences(self) -> list:
        # TODO: kaynak dizini tara, sekans/video kimliklerini dondur
        raise NotImplementedError

    def load_video(self, seq_id: str) -> Path:
        # TODO: seq_id -> mp4 yolu. Yapi "{report.get("structure")}" ise
        # kare dizini oldugunda once ingest/01_frames_to_video.py deseniyle
        # mp4'e cevirmeniz gerekebilir.
        raise NotImplementedError

    def fps(self, seq_id: str) -> float:
        # TODO: {fps_note}
        raise NotImplementedError

    def ground_truth(self, seq_id: str) -> dict:
        # TODO: anotasyon formati "{report.get("annotation_format_guess")}" -
        # bu formata gore sorgu -> zaman araligi turetin. Caption varsa
        # (has_captions=True) GT protokolu FARKLI olabilir (caption->video
        # eslesmesi), anotasyon-turevi degil - eval raporunda ACIKCA belirtin.
        raise NotImplementedError

    def license(self) -> str:
        # TODO: {"LICENSE dosyasi bulunamadi - model karti/repo sayfasindan TEYIT EDIN" if not report.get("license_file_found") else "kokteki LICENSE dosyasindan okuyun"}
        return "UNKNOWN"
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Analiz edilecek dataset dizini")
    ap.add_argument("--name", required=True, help="Adapter dosyasi icin dataset adi (ör. capera)")
    ap.add_argument("--out-dir", default="dataset_adapters", help="Adapter iskeletinin yazilacagi dizin")
    ap.add_argument("--force", action="store_true", help="Var olan adapter dosyasinin uzerine yaz")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    if not root.exists():
        print(f"HATA: {root} yok")
        raise SystemExit(1)

    print(f"Taraniyor: {root}")
    ext_counts = scan_extensions(root)
    structure = classify_structure(ext_counts)

    all_video = [p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    all_ann = [p for p in root.rglob("*") if p.suffix.lower() in ANNOTATION_EXTS]
    video_sample = sample_paths(all_video, first_n=20, random_n=20)
    ann_sample = sample_paths(all_ann, first_n=50, random_n=50)
    json_sample = [p for p in ann_sample if p.suffix.lower() == ".json"]

    video_reports = [probe_video(p) for p in video_sample]
    ann_format_guesses = Counter(guess_annotation_format(p) for p in ann_sample) if ann_sample else Counter()
    caption_detection = detect_captions(ann_sample)
    manifest = compute_manifest(root, ext_counts)
    license_file = find_license_file(root)
    fps_manifest_found = find_fps_manifest(root, json_sample)
    frame_dirs = find_frame_directories(root)
    name_mismatch = check_name_mismatch(
        [p.stem for p in all_video] or [d.name for d in frame_dirs],
        [p.stem for p in all_ann],
    ) if all_ann else []

    report = {
        "root": str(root),
        "structure": structure,
        "ext_counts": dict(ext_counts),
        "manifest": manifest,
        "video_samples": video_reports,
        "annotation_format_guess": ann_format_guesses.most_common(1)[0][0] if ann_format_guesses else "anotasyon bulunamadi",
        "annotation_format_distribution": dict(ann_format_guesses),
        "caption_detection": caption_detection,
        "license_file_found": str(license_file) if license_file else None,
        "fps_manifest_found": fps_manifest_found,
        "name_mismatch": name_mismatch,
    }
    report["red_flags"] = red_flags(report)

    print(f"\n=== Yapi ===\n  {structure}")
    print(f"\n=== Dosya sayilari ===")
    for ext, n in ext_counts.most_common(10):
        print(f"  {ext or '(uzantisiz)'}: {n}")
    print(f"\n=== Boyut === \n  {manifest['total_size_gb']} GB, {manifest['file_count']} dosya")
    print(f"\n=== Anotasyon formati tahmini ===\n  {report['annotation_format_guess']} "
         f"(dagilim: {report['annotation_format_distribution']})")
    print(f"\n=== Caption ===\n  {caption_detection}")
    if video_reports:
        v0 = video_reports[0]
        print(f"\n=== Ornek video (ilk) ===\n  {v0}")
    print(f"\n=== KIRMIZI BAYRAKLAR ===")
    if report["red_flags"]:
        for flag in report["red_flags"]:
            print(f"  [!] {flag}")
    else:
        print("  (yok)")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = out_dir / f"{args.name}.py"
    if adapter_path.exists() and not args.force:
        print(f"\nAdapter iskeleti YAZILMADI - {adapter_path} zaten var (--force ile üzerine yazın)")
    else:
        adapter_path.write_text(generate_adapter_skeleton(args.name, report), encoding="utf-8")
        print(f"\nAdapter iskeleti: {adapter_path}")

    report_path = out_dir / f"{args.name}_inspect_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Tam rapor: {report_path}")


if __name__ == "__main__":
    main()
