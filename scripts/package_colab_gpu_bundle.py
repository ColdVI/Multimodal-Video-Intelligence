"""scripts/colab_gpu_bench.py'nin ihtiyac duydugu dosyalari (bench subset
videolari, pencereler, YOLO ozellikleri, dedektor checkpoint'leri) tek bir
zip'e toplar - Colab'e yuklemek icin. Model agirliklari (X-CLIP/SigLIP2/
Qwen) DAHIL EDILMEZ; Colab'de internet var, o modeller HF'den kendisi
indirilir (offline_mode kapali kalir, Colab air-gapped degil)."""
import pathlib
import sys
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_config

INCLUDE_FILES = [
    "yolo26x.pt",
    "weights/yolo_visdrone/best.pt",
    "weights/yolo_visdrone_s/best.pt",
]
def main():
    cfg = load_config()
    repo = pathlib.Path(__file__).resolve().parents[1]
    videos_dir = pathlib.Path(cfg["paths"]["videos_dir"])

    out_path = repo / "artifacts" / "colab_gpu_bundle.zip"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # config.yaml: paths.* zaten repo-koku-goreli (ör. "data/windows.json") -
    # zip icinde ayni goreli yolu korur.
    include_data_paths = [cfg["paths"]["windows"], cfg["paths"]["features"],
                          cfg["paths"]["manifest"]]

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in INCLUDE_FILES:
            src = repo / name
            if src.exists():
                zf.write(src, arcname=name)
            else:
                print(f"uyari: {src} yok, atlaniyor")

        for rel_name in include_data_paths:
            src = repo / rel_name
            if src.exists():
                zf.write(src, arcname=rel_name)
            else:
                print(f"uyari: {src} yok, atlaniyor")

        for video in sorted(videos_dir.glob("*.mp4")):
            zf.write(video, arcname=f"data/raw/videos/{video.name}")

    print(f"paket: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print("Bu zip'i BIR KERE Google Drive'a yukleyin (Colab'in yerel /content "
         "diski her runtime kopusunda silinir, Drive kalicidir). Her Colab "
         "oturumunda: drive.mount('/content/drive'), sonra zip'i Drive'dan "
         "yerel /content diskine kopyalayip acin (Drive-FUSE uzerinden dogrudan "
         "calistirmak cv2 frame-seek icin cok yavastir). Sonuc JSON'u da "
         "--out ile bir Drive yoluna yazdirin ki kismi sonuc kopma durumunda "
         "kaybolmasin - detay: scripts/colab_gpu_bench.py docstring'i.")


if __name__ == "__main__":
    main()
