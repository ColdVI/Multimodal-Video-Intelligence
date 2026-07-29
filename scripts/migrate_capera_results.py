"""data/downloads/capera/all_results.json'daki GERCEK, ONCEDEN URETILMIS
sonuclari (kullanici Drive'dan yerel repoya kopyaladi) YENIDEN OLCMEDEN
artifacts/capera_validation.json'a tasir - ortak JSON semasina (protocol/
dataset_manifest/known_gaps/results) uydurur. Hicbir embedding uretmez,
hicbir model calistirmaz - bu script SALT TASIMA/BELGELEME."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import dataclasses

from datasets.capera import CapERAAdapter

SOURCE_PATH = pathlib.Path("data/downloads/capera/all_results.json")
OUT_PATH = pathlib.Path("artifacts/capera_validation.json")

KNOWN_GAPS = [
    ("Per-model embedding/checkpoint dosyalari (emb/<model>/chunk_embeddings.npz, "
     "query_embeddings.npy, chunk_meta.json, per_category_metrics.json, "
     "per_query_ranks.csv) bu depoya KOPYALANMADI - yalniz all_results.json'daki "
     "agregatif satirlar mevcut. Per-kategori veya per-sorgu kirilim, ham "
     "embedding vektoru bu artifact'ta YOK."),
    ("capera_retrieval_pipeline.ipynb'de 4 model adaptoru var (Qwen3-VL-"
     "Embedding-2B/8B, ebind-audio-vision, VideoCLIP-XL) ama all_results.json'da "
     "yalniz 3 sonuc var - ebind-audio-vision KOSULMAMIS veya sonucu "
     "kaydedilmemis, sebebi bu dosyalardan belli degil."),
    ("all_results.json'daki n_videos=2863, adapter'in (datasets/capera.py) "
     "gercek 2864 kaydiyla 1 fark - notebook hucre 23, 'train__PoliceChase_048.mp4' "
     "adli 1 video dosyasini Colab diskinde bulamayip degerlendirme disi "
     "birakmis (dogrulandi: notebook kodu). Bug degil, ortam-ozel bir "
     "dosya-eksikligi."),
    ("Video dosyalarinin kendisi (mp4'ler) bu depoda YOK - notebook onlari "
     "Colab'in gecici diskine aciyordu, Drive'a geri kaydedilmemis. Bu yuzden "
     "bu depoda CapERA icin yeni bir embedding/dogrulama kosumu su an "
     "YAPILAMAZ (video verisi eksik, GPU da yok)."),
]


def build_migrated_artifact(raw_results: list, dataset_manifest: dict) -> dict:
    """Saf fonksiyon - dosya I/O yok, testte dogrudan cagrilabilir."""
    return {
        "protocol": ("CapERA aerial-video captioning T2V retrieval - video basina 5 "
                    "caption, her caption AYRI sorgu, video 5sn (alt-pencere YOK, "
                    "MSR-VTT'ye benzer whole-clip retrieval). ClickHouse'a YAZILMAZ "
                    "(retrieval_backend=artifact_matrix)."),
        "source": (f"{SOURCE_PATH} (kullanici tarafindan Google Drive'daki "
                  "capera_dataset_model_test klasorunden kopyalandi - bu script "
                  "HICBIR embedding/model YENIDEN URETMEDI, yalniz mevcut "
                  "sonuclari ortak semaya tasidi)."),
        "dataset_manifest": dataset_manifest,
        "known_gaps": KNOWN_GAPS,
        "results": {row["model"]: row for row in raw_results},
    }


def main():
    if not SOURCE_PATH.exists():
        print(f"HATA: {SOURCE_PATH} yok - once CapERA klasorunu data/downloads/capera/'ya kopyalayin.")
        raise SystemExit(1)

    raw_results = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    dataset_manifest = dataclasses.asdict(CapERAAdapter().manifest())
    out = build_migrated_artifact(raw_results, dataset_manifest)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Tasindi: {len(raw_results)} model sonucu -> {OUT_PATH}")
    for row in raw_results:
        print(f"  {row['model']}: R@1={row['recall_at_1']:.3f} R@5={row['recall_at_5']:.3f} "
             f"R@10={row['recall_at_10']:.3f} MRR={row['mrr']:.3f} "
             f"(n_videos={row['n_videos']}, n_queries={row['n_queries']})")


if __name__ == "__main__":
    main()
