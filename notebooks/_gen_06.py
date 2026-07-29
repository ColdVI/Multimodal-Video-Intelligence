"""notebooks/06_results_and_decision_report.ipynb'i insa edip GERCEK
calistirarak uretir. Spec SS4.7: 'CSV'lerden tablolari ve karar raporunu
uretir. Rakam URETMEZ, yalnizca TOPLAR.' Notebook 04/05 GPU kapisi
nedeniyle hic calistirilmadi (bkz. notebook 02) - bu notebook o bosluğu
GIZLEMEZ, acikca "Kanit yetersiz" olarak isaretler (spec SS14'un kendi
izin verdigi mesru sonuc)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.research.nb_build import build_and_execute

CELLS = [
("md", """# 06 - Sonuc ve karar raporu

Spec SS4.7. Bu notebook rakam URETMEZ - yalniz notebook 00-03'un GERCEK
ciktilarini toplar. Notebook 02'nin GPU kapisi nedeniyle Blok A-F (MRL
kalitesi, backend karsilastirmasi) hic COSMADI - bu SS14'un "Kanit
yetersiz" secenegini mesru kilan durumdur, gizlenmedi."""),

("code", """import json
import pathlib

OUT = pathlib.Path("artifacts/research")

artifacts = {}
for name in ["dataset_matrix.csv", "reachability_audit.csv", "alfa_telemetry_mapping.json",
            "dataset_recommendation.md", "auair_segments.parquet", "auair_telemetry.parquet",
            "selectivity_thresholds.json", "auair_audit.md", "qwen2b_gpu_cost_estimate.json",
            "qwen2b_mrl_report.md", "auair_selectivity_postgres_verification.csv",
            "pg_load_report.md"]:
    p = OUT / name
    artifacts[name] = {"exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else None}

print(json.dumps(artifacts, indent=2, ensure_ascii=False))
missing = [k for k, v in artifacts.items() if not v["exists"]]
assert not missing, f"beklenen artifactlar eksik: {missing}"
print("\\nTum beklenen artifactlar mevcut - notebook 06 bunlari TOPLAR, YENIDEN HESAPLAMAZ.")
"""),

("md", "## SS14 - Karar raporunda cevaplanacak sorular"),

("code", """gpu_cost = json.loads((OUT / "qwen2b_gpu_cost_estimate.json").read_text(encoding="utf-8"))
nb02_manifest = json.loads((OUT / "02_qwen2b_embedding_and_mrl_manifest.json").read_text(encoding="utf-8"))

answers = {
    "1_hangi_dimension": "KANIT YETERSIZ. Notebook 02'nin GPU kapisi tetiklendi "
        f"({nb02_manifest['extra']['gpu_gate_message'].splitlines()[0]}) - bu makinede "
        "hicbir boyutta (2048/1024/512/256) gercek Qwen embedding'i URETILEMEDI. "
        "Mevcut TEK gercek veri noktasi (artifacts/capera_validation.json, bu spec'in "
        "DISINDA, onceki is paketinden): Qwen3-VL-Embedding-2B'nin FULL 2048d hali "
        "CapERA'da recall@1=0.1357 - ama bu TEK boyut, MRL karsilastirmasi degil.",
    "2_boyut_kalite_kaybi": "KANIT YETERSIZ - ayni GPU kapisi nedeniyle 256/512d hic "
        "uretilmedi, mutlak/goreli fark veya bootstrap CI hesaplanamadi.",
    "3_adaptive_mrl_fayda": "KANIT YETERSIZ (bu spec kapsaminda) - ama bu depoda "
        "AYRI bir onceki is paketinden VisDrone uzerinde 28-sorgulu PILOT bir adaptive "
        "MRL kosumu MEVCUT (bench/adaptive_mrl.py, artifacts/search_runs/"
        "adaptive_mrl_visdrone_bf236d0b76/) - 150-sorgu esiginin ALTINDA, baglayici "
        "degil, bu spec'in AU-AIR/CapERA/MSR-VTT'sinden BAGIMSIZ bir sinyal.",
    "4_en_pratik_backend": "KANIT YETERSIZ - notebook 04/05 (Blok C-F, backend "
        "dogruluk/gecikme karsilastirmasi) GPU kapisi nedeniyle HIC BASLAMADI.",
    "5_en_hizli_dogru_backend": "KANIT YETERSIZ - ayni neden.",
    "6_hot_filter_alanlari": "KISMI KANIT. AU-AIR icin GERCEK secicilik esikleri "
        "notebook 01+03'te uretildi ve CANLI Postgres sorgusuyla dogrulandi "
        "(altitude_m, velocity_mps, person_count, vehicle_count - bkz. "
        "auair_selectivity_postgres_verification.csv). Ama bunlarin bir vector DB'ye "
        "payload/kolon olarak kopyalanmasinin storage maliyeti OLCULMEDI (notebook 04 "
        "calismadigi icin).",
    "7_tek_pgvector_yeterli_mi": "KANIT YETERSIZ - 2048d halfvec hassasiyet cezasi "
        "(spec SS7.3 kontrol kosusu) hic olculmedi, embedding uretilmedigi icin.",
    "8_onerilen_mimari": "KANIT YETERSIZ - spec SS14'un kendi tanimladigi 6 secenekten "
        "biri ('Kanit yetersiz') burada GECERLI VE DURUST sonuctur. Notebook 00-03 "
        "gercek, dogrulanmis (dataset erisilebilirligi, AU-AIR sema/video/hard-stop/"
        "pencereleme/telemetri/secicilik, Postgres canli dogrulama) urettiler - ama "
        "mimari karari belirleyen Blok A-F GPU kapisinda durdu.",
}
for k, v in answers.items():
    print(f"{k}:\\n  {v}\\n")

(OUT / "decision_report_answers.json").write_text(
    json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")
"""),

("md", "## Mentor ozeti (SS15 basliklarini dolduran TEK sayfa)"),

("code", """auair_seg = __import__("pandas").read_parquet(OUT / "auair_segments.parquet")
nb01_manifest = json.loads((OUT / "01_auair_download_and_validation_manifest.json").read_text(encoding="utf-8"))
nb03_manifest = json.loads((OUT / "03_postgres_metadata_telemetry_manifest.json").read_text(encoding="utf-8"))

mentor_summary = f'''# Mentor ozeti - Faz 6 MRL & Vector Backend Arastirmasi

**Kapsam:** spec SS13'teki 14 adimin GERCEKTEN calistirilan kismi: 1-5, 9-11
(mevcut is kapatildi, dataset denetimi, AU-AIR indirme/dogrulama, GPU kapisi,
Postgres semasi+yukleme). 6-8, 12 (embedding uretimi, MRL, Blok A-F, hybrid
benchmark) GPU KAPISINDA DURDU - asagida acikca isaretli.

## 1-3. Arastirilan/secilen datasetler, indirilen/yeniden kullanilan veri
AU-AIR: orijinal GitHub Pages barindirmasi kayboldu (404), web aramasiyla
GUNCEL Google Drive ID'leri bulundu ve GERCEKTEN indirildi/dogrulandi -
annotations tam (32.823 kayit), images.zip bu oturumda KISMEN indi (bkz.
manifest, ~{nb01_manifest["extra"]["images_bytes_downloaded"]/1e6:.0f} MB / ~2200 MB - baglanti hizi
kisitliydi ama bu, asagidaki dogrulamalarin HICBIRINI ETKILEMEDI).
CapERA/MSR-VTT: bu spec kapsaminda YENIDEN indirilmedi (onceki is paketinden
`data/downloads/` altinda zaten mevcut, agregatif CapERA sonuclari var).

## 4. Qwen3-VL-Embedding-2B MRL sonuclari
YOK. GPU kapisi tetiklendi: CPU'da 3 is yuku (AU-AIR 1866 pencere + CapERA
2864 video + MSR-VTT 1000 video) icin toplam tahmini
**{sum(r["cpu_estimate_h"] for r in gpu_cost):.0f} saat (~{sum(r["cpu_estimate_h"] for r in gpu_cost)/24:.0f} gun)** -
GPU'da ise **{sum(r["gpu_estimate_min"] for r in gpu_cost):.0f} dakika**. Bu makinedeki GT 1030
(4GB VRAM) icin aktif Torch CPU-only derleme (`{nb01_manifest["hardware_profile"]}`).

## 5-7. ClickHouse/Qdrant/pgvector sonuclari
YOK - hicbiri calistirilmadi (embedding'e bagimli, GPU kapisinda durdu).

## 8. PostgreSQL metadata entegrasyonu
GERCEK VE TAMAMLANDI. Gecici arastirma konteyneri (`research_postgres_faz6`,
port 5433, ana docker-compose.yml DEGISTIRILMEDI) - 5 tablo, {nb03_manifest["extra"]["row_counts"]["segments"]}
segment, satir sayisi kaynak parquet ile birebir dogrulandi. Secicilik
esikleri (numpy quantile) ile canli Postgres sorgu sonuclari TAM UYUSTU
(16/16 satir).

## 9. Hybrid sorgu sonuclari
YOK - AU-AIR'in semantic tarafi (caption yok) zaten spec SS5.3'un kendi
sinirlamasi; hybrid'in metadata/telemetri tarafi da Blok C-F calismadigi
icin olculmedi.

## 10. Storage karsilastirmasi
YOK - ayni neden.

## 11. Nihai oneri
**KANIT YETERSIZ** (spec SS14'un 6 seceneginden biri, MESRU sonuc).
Notebook 00-03 GERCEK ve DOGRULANMIS calisti (bkz. yukarida) - ama mimari
karari (hangi vector backend, hangi MRL boyutu) belirleyecek Blok A-F hic
kosulamadi. Bu depodaki tek MEVCUT ilgili sinyal, bu spec'in DISINDAKI iki
onceki is paketi: (a) CapERA'da Qwen3-VL-Embedding-2B TAM 2048d ile
recall@1=0.1357 (MRL karsilastirmasi degil, tek nokta), (b) VisDrone'da
28-sorgulu (150 esiginin altinda, baglayici degil) adaptive MRL pilotu.

## 12. Yapilmayan isler
- Notebook 04 (vector backend yukleme): GPU kapisi -> hic baslamadi.
- Notebook 05 (hybrid benchmark, Blok C-F, ~220 konfigurasyon): ayni neden.
- AU-AIR images.zip'in kalan ~%53'u (baglanti hizi kisitliydi).
- ALFA tam CSV kolon esmesi (protokol - MAVLink 2.0 - dogrulandi, tam alan
  listesi icin gercek sequence dosyasi gerekir, future_work.md'de).

## 13. Notebook ve artifact yollari
Asagidaki bolumde tam liste.
'''

(OUT / "mentor_summary.md").write_text(mentor_summary, encoding="utf-8")
print(mentor_summary)
"""),

("md", "## Artifact envanteri (SS4.7 - toplama, uretim degil)"),

("code", """import subprocess

all_artifacts = sorted(str(p) for p in OUT.glob("*") if p.is_file())
notebook_files = sorted(str(p) for p in pathlib.Path("notebooks").glob("*.ipynb"))
print("=== artifacts/research/ ===")
for a in all_artifacts:
    print(" ", a)
print()
print("=== notebooks/ ===")
for n in notebook_files:
    print(" ", n)
"""),
]

if __name__ == "__main__":
    out_path = pathlib.Path("notebooks/06_results_and_decision_report.ipynb")
    nb = build_and_execute(CELLS, out_path, timeout=120)
    print(f"\\nNotebook yazildi: {out_path} ({len(nb.cells)} hucre)")
