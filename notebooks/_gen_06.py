"""notebooks/06_results_and_decision_report.ipynb'i insa edip GERCEK
calistirarak uretir. Spec SS4.7: 'CSV'lerden tablolari ve karar raporunu
uretir. Rakam URETMEZ, yalnizca TOPLAR.' Colab handoff sonrasi: bu notebook
artik 'nihai arastirma sonucu' degil, 'paket durumu' raporlar - GPU/backend
fazlari bu ortamda (Colab disi) hic calismadi, bu ACIKCA yazilir. Colab'da
00->01->02->03->04->05->06 sirasiyla GERCEKTEN calistirildiktan sonra bu
notebook GERCEK sonuclari toplayacak sekilde tasarlandi."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.research.nb_build import build_and_execute
from src.research.colab_paths import COLAB_BOOTSTRAP_CELL

CELLS = [
("code", COLAB_BOOTSTRAP_CELL),
("md", """# 06 - Sonuc ve karar raporu / paket durumu

Spec SS4.7 + Colab handoff. Bu notebook rakam URETMEZ - yalniz notebook
00-05'in GERCEK ciktilarini toplar. Asagidaki calistirma bu depodaki
(Colab disi, GPU'suz) durumu yansitir - Colab'da GPU+backend fazlari
GERCEKTEN calistiktan sonra ayni notebook GERCEK arastirma sonuclarini
toplayacaktir."""),

("code", """import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path.cwd()))
from src.research import colab_paths

OUT = colab_paths.research_root()

REQUIRED = ["dataset_matrix.csv", "reachability_audit.csv", "alfa_telemetry_mapping.json",
           "dataset_recommendation.md", "auair_segments.parquet", "auair_telemetry.parquet",
           "selectivity_thresholds.json", "auair_audit.md", "dataset_download_manifest.json"]
OPTIONAL_PHASE2 = ["qwen2b_mrl_report.md", "auair_selectivity_postgres_verification.csv",
                  "pg_load_report.md", "environment_capability_report.json",
                  "ingest_report.csv", "vector_database_results.csv", "vector_database_report.md"]

artifacts = {}
for name in REQUIRED + OPTIONAL_PHASE2:
    p = OUT / name
    artifacts[name] = {"exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else None}

print(json.dumps(artifacts, indent=2, ensure_ascii=False))
missing_required = [k for k in REQUIRED if not artifacts[k]["exists"]]
assert not missing_required, f"ZORUNLU artifactlar eksik: {missing_required}"
missing_optional = [k for k in OPTIONAL_PHASE2 if not artifacts[k]["exists"]]
print(f"\\nZorunlu (00-01) artifactlarin hepsi mevcut. "
     f"Faz2 (02-05) artifactlarindan eksik olanlar: {missing_optional or 'yok'}")
"""),

("md", "## SS14 - Karar raporunda cevaplanacak sorular (paket durumu)"),

("code", """def _read_json(name):
    p = OUT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

def _read_csv_rowcount(name):
    p = OUT / name
    if not p.exists():
        return None
    import pandas as pd
    return len(pd.read_csv(p))

nb02_manifest = _read_json("02_qwen2b_embedding_and_mrl_manifest.json")
nb04_manifest = _read_json("04_vector_backend_loading_manifest.json")
nb05_manifest = _read_json("05_hybrid_query_benchmark_manifest.json")

gpu_ran = bool(nb02_manifest and nb02_manifest["extra"].get("gpu_available"))
backend_rows = nb05_manifest["extra"]["n_bench_rows"] if nb05_manifest else 0

_NOT_RUN = ("KANIT YOK - bu ortamda (Colab disi, GPU yok / backend kurulamadi) "
           "bu adim henuz GERCEKTEN calismadi. COLAB_RUNBOOK.md'deki sirayla "
           "calistirildiktan sonra bu hucre GERCEK sonucu toplayacak.")

answers = {
    "1_hangi_dimension": _NOT_RUN if not gpu_ran else "notebook 02'nin GERCEK Colab sonucuna bakin",
    "2_boyut_kalite_kaybi": _NOT_RUN if not gpu_ran else "notebook 02'nin GERCEK Colab sonucuna bakin",
    "3_adaptive_mrl_fayda": "Bu spec disi, VisDrone'da 28-sorgulu PILOT bir adaptive MRL "
        "kosumu bu depoda ONCEDEN mevcut (bench/adaptive_mrl.py) - 150-sorgu esiginin "
        "altinda, baglayici degil. " + (_NOT_RUN if not backend_rows else ""),
    "4_en_pratik_backend": _NOT_RUN if not backend_rows else "notebook 05'in GERCEK Colab sonucuna bakin",
    "5_en_hizli_dogru_backend": _NOT_RUN if not backend_rows else "notebook 05'in GERCEK Colab sonucuna bakin",
    "6_hot_filter_alanlari": "KISMI KANIT (bu ortamda calisti): AU-AIR icin GERCEK "
        "secicilik esikleri notebook 01+03'te uretildi ve CANLI Postgres sorgusuyla "
        "dogrulandi. Vector DB'ye kopyalama storage maliyeti icin: " + (
            _NOT_RUN if not backend_rows else "notebook 05 sonucuna bakin"),
    "7_tek_pgvector_yeterli_mi": _NOT_RUN if not backend_rows else "notebook 05'in GERCEK Colab sonucuna bakin",
    "8_onerilen_mimari": _NOT_RUN if not (gpu_ran and backend_rows) else "asagidaki mentor ozetine bakin",
}
for k, v in answers.items():
    print(f"{k}:\\n  {v}\\n")

(OUT / "decision_report_answers.json").write_text(
    json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")
"""),

("md", "## Mentor ozeti (SS15 basliklarini dolduran TEK sayfa - PAKET DURUMU)"),

("code", """auair_seg = __import__("pandas").read_parquet(OUT / "auair_segments.parquet")
nb01_manifest = json.loads((OUT / "01_auair_download_and_validation_manifest.json").read_text(encoding="utf-8"))
nb03_manifest = json.loads((OUT / "03_postgres_metadata_telemetry_manifest.json").read_text(encoding="utf-8"))
env_report = _read_json("environment_capability_report.json")

mentor_summary = f'''# Mentor ozeti - Faz 6 MRL & Vector Backend Arastirmasi (Colab handoff)

**Bu calistirma Colab DEGIL** (bu depo, GPU'suz yerel makine) - asagidaki
durum bu ortamda GERCEKTEN dogrulanan altyapiyi gosterir. GPU/backend
arastirma SONUCLARI Colab'da COLAB_RUNBOOK.md'ye gore calistirilinca uretilir.

## 1-3. Datasetler
AU-AIR: orijinal GitHub Pages barindirmasi kayboldu (404), web aramasiyla
GUNCEL Google Drive ID'leri bulundu ve GERCEKTEN indirildi/dogrulandi -
annotations tam ({len(auair_seg)} pencere uretildi), images.zip Colab'da
tamamlanmali (dataset_download_manifest.json'da resume/sha256/lisans kayitli).

## 4. Qwen3-VL-Embedding-2B + MRL (notebook 02)
Bu ortamda calismadi (gpu_available={gpu_ran}) - GPU gerektirir, kod HAZIR
(checkpoint/resume + 1024/512/256 turetme dahil), Colab GPU runtime'inda
calistirilmali.

## 5-7. ClickHouse/Qdrant/pgvector (notebook 04-05)
Bu ortamda calismadi ({backend_rows} benchmark satiri uretildi) - install/
start/health-check kodu HAZIR ve GERCEKTEN denendi (hepsi dogru sekilde
`environment_unavailable` olarak isaretlendi, sahte sonuc YOK). Colab
CPU/high-RAM runtime'inda apt-get/static-binary kurulumlariyla calisir.

## 8. PostgreSQL metadata entegrasyonu (notebook 03)
GERCEK VE TAMAMLANDI (bu ortamda da calisir - GPU/Colab gerektirmez).
{nb03_manifest["extra"]["row_counts"]["segments"]} segment, satir sayisi
kaynak parquet ile birebir dogrulandi. Secicilik esikleri (numpy quantile)
ile canli Postgres sorgu sonuclari TAM UYUSTU.

## 9-10. Hybrid sorgu / storage karsilastirmasi
YOK - notebook 05'e bagimli, bu ortamda calismadi.

## 11. Nihai oneri
**Bu calistirmada: KANIT YOK (paket hazirlama asamasi).** Colab'da
COLAB_RUNBOOK.md sirasiyla calistirildiktan sonra notebook 06 GERCEK
sonuclari toplayacak.

## 12. Yapilmayan isler (bu ortamda - Colab'da yapilacak)
- Notebook 02: GPU embedding uretimi.
- Notebook 04-05: backend kurulum + benchmark (kod hazir, denendi, ortam yok).
- AU-AIR images.zip'in tamami.
- ALFA tam CSV kolon eslemesi (future_work.md'de).

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
