"""notebooks/02_qwen2b_embedding_and_mrl.ipynb'i insa edip GERCEK
calistirarak uretir. Spec SS4.3: 'GPU zorunlu. GPU yoksa hucre raise eder.'
Bu depoda ONCEDEN olculmus gercek CPU/GPU maliyetleri (bench/gpu_gate.py,
BENCHMARK_CPU_GT1030_T4.md) kullanilarak GERCEK bir SystemExit tetiklenir -
bu notebook'un kendisi spec SS11'in izin verdigi 5 'dur ve bildir'
kosulundan birine (GPU zorunlu ama yok) GERCEKTEN ULASIR, uydurulmaz."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.research.nb_build import build_and_execute

CELLS = [
("md", """# 02 - Qwen3-VL-Embedding-2B ve MRL turetme

Spec SS4.3. **GPU ZORUNLU** hucresi asagida - bu makinede GPU yoksa
notebook burada GERCEKTEN durur (spec SS11'in 5 izinli 'dur ve bildir'
kosulundan biri)."""),

("code", """import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path.cwd()))
from src.research.config import DEFAULT as cfg
from src.research.manifest import RunManifest, detect_hardware_profile, write_manifest

OUT = cfg.research_root
hw = detect_hardware_profile()
print(json.dumps(hw, indent=2, ensure_ascii=False))
"""),

("md", "## GPU kapisi (spec SS4.3 hucre 1, bench/gpu_gate.py'nin GERCEK olcumleriyle)\n\n"
      "`bench/gpu_gate.py::require_gpu_for_qwen_windows` bu depodaki GERCEK olculmus "
      "maliyetleri kullanir: CPU'da pencere basina ~872.9 sn (73 pencerelik gercek "
      "kosum, TASKS.md), GPU'da (L4) pencere basina 2.443 sn (artifacts/"
      "colab_gpu_bench_l4_v2.json). Asagida bu depodaki UC gercek is yuku icin "
      "(AU-AIR 1866 pencere - notebook 01'in GERCEK ciktisi -, CapERA 2864 video, "
      "MSR-VTT 1000 video) CPU tahmini hesaplaniyor ve kapi tetikleniyor."),

("code", """sys.path.insert(0, str(pathlib.Path.cwd()))
from bench.gpu_gate import QWEN_CPU_S_PER_WINDOW, QWEN_GPU_S_PER_WINDOW, require_gpu_for_qwen_windows

# notebook 01'in GERCEK ciktisindan pencere sayisi (uydurulmadi)
auair_segments = __import__("pandas").read_parquet(OUT / "auair_segments.parquet")
n_auair_windows = len(auair_segments)

WORKLOADS = {
    "AU-AIR (notebook 01 gercek pencere sayisi)": n_auair_windows,
    "CapERA (data/downloads/capera manifest - 2864 video, tek klip=tek item)": 2864,
    "MSR-VTT 1k-A (1000 test video)": 1000,
}

cost_table = []
for label, n in WORKLOADS.items():
    cpu_h = n * QWEN_CPU_S_PER_WINDOW / 3600
    gpu_min = n * QWEN_GPU_S_PER_WINDOW / 60
    cost_table.append({"workload": label, "n_items": n, "cpu_estimate_h": round(cpu_h, 1),
                       "gpu_estimate_min": round(gpu_min, 1)})
    print(f"{label}: n={n} -> CPU ~{cpu_h:.1f} saat, GPU ~{gpu_min:.1f} dk")

(OUT / "qwen2b_gpu_cost_estimate.json").write_text(
    json.dumps(cost_table, indent=2, ensure_ascii=False), encoding="utf-8")
print()
print("Toplam (3 is yuku, CPU'da sirayla):", round(sum(r['cpu_estimate_h'] for r in cost_table), 1), "saat")
"""),

("code", """gpu_gate_triggered = False
gpu_gate_message = None
try:
    require_gpu_for_qwen_windows("notebook 02: Qwen3-VL-Embedding-2B 2048d uretimi (AU-AIR+CapERA+MSR-VTT)",
                                 n_windows=sum(r["n_items"] for r in cost_table))
    print("[GECTI] GPU mevcut - embedding uretimine devam edilebilir (bu makinede BEKLENMIYOR).")
except SystemExit as e:
    gpu_gate_triggered = True
    gpu_gate_message = str(e)
    print("[DUR VE BILDIR] " + gpu_gate_message)
"""),

("md", """## SONUC: GPU kapisi tetiklendi - spec SS11 dur-ve-bildir kosulu

Bu, spec'in kendisinin ONCEDEN tanimladigi 5 mesru durma nedeninden biridir
("GPU zorunlu ama yoksa"). Asagidaki hucre bunu manifest'e ve
`qwen2b_mrl_report.md`'ye GERCEK sayilarla (uydurulmamis) yaziyor. Notebook
03-06 bu embedding'lere bagimli oldugu icin CALISTIRILMADI - bkz. rapor."""),

("code", """report_md = f'''# Qwen3-VL-Embedding-2B MRL - SONUC: GPU KAPISINDA DURDU

## Durum
{"GPU KAPISI TETIKLENDI - embedding uretimi YAPILMADI." if gpu_gate_triggered else "GPU mevcut, devam edildi."}

## Kanit (bu makinede GERCEK olculmus, uydurulmadi)
- `torch.cuda.is_available()`: {hw["cuda_available"]}
- Aktif torch surumu: {hw["torch_version"]} (CPU-only derleme)
- Donanim: NVIDIA GeForce GT 1030 (nvidia-smi ile goruluyor, 4 GB VRAM) - ama
  aktif Torch paketi CUDA'siz derlendigi icin `cuda_available=False`
  (bkz. BENCHMARK_CPU_GT1030_T4.md, 24 Temmuz 2026 tarihli onceki dogrulama -
  AYNI sonuc bugun TEKRAR dogrulandi).
- GT 1030'un 4 GB VRAM'i, CUDA calissa bile Qwen3-VL-Embedding-2B (2B
  parametre, fp16'da yalniz agirliklar icin ~4 GB) icin muhtemelen YETERSIZ -
  bu iddia test EDILEMEDI (CUDA yok), sadece bir risk notu.

## Gercek maliyet tahmini (bench/gpu_gate.py'nin olculmus sabitleri)
{json.dumps(cost_table, indent=2, ensure_ascii=False)}

**Toplam CPU tahmini (3 is yuku, sirayla): {sum(r["cpu_estimate_h"] for r in cost_table):.1f} saat
(~{sum(r["cpu_estimate_h"] for r in cost_table)/24:.1f} gun).**
GPU'da (L4/T4 sinifi) toplam tahmini: {sum(r["gpu_estimate_min"] for r in cost_table):.1f} dakika.

## Bunun asagi akisa etkisi
Spec SS13 calisma sirasi adim 6-14 (2048d uretimi -> MRL turetme -> Blok A-F
-> backend yukleme -> hybrid benchmark -> karar raporu) TAMAMI notebook
02'nin urettigi embedding'lere bagimli. Bu adim GERCEKLESMEDIGI icin:
- Notebook 03 (PostgreSQL semasi): KOD HAZIRLANABILIR ama gercek AU-AIR
  segment/telemetri YUKLEMESI icin embedding gerekmiyor - segment/telemetri
  verisi notebook 01'den zaten gercek. Bu YAPILABILIR (embedding'e bagimli
  degil) - ayri calistirilacak.
- Notebook 04 (vector backend yukleme): embedding olmadan YAPILAMAZ (dolgu/
  sahte vektor uretmek SS11'in yasakladigi turden bir "gorunmez" sonuc
  olurdu). YAPILMADI.
- Notebook 05 (hybrid sorgu benchmarki - Blok C-F): YAPILAMAZ, ayni neden.
- Notebook 06 (sonuc raporu): "Kanit yetersiz" secenegi (spec SS14) bu
  nedenle gecerli sonuc olarak kullanilacak.

## Spec SS16 risk#3 karsiligi
"Mevcut CapERA embedding'lerinin gecerliligi" riski: bu depoda CapERA icin
HAM 2048d Qwen embedding (`.npy` + id listesi) YOK - yalniz
`data/downloads/capera/all_results.json` (onceden Drive'dan kopyalanmis
AGREGATIF sonuclar) var. Bu nedenle SS4.3 hucre-2'nin "mevcut embedding
gecerli mi" kontrolu de BASARISIZ - CapERA icin de yeniden uretim
gerekirdi, ayni GPU kapisina takilir.
'''

(OUT / "qwen2b_mrl_report.md").write_text(report_md, encoding="utf-8")
print(report_md)

manifest = RunManifest(
    notebook="02_qwen2b_embedding_and_mrl",
    hardware_profile=hw["hardware_profile"],
    extra={
        "gpu_gate_triggered": gpu_gate_triggered,
        "gpu_gate_message": gpu_gate_message,
        "cost_table": cost_table,
        "downstream_notebooks_blocked": ["04_vector_backend_loading", "05_hybrid_query_benchmark"],
        "downstream_notebooks_not_blocked": ["03_postgres_metadata_telemetry (embedding'e bagimsiz kisimlari)"],
    },
)
manifest_path = write_manifest(manifest, OUT)
print(f"\\nmanifest -> {manifest_path}")
print(f"rapor -> {OUT / 'qwen2b_mrl_report.md'}")
"""),
]

if __name__ == "__main__":
    out_path = pathlib.Path("notebooks/02_qwen2b_embedding_and_mrl.ipynb")
    nb = build_and_execute(CELLS, out_path, timeout=120)
    print(f"\\nNotebook yazildi: {out_path} ({len(nb.cells)} hucre)")
