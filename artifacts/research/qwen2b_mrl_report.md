# Qwen3-VL-Embedding-2B MRL - SONUC: GPU KAPISINDA DURDU

## Durum
GPU KAPISI TETIKLENDI - embedding uretimi YAPILMADI.

## Kanit (bu makinede GERCEK olculmus, uydurulmadi)
- `torch.cuda.is_available()`: False
- Aktif torch surumu: 2.13.0+cpu (CPU-only derleme)
- Donanim: NVIDIA GeForce GT 1030 (nvidia-smi ile goruluyor, 4 GB VRAM) - ama
  aktif Torch paketi CUDA'siz derlendigi icin `cuda_available=False`
  (bkz. BENCHMARK_CPU_GT1030_T4.md, 24 Temmuz 2026 tarihli onceki dogrulama -
  AYNI sonuc bugun TEKRAR dogrulandi).
- GT 1030'un 4 GB VRAM'i, CUDA calissa bile Qwen3-VL-Embedding-2B (2B
  parametre, fp16'da yalniz agirliklar icin ~4 GB) icin muhtemelen YETERSIZ -
  bu iddia test EDILEMEDI (CUDA yok), sadece bir risk notu.

## Gercek maliyet tahmini (bench/gpu_gate.py'nin olculmus sabitleri)
[
  {
    "workload": "AU-AIR (notebook 01 gercek pencere sayisi)",
    "n_items": 1866,
    "cpu_estimate_h": 452.4,
    "gpu_estimate_min": 76.0
  },
  {
    "workload": "CapERA (data/downloads/capera manifest - 2864 video, tek klip=tek item)",
    "n_items": 2864,
    "cpu_estimate_h": 694.4,
    "gpu_estimate_min": 116.6
  },
  {
    "workload": "MSR-VTT 1k-A (1000 test video)",
    "n_items": 1000,
    "cpu_estimate_h": 242.5,
    "gpu_estimate_min": 40.7
  }
]

**Toplam CPU tahmini (3 is yuku, sirayla): 1389.3 saat
(~57.9 gun).**
GPU'da (L4/T4 sinifi) toplam tahmini: 233.3 dakika.

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
