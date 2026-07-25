# Gorev listesi

Güncel kanıt ve ölçümler: `STATUS.md`.

## Faz 0 — ortam (insan + agent birlikte)
- [x] `.venv` içinde `pip install -r requirements.txt` (Windows/Python 3.14)
- [x] Docker Compose + gerçek ClickHouse schema — iki tablo `SHOW CREATE`
      ile ve insert/query smoke testiyle doğrulandı
- [x] Resmî Task 4 VisDrone-MOT trainsetini indir, SHA-256 ve
      56 sekans/56 annotation/24.201 kare sözleşmesini doğrula

## Faz 1 — ingest (bir model ile uctan uca)
- [x] 5 gerçek smoke sekansında `frames && windows` (691 kare / 7 pencere)
- [x] 5 gerçek smoke sekansında `detect` — 7 özellik satırı
- [x] 5 gerçek smoke sekansında `embed MODEL=xclip_hf_zeroshot` — 7×512d
- [x] 5 gerçek smoke sekansında `load MODEL=xclip_hf_zeroshot` — 7 satır
- [x] Gerçek embedding ile filtreli/filtresiz `otobüsü göster` sorgusu
- [x] Kaynak ölçümünden sonra seçili 5-sekans iki-model ingest
- [ ] Tam 56-sekans ingest (CPU maliyeti nedeniyle ölçümlü/onaylı başlat)
- [x] Elle bir sorgu dene:
      `python -c "from search.query import search; from search.merge import merge_intervals, pretty; print(pretty(merge_intervals(search('otobusu goster', 'xclip_hf_zeroshot'))))"`

## Faz 2 — ground truth + degerlendirme
- [x] 5-sekans manifest için `groundtruth` — 6 sorgu gerçek annotation ile eşleşti
- [ ] 5-10 sekansi FiftyOne'da gozle kontrol et — ozellikle `gt_walking`
      kamera-hareketi yanlis pozitifi var mi bak
- [x] İki-model smoke `eval` — 24 özet + 98 detay satırı (kalite iddiası değil)
- [ ] `make fiftyone` — pred/gt araliklarini yan yana incele

## Faz 3 — ikinci model + filtre A/B
- [x] 5-sekans `embed/load MODEL=siglip2_frameavg` — 7×1152d / 7 satır
- [x] `eval/run_eval.py` her iki modeli de kosturuyor mu dogrula
      (`MODELS` listesi)
- [ ] Sonuc tablosunu kategori kirilimiyla (tekli/hareket/bilesik) incele —
      filtre ACIK bilesik sorguda gercekten kazanc veriyor mu?

## Faz 4 — olcek testi (opsiyonel ama onerilir)
- [ ] Vektorleri sentetik cogaltip 1M/10M satirda filtreli/filtresiz
      sorgu gecikmesini ol (bkz. CONTEXT.md'deki ClickHouse HNSW supesi)

## Her fazin sonunda
- [x] `make test` hala geciyor mu (46/46 regresyon kontrolu)

---

## Benchmark/ClickHouse/YOLO optimizasyon planı (docs/codex/05_...)

Aşağıdaki fazlar yukarıdakinden ayrı, `docs/codex/05_CODEX_BENCHMARK_VE_OPTIMIZASYON_PLANI.md`
planına aittir. Numaralandırma o dosyayla birebir eşleşir (kendi Faz 0'ı var).

### Faz 0 — Regresyon tabanı + doküman düzeltmeleri
- [x] `pytest tests/ -v`: 50/50 geçti (46 taban + 4 yeni `offline_mode_enabled` testi)
- [x] `py_compile`: 42/42 (40 taban + `scripts/package_weights.py` + `tests/test_common.py`)
- [x] `docs/codex/02_FIKIRLER_VE_KARARLAR.md` SigLIP2 notu koda uyacak şekilde düzeltildi (`AutoModel`, `Siglip2Model` değil)
- [x] `common.offline_mode_enabled()` + `config.yaml: offline_mode` + her iki adaptörde `local_files_only`
- [x] `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` gerçek smoke: X-CLIP ve SigLIP2 lokal cache'ten yüklendi, ağ çağrısı yok
- [x] `scripts/package_weights.py` gerçek çalıştırıldı: `weights/weights_manifest.json` (X-CLIP 783.7 MB, SigLIP2 4578.6 MB, yolo26x.pt 118.7 MB)

### Faz 1 — Benchmark altyapısı
- [x] Çalıştırma yüzeyi açıldı: `Makefile: bench:` hedefi ve `poc.ps1 ValidateSet`'e `'bench'`
- [x] `bench/` paketi: `spec.py` (RunSpec+run_id), `timing.py` (StageTimer, p50/p95),
      `metrics.py` (eval/metrics.py sarmalayıcı: MRR + K=1,5,10), `manifest.py`
      (git hash, OS/Python/Torch/ClickHouse sürümü, GPU, model meta, config snapshot),
      `report.py` (tek HTML+JSON), `runner.py` (RunSpec→sonuç, `--check-determinism`)
- [x] GT seti 6 → 28 sorguya çıkarıldı, TR+EN çiftli (tekli×5, hareket×1, sayısal×3,
      bileşik×3, negatif-kontrol×2 — `eval/make_groundtruth.py::build_queries` +
      `build_query_metadata`); `search/parser.py` İngilizce eşanlamlı + sayısal eşik
      (`en az N` / `at least N`) desteğiyle genişletildi (dil karşılaştırması filtre
      varlığıyla değil yalnızca semantik kaliteyle ayrışsın diye)
- [x] Bench subset: 56 sekanstan seçilmiş 19 sekans (`config.yaml: bench.subset`,
      kanıt: `scripts/select_bench_subset.py`) — trafik yoğunluğu, bus/truck
      varlığı, kare-parlaklığı (61.7–189.3) çeşitliliği kapsar
- [x] 19 sekans gerçek ingest edildi: frames (19 mp4) → windows (73 pencere) →
      detect (73 YOLO satırı) → embed+load iki model (73×512d X-CLIP, 73×1152d
      SigLIP2, ClickHouse tabloları truncate edilip taze yüklendi)
- [x] `python -m bench.runner`: 2 model × 2 filtre = 4 run, gerçek
      `artifacts/benchmark_report.{html,json}` üretti (kategori kırılımlı
      recall@10/precision@10/MRR + p50/p95 sorgu süresi)
- [x] Determinizm kontrolü gerçek çalıştırıldı (`--check-determinism`, 19 sekans,
      28 sorgu, iki koşum arasında fark yok): **GEÇTİ**
- [x] 20 yeni saf-mantık testi eklendi (`test_bench_spec/timing/metrics/report.py`,
      genişletilmiş `test_parser.py`, `test_common.py`) — toplam 72/72 geçiyor
