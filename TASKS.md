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

### Faz 2 — ClickHouse arama katmanı: davranış doğrulaması + ölçek
- [x] `system.settings` gerçekten sorgulandı (26.7.1): tüm 7 ayar bu sürümde var.
      **R2 düzeltmesi:** `max_limit_for_vector_search_queries` varsayılanı
      planın varsaydığı 100 değil **1000** — `search/query.py`'nin `top_k=200`
      varsayılanı zaten güvenli sınırın altında, bu risk bu sürümde canlı değil.
      `vector_search_filter_strategy` varsayılanı `'auto'` = ClickHouse'un
      kendi dokümantasyonuna göre **postfiltering** (R1'i doğrular).
- [x] 8 yeni SQL kataloğu dosyası (`sql/search_lab/08-15`): 4 strateji ×
      2 seçicilik (loose: `person_count>=1`, strict: `bus_count>=1 AND
      person_count>=3`); `search/sql_catalog.py` `strategy`/`selectivity`
      alanlarıyla genişletildi, ikinci tablo `sql_for_table()` ile üretiliyor.
- [x] `search/query.py::search()`'e `strategy` parametresi eklendi (imza
      değişmedi, varsayılan `'auto'` önceki davranışla birebir aynı) —
      `'exact' | 'prefilter' | 'postfilter_rescore'`, SETTINGS değerleri
      sql/search_lab/06-07 ile senkron.
- [x] **R1 GERÇEK VERİYLE KANITLANDI (73 satır):** `hnsw strict` LIMIT 10
      yerine 4 satır döndürdü; `prefilter strict` ve `bruteforce strict` tam
      10 döndürdü. EXPLAIN indexes=1: `vector_index_in_plan=True` yalnızca
      hnsw/postfilter_rescore'da, prefilter/bruteforce'ta False (beklendiği
      gibi index'i atlıyorlar).
- [x] **100K satırlık sentetik ölçek testi çalıştırıldı** (`bench_scale_512`,
      ayrı tablo, `scripts/build_scale_table.py` — insert 35.6sn, `OPTIMIZE
      TABLE FINAL` ile index inşası 38.7sn, index boyutu 113.31 MiB):
      `hnsw strict` **0 satır döndürdü** (tam sessiz kayıp), `postfilter_rescore
      strict` (fetch_multiplier=5) de **0 satır**; `prefilter`/`bruteforce`
      strict her ikisi de doğru 10 satır döndürdü. HNSW loose 22.5ms vs
      bruteforce loose 156ms (~7× hızlı) — ama strict'te güvenilmez.
- [x] `fetch_multiplier` sweep 100K ölçekte: {1,5,20,50,100} → satır sayısı
      0,0,8,10,10. **Sonuç: varsayılan (1) ve planın önerdiği 5 yetersiz;
      tam LIMIT için ~50 gerekli, o noktada gecikme (132ms) zaten prefilter'a
      (104ms) yakınsıyor** — seçici filtrede prefilter hem daha doğru hem
      rakip gecikmede. **Öneri: seçici filtre içeren sorgularda varsayılan
      `strategy='prefilter'` kullanılmalı**, `'auto'` yalnızca gevşek/filtresiz
      sorgularda güvenli.
- [x] `ef_search` (`hnsw_candidate_list_size_for_search`) sweep {64,256,512}:
      100K ölçekte satır sayısını değiştirmedi (4→4→4 küçük ölçekte) — bu
      ayar aday-listesi kalitesini etkiler, LIMIT-altı dönme sorununu
      `fetch_multiplier` kadar çözmüyor.
- [x] HNSW recall@10: küçük ölçekte (73 satır) 1.00 — anlamsız (plan
      kuralı); 100K ölçekte de 1.00 (bu veri dağılımında HNSW top-10 exact
      top-10 ile birebir örtüştü, seçici filtre uygulanmadığı loose sorguda).
- [x] **Bellek projeksiyonu (gerçek 100K/512d ölçümünden ekstrapole):**
      1M×512d ≈ 1.19 GB, 10M×512d ≈ 11.88 GB, 1M×1152d ≈ 2.67 GB (planın
      bağımsız tahmini 2.6 GB ile örtüşüyor), 10M×1152d ≈ 26.7 GB.
      `vector_similarity_index_cache_size` varsayılanı 5 GB — 1M ölçekte iki
      model toplamı (~3.9 GB) sığar ama büyüme payı yok; 10M'de (~38.6 GB)
      açıkça yetersiz.
- [ ] 1M/10M sentetik ölçek — süre bütçesi nedeniyle yapılmadı (100K'nın
      kendisi kararı değiştirecek kadar net sinyal verdi).
- [ ] `CODEC(NONE)`, binary vector bind, `materialize_skip_indexes_on_insert`
      insert-throughput denemesi — süre bütçesi nedeniyle yapılmadı.
- [x] 11 yeni saf-mantık testi (`test_sql_catalog.py` genişletildi +2,
      `test_query_strategy.py` +6, `test_strategy_matrix_html.py` +3) —
      toplam 83/83 geçiyor.

**Çıkış kapısı durumu:** "Prefilter gerçekten prefilter mı?" — EVET, EXPLAIN
+ rows-returned kanıtıyla doğrulandı. Sıkı filtrede LIMIT-altı dönme —
belgelendi, 100K'da tam sıfıra kadar gidiyor. 100K stratejı karşılaştırması
tamam; 1M yapılmadı (raporda açıkça işaretli). R2 düzeltmesi: risk bu
sürümde zaten canlı değildi, kanıtlandı.

### Faz 3 — YOLO optimizasyonu
- [x] **Plan düzeltmesi:** planın önerdiği `dronefreak/visdrone-detection-model-zoo`
      HF reposu gerçekte YOK (401/bulunamadı doğrulandı, indirmeden önce
      kontrol edildi). Gerçek, doğrulanmış alternatif: `mshamrai/yolov8{n,s,m}
      -visdrone` (gerçek mAP@0.5: 0.341/0.408/0.454, gerçek VisDrone sınıf
      haritası indirilip `model.names`'ten okunarak doğrulandı: {0:pedestrian,
      1:people, 2:bicycle, 3:car, 4:van, 5:truck, 6:tricycle,
      7:awning-tricycle, 8:bus, 9:motor}).
- [x] `ingest/04_detect.py`'deki sabit `COCO_MAP` config'e taşındı
      (`config.yaml: detector.variants.<ad>.{checkpoint,class_map}` +
      `detector.n_sample`); `--variant` CLI argümanı eklendi, varsayılan
      davranış (variant verilmezse) eski COCO_MAP ile birebir aynı.
      `window_features()` artik checkpoint+class_map parametreli.
- [x] `yolov8n-visdrone` (6.2 MB) ve `yolov8s-visdrone` (22.5 MB) gerçekten
      indirildi (`huggingface_hub`), `weights/weights_manifest.json`'a
      SHA-256 ile eklendi.
- [x] Dedektör bake-off matrisi 73 pencerede gerçek çalıştırıldı
      (`bench/detector_baseline.py`, `scripts/run_detector_bakeoff.py`):

  | varyant | fps (CPU, tek kare) | person P/R | car P/R | bus P/R | truck P/R |
  |---|---:|---|---|---|---|
  | yolo26x_coco (COCO, mevcut baseline) | 0.82 | 1.00/0.98 | 1.00/1.00 | 0.83/0.75 | 0.73/0.88 |
  | yolov8n_visdrone | **1.61** | 1.00/0.95 | 1.00/0.97 | 0.81/0.85 | 0.85/0.50 |
  | yolov8s_visdrone | 1.53 | 1.00/0.98 | 1.00/0.97 | 0.73/0.80 | 0.75/0.71 |

  Count MAE person/car mutlak olarak yüksek (8.7–14) çünkü VisDrone
  sahneleri yoğun (pencere başına onlarca küçük nesne); asıl kullanılan
  sinyal eşik P/R'dır (filtreler `count>=1` kullanıyor). **Karışık ama net
  bulgu:** VisDrone-tuned modeller person/bus'ta COCO modeline eşdeğer/
  hafif iyi, ama truck recall'da COCO x-large belirgin önde (0.88 vs
  0.50/0.71) — büyük/genel model bazı sınıflarda küçük/özel modelden daha
  iyi genelliyor; "küçük fine-tune'lu her zaman kazanır" varsayımı burada
  doğrulanmadı.
- [x] **Downstream etki gerçek ölçüldü** (aynı embedding modeli — X-CLIP —
      sabit, yalnız dedektör değişti, filtre AÇIK, 28 sorgu):

  | varyant | tekli P@10 | sayısal P@10 | bileşik P@10 |
  |---|---:|---:|---:|
  | yolo26x_coco | 0.810 | 0.838 | 0.757 |
  | yolov8n_visdrone | **0.856** | **0.867** | **0.849** |
  | yolov8s_visdrone | 0.818 | 0.883 | 0.780 |

  Recall@10 üç varyantta da ~aynı (0.50 tekli, 0.40–0.46 sayısal, 0.52
  bileşik) — truck recall'daki izole zayıflık downstream retrieval'a önemli
  bir zarar olarak yansımadı (truck-özel sorgu payı düşük); tersine
  yolov8n_visdrone çoğu kategoride precision'ı gerçek biçimde artırdı.
- [x] **Karar:** `yolov8n_visdrone` varsayılan dedektör (config.yaml:
      `detector.default_variant`) — en hızlı (COCO'dan ~2× hızlı) VE
      downstream'de en az eşdeğer/genelde daha iyi. ClickHouse'daki her iki
      model tablosu bu varyantın ürettiği filtre kolonlarıyla yeniden
      yüklendi (73 satır × 2 model); `artifacts/benchmark_report.{html,json}`
      bu durumu yansıtacak şekilde yeniden üretildi.
- [ ] Batch inference (batch=8/16), tek-decode paylaşımı, `imgsz` ablation'ı
      (640 vs 960/1280), FP16, `n_sample` {3,6,12} ablation'ı — süre bütçesi
      nedeniyle yapılmadı.
- [x] 4 yeni saf-mantık testi (`test_detector_baseline.py`) — toplam 87/87
      geçiyor.

**Çıkış kapısı durumu:** ≥2 VisDrone-tuned YOLO gerçek veride ölçüldü (3
varyant); varsayılan dedektör kararı sayıyla (hız + downstream P/R@10)
gerekçelendirildi. Inference-optimizasyon ablation'ları (batch/imgsz/FP16/
n_sample) yapılmadı — TASKS.md'de açıkça işaretli.

### Faz 4 — Embedding bake-off
- [x] **Plan düzeltmesi:** planın iki adayı da "kolay HF indirme" varsayımını
      doğrulamadı. `alibaba-pai/VideoCLIP-XL` gerçek ama lisansı
      **CC-BY-NC-SA-4.0 (ticari olmayan)** — kurumsal üretim hedefiyle
      uyumsuz, ayrıca özel `modeling.py`/`utils/` kodu gerektiriyor
      (`transformers`-native değil). `LanguageBind/LanguageBind_Video`
      gerçek ve MIT ama `model_type=LanguageBindVideo` yüklü `transformers`
      5.14.1 tarafından tanınmıyor (gerçek `AutoModel.from_pretrained()`
      hatası ile doğrulandı) — resmi olmayan bir PyPI paketi veya orijinal
      GitHub kodu gerekiyor. İkisi de bu oturumda entegre edilmedi.
- [x] Gerçek yeni aday: `Qwen/Qwen3-VL-Embedding-2B` (Apache-2.0, gerçek
      doğrulandı, `transformers` 5.14.1 mimariyi native destekliyor,
      `sentence-transformers` üzerinden standart `model.encode()`).
      `models/qwen3vl_emb.py`: frame-average (n_sample=6), agir bagimlilik
      (`sentence-transformers`, `qwen-vl-utils`) lazy-import.
- [x] MRL (Matryoshka) boyutları ayrı registry/tablo:
      `qwen3vl_emb_{2048,1024,512,256}`. 1024/512/256, tek gerçek 2048d
      video-embed koşumundan `scripts/mrl_truncate_embeddings.py` ile
      (kırp + L2-yeniden-normalize) türetildi — modeli 4 kez koşturmaya
      gerek kalmadı.
- [x] **Kritik CPU-maliyet bulgusu:** 73 pencerenin tamamı için gerçek
      `embed_video` koşumu **1062 dakika (~17.7 saat)** sürdü — pencere
      başına ~14.5 dakika. Karşılaştırma: X-CLIP ~32sn/pencere, SigLIP2
      ~62sn/pencere (STATUS.md). İlk smoke testi (sentetik gürültü
      görüntüleriyle, gerçek video karesi değil) bunu ~52sn/pencere olarak
      hatalı tahmin etmişti — gerçek kare boyutu/karmaşıklığı maliyeti
      ~17× hafife aldırmış. **Qwen3-VL-Embedding-2B bu CPU'da ingest için
      pratik değil; GPU zorunlu.** Bu CPU-'ye özgü bir sonuçtur, modelin
      kendisinin "kötü" olduğu anlamına gelmez — gerçek GPU sayıları
      henüz yok (bkz. aşağıdaki "GPU ölçümü" maddesi).
  - `embed_text` (sorgu) çok daha ucuz: model ısındıktan sonra ~1-6sn/sorgu
    — arama zamanında (yalnızca metin embed edilir) kullanılabilir kalır.
- [x] **MRL boyut taraması (gerçek, 19 sekans, 28 sorgu, filtre AÇIK):**

  | boyut | tekli R@10/P@10 | hareket | sayısal | bileşik | HNSW index (73 satır) | 1M satırda ekstrapole |
  |---|---|---|---|---|---:|---:|
  | 2048 | 0.473/0.816 | 0.474/0.900 | 0.458/0.883 | 0.522/0.849 | 295.5 KiB | ~4.15 GB |
  | 1024 | 0.460/0.796 | 0.474/0.900 | 0.458/0.883 | 0.522/0.849 | 148.9 KiB | ~2.09 GB |
  | 512  | 0.466/0.806 | 0.474/0.900 | 0.466/0.900 | 0.522/0.849 | 75.6 KiB  | ~1.06 GB |
  | 256  | 0.473/0.816 | 0.474/0.900 | 0.466/0.900 | 0.522/0.849 | 39.7 KiB  | ~0.56 GB |

  **Sonuç: 2048d'den 256d'ye (8× küçültme) kalite kaybı ölçülemez düzeyde
  (tüm kategorilerde <0.02 recall farkı), depolama ~7.4× azalıyor.** MRL
  truncation burada teorik değil, doğrudan uygulanabilir bir öneri:
  üretim adayı olarak 256d veya 512d, 2048d yalnızca üst-sınır referansı.
- [x] **Model karşılaştırması (aynı 19 sekans/28 sorgu, filtre AÇIK,
      `yolov8n_visdrone` dedektörüyle):** Qwen-2048 (tekli 0.473/hareket
      0.474/bileşik 0.522) X-CLIP'e (tekli 0.500/hareket 0.500/bileşik
      0.522) karşı **eşdeğer, belirgin biçimde üstün değil** — MMEB-V2'de
      iddia edilen sınıf-lideri konumuna rağmen. Bu, ayrı bir konuşmada
      öne sürülen hipotezi doğruluyor: darboğaz muhtemelen model kalitesi
      değil, pencereleme/GT/görev tanımı (hareket kategorisinde bile fark
      yok — gerçek zamansal model X-CLIP ile kare-ortalama Qwen/SigLIP2
      ayrışmıyor).
- [x] Offline doğrulama: `HF_HUB_OFFLINE=1` ile gerçek çalıştırma geçti
      (model zaten cache'te, yükleme 12.3sn, embed_text 6.1sn, ağ çağrısı
      yok). `weights_manifest.json`'a eklendi (4271.1 MB).
- [ ] **GPU ölçümü henüz yok.** Bu oturumda GT1030 CUDA kurulumu
      Windows `MAX_PATH` sınırına takıldı (Faz 2 notu); Colab'i canlı
      sürmek için tarayıcı/API erişimim yok. `scripts/colab_gpu_bench.py`
      hazırlandı (kullanıcı Colab'de kendi çalıştıracak) ama bu script
      **bu oturumda gerçek GPU'da test edilmedi** — kodun kendisi zaten
      doğrulanmış fonksiyonları (`bench/timing.py`, `models.get_embedder`)
      tekrar kullanıyor ama uçtan uca koşum kanıtlanmadı.
- [x] 3 yeni saf-mantık testi (`test_mrl_truncate.py`) — toplam 90/90
      geçiyor (1 gerçek regresyon da düzeltildi: `test_load_clickhouse.py`
      yeni 4 tabloyu hesaba katmıyordu).

**Çıkış kapısı durumu:** ≥1 yeni video-native model gerçek veride ölçüldü
ve aynı harness'ta karşılaştırıldı — EVET (Qwen3-VL-Embedding-2B, 4 MRL
boyutu). Kalite+hız+depolama tek tabloda — EVET (hız CPU-only, GPU eksik).
TR/EN satırı — Faz 1'in GT setinden geliyor, ayrı ölçülmedi bu fazda.
