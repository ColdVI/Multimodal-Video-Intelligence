# Uygulama durumu — 24 Temmuz 2026

## Tamamlanan ve gerçek çalıştırmayla doğrulananlar

- Windows/Python 3.14 için `.venv` kuruldu; `requirements.txt` eksiksiz yüklendi.
- 46 pytest testi geçti; Python dosyaları `py_compile` kontrolünden geçti.
- Üçüncü taraf cache/config yazımları repo içindeki `.runtime/` dizinine alındı.
- Sistem ffmpeg'i yokken `imageio-ffmpeg` fallback'i gerçek binary ile çalıştı.
- Docker Desktop başlatıldı; ClickHouse `26.7.1.1315` ve MinIO çalışıyor.
- `schema.sql` gerçek ClickHouse'a uygulandı; 512d ve 1152d tablolar doğrulandı.
- İki tabloda insert + filtre + `cosineDistance` smoke testi geçti ve satırlar temizlendi.
- X-CLIP checkpoint'i indirildi; CPU'da gerçek 512d video/metin embedding üretildi.
- Transformers 5.x pooled-output değişikliği düzeltildi ve regresyon testi eklendi.
- Gerçek X-CLIP vektörüyle parser → filtreli/filtresiz ClickHouse sorgusu →
  interval merge hattı uçtan uca geçti.
- `yolo26x.pt` indirildi; CPU'da tek görüntü inference ve sentetik MP4 üzerinde
  `window_features()` filtresi/optical-flow hattı geçti.
- Windows için `scripts/poc.ps1` runner eklendi; `test` ve `schema` görevleri geçti.
- Resmî Task 4 VisDrone-MOT trainset Google Drive'dan indirildi: 8.080.572.990
  bayt, SHA-256 `566d08fb53fff4e539f386f5a408ccf17854fd53814dc756bdede2de1dbb4014`.
- Veri sözleşmesi doğrulandı: 56 sekans, 56 annotation, 24.201 JPEG ve sıfır
  sekans/annotation isim uyuşmazlığı. `scripts/poc.ps1 download-data` aynı
  sözleşmeyi yeniden üretilebilir biçimde doğruluyor.
- 5 gerçek sekans/691 kare üzerinde ingest smoke'u geçti: 5 MP4 -> 7 pencere ->
  7 YOLO özellik satırı -> 7×512d X-CLIP ve 7×1152d SigLIP2 embedding ->
  model başına ayrı ClickHouse tablosunda 7 satır.
- Gerçek annotation'dan altı sorgu için GT ve iki model × iki filtre eval
  üretildi (`results.json`: 24, `results_detail.json`: 98 satır).
- Gerçek çalıştırmada ayrıca annotation yolu, ClickHouse HTTP multi-statement
  load ve Windows Python UTF-8 hataları bulunup düzeltildi.
- Exact filtre, exact brute-force vektör, HNSW ve iki hibrit strateji için yedi
  salt-okunur SQL tek katalogda toplandı; aynı katalog test ve rapor tarafından
  kullanılıyor. Gerçek ClickHouse raporu HTML ve JSON olarak üretildi.
- Mevcut 14 satırlık smoke verisinde exact brute-force ve HNSW aynı sıralamayı
  verdi; yaklaşık mesafe değerleri birebir aynı değildi (maksimum fark
  `0.0001023`). Exact ve iki hibrit sorgunun her biri dört filtre eşleşmesi döndürdü.
  `EXPLAIN indexes = 1`, HNSW indeksini similarity ve postfilter/rescore planında
  gösterdi; brute-force exact ve prefilter planında göstermedi.

## Ölçülen ortam

- GPU: NVIDIA GeForce GT 1030, 4 GB.
- Kurulu Torch: `2.13.0+cpu`; CUDA kullanılamıyor.
- X-CLIP CPU smoke: model yükleme 0.87 sn, 32 kare embedding 7.92 sn,
  metin embedding 0.32 sn.
- YOLO26x CPU smoke: ilk indirme+yükleme 207.8 sn; tek 640×640 inference 1.54 sn.
- Gerçek iki-pencere YOLO: 17.7 sn; gerçek iki-pencere X-CLIP: 51.2 sn.
- 5-sekans ölçümü: frames 23.4 sn, YOLO 59.9 sn, X-CLIP 158.2 sn,
  SigLIP2 311.6 sn, iki-model eval 42.4 sn.
- `.venv` boyutu: yaklaşık 1.83 GB (model cache'leri hariç).
- Smoke-5 sıcak query benchmark'ında T4, X-CLIP'i yerel CPU'dan `2,51×`,
  SigLIP2'yi `11,11×` hızlı çalıştırdı. GT 1030 sistemde görünmesine rağmen
  aktif Torch CPU-only olduğu için CUDA benchmark'ı çalıştırılamadı. Ayrıntı:
  `BENCHMARK_CPU_GT1030_T4.md`.

## Faz 1 benchmark altyapısı ([docs/archive/phases/faz11-development/05_CODEX_BENCHMARK_VE_OPTIMIZASYON_PLANI.md](../archive/phases/faz11-development/05_CODEX_BENCHMARK_VE_OPTIMIZASYON_PLANI.md), 25 Temmuz 2026)

- `bench/` paketi gerçek çalıştırıldı: 19 sekans (56'dan seçilmiş temsili bench
  subset, `config.yaml: bench.subset`) → 73 pencere → 2 model × 2 filtre = 4
  run, `artifacts/benchmark_report.{html,json}`.
- GT 28 sorguya çıkarıldı (TR+EN çiftli, tekli/hareket/sayısal/bileşik/
  negatif-kontrol). Filtre AÇIK tutarlı biçimde precision ve MRR'ı artırıyor
  (ör. X-CLIP bileşik: MRR 0.92 filtreli vs 0.51 filtresiz); recall filtresiz
  biraz daha yüksek kalabiliyor (aday havuzu daralmıyor) — beklenen ödünleşim.
  Negatif-kontrol sorguları (n_gt=0) recall/precision=0 veriyor; bu P/R@K
  metriğinin bir sınırıdır (boş GT'ye karşı hiçbir tahmin "doğru" sayılamaz),
  başarısızlık değildir.
- Determinizm kontrolü GEÇTİ: aynı RunSpec iki kez koşuldu, 28 sorgunun tümünde
  metrikler birebir aynı çıktı.
- SigLIP2 sorgu gecikmesi X-CLIP'in ~3 katı (mean 0.92sn vs 0.31sn, filtreli),
  512d/1152d boyut farkına ve CPU-only ortama tutarlı.
- GT 1030 CUDA denemesi bloke oldu (Windows `MAX_PATH`/`LongPathsEnabled=0`);
  detay
  [docs/archive/phases/faz11-development/02_FIKIRLER_VE_KARARLAR.md](../archive/phases/faz11-development/02_FIKIRLER_VE_KARARLAR.md).
  CPU-only ortam korunuyor,
  54→72 test hâlâ geçiyor.

## Faz 2 ClickHouse strateji doğrulaması ([docs/archive/phases/faz11-development/05_CODEX_BENCHMARK_VE_OPTIMIZASYON_PLANI.md](../archive/phases/faz11-development/05_CODEX_BENCHMARK_VE_OPTIMIZASYON_PLANI.md), 25 Temmuz 2026)

- **En önemli bulgu:** varsayılan davranış (`vector_search_filter_strategy=
  'auto'`, ClickHouse'un kendi dokümantasyonuna göre postfiltering) seçici
  filtrede (`bus_count>=1 AND person_count>=3`) 100K satırlık sentetik ölçekte
  **0 satır döndürdü** — LIMIT 10 istenmesine rağmen. Aynı filtre `prefilter`
  ve `bruteforce` stratejileriyle doğru 10 satır döndürdü. Bu, planın R1
  riskinin teorik değil gerçek ve ciddi olduğunu kanıtlıyor.
- `vector_search_index_fetch_multiplier`'ı varsayılan 1'den 50'ye çıkarmak
  0-satır sorununu düzeltiyor ama o noktada gecikme zaten prefilter'a
  yakınsıyor — **öneri: seçici filtreli sorgularda `strategy='prefilter'`
  varsayılan olmalı**, `search/query.py::search()`'e eklenen yeni `strategy`
  parametresiyle artık seçilebilir (varsayılan `'auto'`, önceki davranışla
  birebir aynı, hiçbir çağıran kod bozulmadı).
- R2 (`max_limit_for_vector_search_queries`) düzeltmesi: bu ClickHouse
  sürümünde (26.7.1) gerçek varsayılan 1000, planın varsaydığı 100 değil —
  `system.settings`'ten doğrulandı. `top_k=200` zaten güvenli.
- 100K satırlık `bench_scale_512` tablosu (üretim `clips_*` tablolarına
  dokunmadan) gerçekten oluşturuldu: insert 35.6sn, index inşası
  (`OPTIMIZE TABLE FINAL`) 38.7sn, index boyutu 113.31 MiB. Buradan
  ekstrapole: 1M×512d≈1.19GB, 1M×1152d≈2.67GB (planın 2.6GB tahminiyle
  örtüşüyor), 10M×1152d≈26.7GB — varsayılan 5GB cache 10M ölçekte açıkça
  yetersiz. Temizleme: `DROP TABLE bench_scale_512` (otomatik silinmedi).
- 1M/10M ölçek testi ve CODEC(NONE)/binary-bind/materialize-throughput
  denemeleri süre bütçesi nedeniyle yapılmadı (TASKS.md'de açıkça işaretli).

## Faz 3 YOLO dedektör bake-off ([docs/archive/phases/faz11-development/05_CODEX_BENCHMARK_VE_OPTIMIZASYON_PLANI.md](../archive/phases/faz11-development/05_CODEX_BENCHMARK_VE_OPTIMIZASYON_PLANI.md), 25 Temmuz 2026)

- Planın önerdiği `dronefreak/visdrone-detection-model-zoo` HF reposu
  yok (indirmeden önce doğrulandı: 401). Gerçek alternatif kullanıldı:
  `mshamrai/yolov8{n,s}-visdrone` (gerçek mAP@0.5 0.341/0.408).
- `ingest/04_detect.py` config-driven `detector.variants` registry'sine
  taşındı (checkpoint + class_map per varyant); eski COCO davranışı
  varyant belirtilmezse birebir korunuyor.
- 73 pencerede gerçek bake-off: `yolov8n_visdrone` COCO baseline'a göre
  ~2× hızlı (1.61 vs 0.82 fps, CPU) VE downstream Recall/Precision@10'da
  eşdeğer/genelde daha iyi (precision tekli 0.86 vs 0.81, sayısal 0.87 vs
  0.84, bileşik 0.85 vs 0.76) — **yeni varsayılan dedektör**. İlginç ama
  dürüst bulgu: truck sınıfında COCO x-large modeli (recall 0.88) her iki
  VisDrone-tuned varyanttan (0.50/0.71) daha iyi — "küçük fine-tune'lu
  model büyüğü her zaman geçer" varsayımı burada doğrulanmadı, ama truck
  payı düşük olduğu için downstream'e yansımadı.
- Her iki ClickHouse tablosu (X-CLIP + SigLIP2) yeni varsayılan dedektörün
  ürettiği filtre kolonlarıyla yeniden yüklendi.
- Batch/imgsz/FP16/n_sample ablation'ları süre bütçesi nedeniyle yapılmadı.

## Faz 4 embedding bake-off ([docs/archive/phases/faz11-development/05_CODEX_BENCHMARK_VE_OPTIMIZASYON_PLANI.md](../archive/phases/faz11-development/05_CODEX_BENCHMARK_VE_OPTIMIZASYON_PLANI.md), 26-27 Temmuz 2026)

- Planın iki adayı da (VideoCLIP-XL, LanguageBind_Video) doğrulandı ama
  ikisi de "kolay entegrasyon" değildi: VideoCLIP-XL ticari olmayan lisans
  + özel kod; LanguageBind_Video (MIT) `transformers` 5.14.1'de tanınmıyor
  (gerçek hata ile doğrulandı). Gerçek yeni aday: `Qwen/Qwen3-VL-Embedding-2B`
  (Apache-2.0, `sentence-transformers` üzerinden çalışıyor).
- **En büyük bulgu:** Qwen'in 73 pencerelik gerçek `embed_video` koşumu
  **17.7 saat** sürdü (pencere başına ~14.5 dakika — X-CLIP'in ~27 katı).
  İlk smoke testi (sentetik görüntüyle) bunu ~17× hafife aldırmıştı.
  **Bu CPU'ya özgü bir sonuç** — GPU'da (Colab/T4 veya başka bir GPU)
  çok daha hızlı olması beklenir ama bu oturumda GPU ölçümü yapılamadı
  (GT1030 CUDA kurulumu bloke, Colab'i canlı süremiyorum).
  `scripts/colab_gpu_bench.py` kullanıcının Colab'de çalıştırması için
  hazırlandı ama gerçek GPU'da test edilmedi.
- MRL (Matryoshka) boyut taraması gerçek ölçüldü: 2048d'den 256d'ye (8×
  küçültme) kalite kaybı <0.02 recall, depolama ~7.4× azalıyor — üretim
  için 256d/512d öneriliyor, 2048d yalnızca referans.
- Qwen-2048, X-CLIP'e karşı bu benchmarkta eşdeğer (hareket kategorisinde
  bile fark yok) — MMEB-V2 liderliğine rağmen. Muhtemel yorum: darboğaz
  model kalitesi değil, pencereleme/GT/görev tasarımı.
- Offline mod doğrulandı (`HF_HUB_OFFLINE=1`, gerçek çalıştırma, ağ
  çağrısı yok); `weights_manifest.json`'a eklendi (4271.1 MB).

## Faz 5 profiller ve nihai rapor ([docs/archive/phases/faz11-development/05_CODEX_BENCHMARK_VE_OPTIMIZASYON_PLANI.md](../archive/phases/faz11-development/05_CODEX_BENCHMARK_VE_OPTIMIZASYON_PLANI.md), 27 Temmuz 2026)

- `config.yaml: profiles` (fast/balanced/accurate) eklendi — yalnızca
  gerçek ölçülen eksenler (dedektör, embedding modeli, arama stratejisi)
  değişir; window/stride/n_sample ablate edilmediği için tüm profillerde
  sabit kaldı.
- Nihai rapor:
  [docs/reports/faz11/FINAL_REPORT.md](../reports/faz11/FINAL_REPORT.md) — 5 ana bulgu, Pareto tablosu,
  ClickHouse/dedektör/model kararları, offline paket içeriği, üretime açık
  8 madde.
- **Dürüst özet:** plan "tamamlandı" değil "kısmen doğrulandı" durumda.
  Eksik kalanlar: GPU ölçümü (Colab bundle hazır, koşulmadı), 1M+ ölçek,
  `gt_walking` görsel denetimi, profil-özel ayrı gerçek koşular.

## Unified Search Harness: dataset registry + adaptive MRL pilotu (28 Temmuz 2026)

Dış bir plan (Codex "Unified Search Strategy + Qwen MRL + PostgreSQL Hybrid
Harness" + üstüne eklenen düzeltmeler) temel alındı. Bu bölümün Faz
numaraları YUKARIDAKİ Faz 1-5 ile AYNI şema DEĞİL - ayrı bir iş paketi.

- [x] **Düzeltildi, önemli**: kaynak dokümanın "VisDrone 28 sorgu" varsayımı
      DOĞRU ama gerekçesi yanlış anlaşılmıştı - `data/groundtruth/gt.json`
      (6 kayıt) ESKİ bir 5-sekans smoke çıktısı, hiç kullanılmıyor.
      `eval/make_groundtruth.py::build_queries()` CANLI çağrıldığında
      (doğrulandı: `len(build_queries())` -> 28) gerçek sayı budur.
      `n_sample=6` (`models/qwen3vl_emb.py::embed_video`) AYRI, ilgisiz bir
      parametre - ikisi karıştırılmamalı.
- [x] **PostgreSQL Faz 4-5 (dış plan) araştırıldı, gerekmediği kanıtlandı**:
      model->tablo eşlemesi zaten `f"clips_{model_name}"` isimlendirme
      kuralı, segment kimliği `(video_id, t_start)` demetiyle çalışıyor,
      segment_features zaten ClickHouse satırında. Postgres'e HİÇ
      dokunulmadı. Detay: `CONTEXT.md` "PostgreSQL yok" bölümü.
- [x] `bench/gpu_gate.py`: qwen3vl_emb ailesi için GPU zorunluluğu -
      gerçek ölçülmüş maliyetlerden (872.9sn/pencere CPU, 2.443sn/pencere
      L4) tahmin üretir. `ingest/03_embed.py` ve `scripts/validate_msrvtt.py`
      bağlandı; xclip/siglip2 KAPSAM DIŞI (zaten CPU'da pratik).
- [x] **Config-driven dataset registry** (`datasets/registry.py`,
      `config.yaml: datasets:`): `visdrone` (clickhouse backend, 28 sorgu,
      structured filtreli) ve `msrvtt_1ka` (artifact_matrix backend, 1000
      sorgu, filtresiz) kayıtlı. `datasets/visdrone.py`, `datasets/msrvtt.py`
      somut adaptörler - GERÇEK yerel veriye karşı doğrulandı (19 sekans,
      1000 MSR-VTT videosu). `run_strategy_or_unsupported()`: MSR-VTT'de
      prefilter/postfilter/telemetry_filter istenirse SESSİZCE yoksaymaz,
      açık `unsupported_strategy` sonucu üretir.
- [x] **Adaptive MRL harness GERÇEKTEN ÇALIŞTIRILDI** (`search/adaptive.py`,
      `bench/adaptive_mrl.py`) - Docker Desktop bu oturumda başlatıldı,
      ClickHouse'daki `clips_qwen3vl_emb_{256,512,1024,2048}` tablolarının
      (73 satır, önceden yüklü) hepsi doğrulanıp GERÇEK 28 sorgu × 19 sekans
      koşuldu (95 saniye, CPU metin embed + gerçek ClickHouse sorguları).
      392 satır: 6 referans strateji (2048d/256d/512d × exact/HNSW) + 2
      adaptif strateji (256→2048, 512→2048) × 4 candidate_k (20/50/100/200).
      Artifact: `artifacts/search_runs/adaptive_mrl_visdrone_bf236d0b76/`
      (manifest, query_results.ndjson, metrics.json, dataset_manifest.json,
      strategy_matrix.csv, gerçek EXPLAIN planları).
      **Gerçek bulgu (pilot, 28 sorgu - bağlayıcı değil)**: exact stratejiler
      hiç underfilled değil (0/28); `auto` (HNSW'e düşebilen) stratejiler
      hem referansta (8-12/28) hem adaptif adaylarda (42-45/112) underfilled
      gösteriyor - Faz 2'nin "auto postfiltering LIMIT'i doldurmayabilir"
      bulgusunu farklı bir yüzeyde doğruluyor. Agreement_vs_2048_exact_at10
      ortalama ~0.81-0.82 (256/512→2048 arasında fark küçük). nDCG@10
      256d_exact (0.398) ve 512d_exact (0.397), 2048d_exact'e (0.371) göre bu
      28 sorguda DAHA YÜKSEK çıktı - küçük örneklemde gürültü olabilir,
      **kesin bir "küçük boyut daha iyi" sonucu DEĞİL**.
      **BAĞLAYICILIK UYARISI (zorunlu)**: "Bu sonuç 28 sorguluk pilot
      değerlendirmedir. İstatistiksel olarak bağlayıcı üretim kararı
      değildir." 150-sorgu eşiği hâlâ geçilmedi; bağlayıcı karar MSR-VTT'nin
      1000 sorguluk GPU koşumunu (aşağıda, GPU yok) bekliyor.
- [x] `bench/search_run_artifact.py`: `artifacts/search_runs/<run_id>/`
      sözleşmesi (manifest/ndjson/metrics/dataset_manifest/strategy_matrix.csv
      /sql//explain/) - `run_id` `bench/spec.py::RunSpec` ile aynı desende
      deterministik (SHA1).
- [ ] **MSR-VTT Faz 6 (GPU) - HAZIRLANDI, ÇALIŞTIRILMADI**: bu makinede GPU
      yok (`torch.cuda.is_available()` = False, doğrulandı), gerçek
      1000-video Qwen embed'i başlatılmadı - `bench/gpu_gate.py` bunu
      doğru şekilde engelliyor. Hazırlanan altyapı: `scripts/
      msrvtt_embedding_cache.py` (dataset+model+n_sample'a bağlı cache_key,
      NDJSON tabanlı incremental resume - GERÇEK MSR-VTT videolarıyla test
      edildi, yarım kalan bir koşumda tamamlanmış videoları atladığı
      doğrulandı), `scripts/validate_msrvtt.py::qwen_cache_file()` +
      `compute_environment` alanı (torch/GPU introspection). Colab'de
      GPU açılınca: `python scripts/validate_msrvtt.py --models
      qwen3vl_emb_2048` otomatik resume-capable calisir.
- [ ] **Faz 7 (planner) ve dashboard entegrasyonu - YAPILMADI**: planner
      Faz 2/3'ün OLGUN (150+ sorgu) verisinden eşik türetmeli - şu anki
      pilot veri bunun için yetersiz, tahminle eşik yazmak yerine erteledi.
      Dashboard'a yeni sekme eklemek (Dataset Overview, MRL Dimension,
      Strategy Matrix, Scale/Storage) gerçek/olgun artifact'lar hazır
      olduğunda yapılmalı - "Sekme B/C/D" diye var sanılan sekmeler
      YOKTU (gerçek sekmeler: "1 Veri/GPU, 2 Pipeline, 3 Query, 4
      Accuracy, 5 Rapor"), boş/pilot veriyle yeni sekme eklemek erken
      olurdu.
- **Test**: 245/245 pytest geçti (bu bölüm için 72 yeni test - doğrulandı:
  `pytest --collect-only`; gpu_gate, strategy_matrix D2/D5, dataset
  registry/adapter, adaptive search, adaptive MRL bench, artifact
  contract, MSR-VTT cache/resume - hepsi gerçek yerel veriye veya
  sahte-ama-doğrulanabilir client'lara karşı).

## Açık kalanlar

- Tüm 56 sekans üzerinde toplu MP4/YOLO/model ingest henüz koşulmadı; mevcut
  kanıt 5 sekans/7 pencere smoke subsetine aittir.
- `gt_walking` gerçek annotation'da çalıştı fakat 5-10 sekanslık görsel
  ego-motion denetimi yapılmadı.
- Filtre A/B ve iki model eval teknik olarak üretildi; 5 video `top_k=10`'dan
  küçük olduğu için model sıralama kalitesi bu metrikle ayırt edilemiyor.
- FiftyOne dataset'i headless olarak 98 sample ile kuruldu; insanın GUI'de
  5-10 örnek `gt_walking` denetimi henüz yapılmadı.

## Sonraki kontrollü adım

Veri ve 5-sekans smoke hazır. Sonraki zorunlu kapı FiftyOne'da görsel GT
denetimidir. Mevcut ilk smoke'u
yeniden üretmek için:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/poc.ps1 download-data
powershell -ExecutionPolicy Bypass -File scripts/poc.ps1 frames -Sequence uav0000138_00000_v
powershell -ExecutionPolicy Bypass -File scripts/poc.ps1 windows
```

Tam 56-sekans CPU ingest'i saatler sürebileceği için ölçüm yapmadan başlatılmamalıdır.
