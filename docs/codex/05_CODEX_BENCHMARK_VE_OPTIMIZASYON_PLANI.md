# Codex ana talimatı — Benchmark, ClickHouse arama katmanı ve YOLO optimizasyonu

> Bu dosya `docs/codex/` altına `05_CODEX_BENCHMARK_VE_OPTIMIZASYON_PLANI.md`
> olarak konur. Aşağıdaki "Codex'e yapıştırılacak prompt" bloğu tek mesajda
> verilir; faz detayları aynı dosyada referans olarak kalır.

---

## Codex'e yapıştırılacak prompt

```text
Bu repodaki hibrit video arama POC'unu, kurumsal bir İHA şirketinde yerinde
(on-premise, internetsiz) çalışacak bir sistemin ön hazırlığı olarak geliştir.
Ana hedef BENCHMARK: her iddianın arkasında tekrar üretilebilir ölçüm olacak.

Başlamadan önce sırasıyla oku: CODEX_START_HERE.md, AGENTS.md, CONTEXT.md,
STATUS.md, TASKS.md, docs/codex/05_CODEX_BENCHMARK_VE_OPTIMIZASYON_PLANI.md
(bu plan). Çelişkide öncelik: çalışan kod/test sözleşmeleri > AGENTS.md >
CONTEXT.md > TASKS.md > docs/codex/. Eski hibrit-video-arama-poc-plani.md
yalnızca tarihsel kaynaktır.

Kurumsal kısıtlar (tasarım kararlarında bağlayıcı):
- Hedef ortam air-gapped/lokal. Runtime'da hiçbir model indirmesi, HF Hub
  çağrısı, telemetri/analitik çağrısı olamaz. Tüm model yükleme yolları
  local_files_only=True ve HF_HUB_OFFLINE=1 ile çalışacak şekilde
  düzenlenecek; ağırlıklar tek bir weights/ manifest'iyle paketlenecek.
- Gerçek telemetri (konum, irtifa, hız, platform) üretimde AYRI bir kaynaktan
  gelir. Repodaki YOLO-türevi filtre kolonları bunun vekilidir; şema ve sorgu
  katmanı "filtre kolonları dışarıdan da gelebilir" varsayımını bozamaz.
  OCR/video-içi-yazı-okuma kapsam DIŞI; ekleme.
- Sektörel ihtiyaç değişir: filtre kolon seti ve model registry'si eklenebilir
  olmalı, yeni kolon/model eklemek şemada ve config'te lokaldir, kod
  değişikliği minimaldir.

Ana iş sırası (her fazın kanıt kapısı planda; kanıtsız kutucuk işaretleme):
FAZ 0  Regresyon tabanı: 46 test + py_compile + doküman tutarsızlığı düzeltmesi.
FAZ 1  Benchmark altyapısı (bench/ paketi): sorgu seti, metrikler, zamanlayıcı,
       run manifest, tek HTML/JSON rapor. Bu faz bitmeden model/DB deneyi yok.
FAZ 2  ClickHouse arama katmanı doğrulaması: exact brute-force, HNSW,
       prefilter, postfilter+rescore stratejilerinin EXPLAIN kanıtlı davranışı;
       sentetik 100K/1M ölçek testi; recall@K = HNSW-vs-exact karşılaştırması;
       max_limit_for_vector_search_queries ve filtre stratejisi ayarları.
FAZ 3  YOLO optimizasyonu: VisDrone fine-tuned checkpoint bake-off'u
       (nano/small/medium/x), count-accuracy'nin annotation'a karşı ölçümü,
       batch inference, filtre kolon kalitesinin retrieval'a etkisi.
FAZ 4  Embedding bake-off (offline-uyumlu adaylar): mevcut iki model +
       VideoCLIP-XL + LanguageBind-Video; aynı bench harness, aynı GT.
FAZ 5  Fast/Balanced/Accurate profilleri + nihai Pareto raporu.

Çalışma kuralları:
- Yeni sayısal sabit koda gömme; config.yaml + common.load_config() kullan.
- Model başına ayrı ClickHouse tablosu korunur; farklı boyutlar tek HNSW
  kolonuna karışmaz. Yeni model eklerken AGENTS.md'deki 4 adımlı prosedürü izle.
- search/query.py::search(q, model_name, top_k=200, use_filters=True)
  imzasını değiştirme; davranış değişikliği parametre ekleyerek yapılır.
- Her faz sonunda testleri çalıştır; bench çıktılarında model ID, checkpoint
  revision, boyut, cihaz, ClickHouse sürümü ve ayar seti metadata olarak dursun.
- Ağ erişimi olmayan adımları mock'la geçme; erişim gerekiyorsa dur ve
  kullanıcıdan dosyayı elle yerleştirmesini iste (veri sözleşmesi kuralı gibi).
- microsoft/xclip (Ni) ile Ma ve ark. AOSM X-CLIP'i sonuçlarda ayrı adlandır.
- Doğrulanmayan hiçbir adımı tamamlandı yazma.

Her turun sonunda: değişen dosyalar+neden, çalıştırılan komutlar+özet çıktı,
geçen/kalan kabul kriterleri, doğrulanmamış riskler, tek net sonraki adım.
```

---

## Faz 0 — Regresyon tabanı ve doküman düzeltmeleri

**Girdi:** Mevcut repo, `.venv`, Docker/ClickHouse.

**İş:**
1. `pytest tests/ -v` ve tüm `.py` dosyalarında `py_compile`. 46/46 taban.
2. Bilinen doküman tutarsızlığını düzelt: `docs/codex/02_FIKIRLER_VE_KARARLAR.md`
   SigLIP2 için "açıkça `Siglip2Model` kullanır" diyor; gerçek çalıştırma
   kanıtı (AGENTS.md, RESULTS_SMOKE.md) tersini gösterdi — doğru olan resmî
   model kartındaki `AutoModel`. Notu koda uyacak şekilde güncelle.
3. `models/` adaptörlerine offline yükleme desteği ekle:
   `from_pretrained(..., local_files_only=True)` bir config bayrağıyla
   (`config.yaml: offline_mode: true/false`) kontrol edilsin; `HF_HUB_OFFLINE=1`
   set edildiğinde tüm pipeline'ın çalıştığını bir smoke testiyle kanıtla
   (ağırlıklar zaten lokalde — cache'ten yüklenmeli, ağ çağrısı olmamalı).
4. `scripts/package_weights.py` ekle: kullanılan tüm checkpoint'leri
   (YOLO .pt + HF snapshot dizinleri) tek bir `weights_manifest.json`
   (model ID, revision, SHA-256, boyut) ile `weights/` altında toplasın.
   Air-gapped kuruluma taşınacak paket budur. Git'e commit edilmez.

**Kanıt:** Test çıktısı; `HF_HUB_OFFLINE=1` ile geçen embed smoke logu;
`weights_manifest.json` örneği.

**Çıkış kapısı:** Offline modda X-CLIP + SigLIP2 + YOLO yüklemesi ağ hatası
vermeden geçiyor.

---

## Faz 1 — Benchmark altyapısı (ANA HEDEF, her şeyden önce)

Bu faz bitmeden hiçbir model/DB karşılaştırması "sonuç" sayılmaz. Amaç:
aynı sorgu seti + aynı GT + aynı zamanlayıcı ile her deneyin tek komutla
tekrar üretilebilmesi.

**İş:**

0. **Önce çalıştırma yüzeyini aç.** Repoda şu an `Makefile` hedefleri
   `help, download-data, infra-up, infra-down, schema, frames, windows,
   detect, embed, load, ingest, groundtruth, eval, search-report, fiftyone,
   test, clean` ile sınırlı; `bench` hedefi ve `scripts/poc.ps1`'in
   `ValidateSet` listesinde `bench` YOK. Bu adımların ilk işi: `Makefile`'a
   `bench:` hedefini ve `scripts/poc.ps1`'in `ValidateSet` dizisine `'bench'`
   değerini eklemek (mevcut `test`/`search-report` hedefleriyle aynı desende).
   Bu yapılmadan aşağıdaki "tek komutla" kanıt adımları çalışmaz.
1. **`bench/` paketi** oluştur:
   - `bench/runner.py` — bir `RunSpec` alır (model adı, filtre modu, arama
     stratejisi, window/stride profili, YOLO varyantı, top_k), pipeline'ın
     ilgili kısmını koşar, `artifacts/bench/<run_id>/` altına yazar.
   - `bench/timing.py` — aşama bazlı duvar-saati + (varsa) CUDA event
     zamanlayıcı: decode, YOLO, embed, CH load, query. `time.perf_counter`
     taban; GPU'da `torch.cuda.synchronize` sonrası ölç.
   - `bench/metrics.py` — mevcut `eval/metrics.py`'yi sarmalar; Recall@K,
     Precision@K (K=1,5,10), MRR ve interval-IoU tabanlı temporal hit ekler.
     Sorgu başına `n_gt` her satırda raporlanır.
   - `bench/report.py` — tüm run'ları tek `benchmark_report.html` + `.json`
     içinde birleştirir; `reports/` altındaki mevcut ClickHouse raporlayıcı
     desenini (aynı SQL/veri hem insan hem test tarafından okunur) izler.
2. **Run manifest zorunlu:** her run şunları kaydeder — git hash veya paket
   hash'i, OS/Python/Torch/ClickHouse sürümleri, GPU adı+VRAM, model ID +
   checkpoint revision + embedding boyutu + normalize durumu, config
   snapshot'ı, veri kapsamı (sekans listesi + pencere sayısı), toplam süre.
3. **Sorgu setini büyüt:** mevcut 6 GT sorgusu model ayrıştırmaya yetmez
   (5 video < top_k=10 problemi belgeli). `eval/make_groundtruth.py`'deki
   QUERIES sözlüğünü 56 sekans üzerinde en az 25-40 sorguya çıkar:
   tekli nesne (bus/truck/car/pedestrian/van...), sayısal ("en az 3 araba"),
   hareket (gt_walking), bileşik (nesne ∩ hareket, nesne ∩ nesne),
   negatif-kontrol ("tren" gibi corpus'ta olmayan kavram → boş dönmeli).
   Her sorgu Türkçe + İngilizce çiftiyle tanımlansın (model çoğu İngilizce
   eğitildi; dil farkı ayrı bir ölçüm satırı olur, sessiz bir bias olmaz).
4. **GT güvenilirlik kapısı:** `gt_walking` için 5-10 sekanslık FiftyOne
   görsel denetimi hâlâ açık. Bench sonuçları yayınlanmadan önce bu denetim
   yapılmalı ve hatalı GT örnekleri rapora eklenmeli; gerekiyorsa
   ego-motion telafisi (aynı karedeki statik nesne track'lerinin medyan
   hareketini çıkar) eklenip kalibre edilmeli.
5. **Ingest kapsamı:** tam 56-sekans ingest CPU'da saatler sürer. Önce
   sabit, temsili bir 15-20 sekanslık "bench subset" tanımla (config'te
   liste olarak; gece/gündüz, yoğun/seyrek trafik, otobüslü/otobüssüz
   çeşitliliği kapsasın). CUDA'lı ortam varsa tam set; yoksa subset ile
   ilerle ve raporda kapsamı açıkça yaz.

**Kanıt:** Tek komutla (`scripts/poc.ps1 bench` / `make bench`) üretilen
örnek rapor; iki kez üst üste koşulduğunda metriklerin aynı çıktığını
gösteren determinizm kontrolü (embedding float farkları toleransla).

**Çıkış kapısı:** `bench/runner.py` mevcut iki model × iki filtre modunu
subset üzerinde koşup raporu üretiyor; 46 test hâlâ geçiyor; yeni saf-mantık
testleri (metrics, manifest, spec parsing) eklendi.

---

## Faz 2 — ClickHouse arama katmanı: davranış doğrulaması + ölçek

Mevcut `sql/search_lab/` yedi sorgusu iyi bir iskelet; şimdi her stratejinin
GERÇEKTE ne yaptığını kanıtla ve ölçekte ölç. Dokümante edilmiş üç somut
risk var, üçü de burada kapanır:

**Bilinen riskler (araştırma ile doğrulanmış):**
- R1. ClickHouse, ek filtre bir skip index (minmax) ile değerlendirilebiliyorsa
  varsayılan olarak POST-filtering uygular — yani `06_hybrid_prefilter.sql`
  adının vaat ettiği davranış, ayar verilmezse gerçekleşmeyebilir. Kontrol
  ayarları: `vector_search_filter_strategy='prefilter'` ve
  `vector_search_index_fetch_multiplier`. Post-filtering ayrıca LIMIT'ten az
  satır döndürebilir (seçici filtrede sessiz kayıp).
- R2. `search/query.py` varsayılan `top_k=200`; planner HNSW'yi seçerse
  `max_limit_for_vector_search_queries` (varsayılan 100) hata döndürür.
  Küçük smoke'ta hiç tetiklenmedi; ölçekte kesin tetiklenir.
- R3. HNSW index'i part merge'lerinde yeniden inşa edilir ve inşa pahalıdır;
  sürekli ingest eden arşivde insert/merge'i yavaşlatır. Ayrıca 10M vektör
  ölçeğinde ClickHouse HNSW'nin scan-ağırlıklı kalabildiği canlı bir issue
  ile raporlu (#103466). Bunlar ClickHouse'u elemez — tek-SQL hibrit sorgu
  hâlâ ana değer — ama sayıyla bilinmeli.

**İş:**

1. **Strateji matrisi:** dört strateji × iki filtre seçiciliği (gevşek:
   `person_count>=1`; sıkı: `bus_count>=1 AND person_count>=3`) × iki tablo.
   Her hücrede: dönen satır sayısı, p50/p95 latency (50+ tekrar, warm-up
   sonrası), `EXPLAIN indexes=1` çıktısında vector index kullanımı,
   `rows_read`/`bytes_read`. Mevcut `reports/` runner'ını genişlet;
   `vector_index_in_plan` alanı zaten var, üstüne `rows_read` ekle.
2. **Ayar deneyleri (her biri ayrı bench satırı):**
   - `SETTINGS vector_search_filter_strategy = 'prefilter'` açık/kapalı.
   - `vector_search_index_fetch_multiplier` — {1.0, 3.0, 10.0} — sıkı
     filtrede LIMIT-altı dönme oranını ölç.
   - `hnsw_candidate_list_size_for_search` (ef_search) — {64, 256, 512} —
     recall/latency eğrisi.
   - Not: sadece 26.7.1'de gerçekten var olan ayarları kullan; her ayarı
     `SELECT * FROM system.settings WHERE name = '...'` ile önce doğrula,
     yoksa raporda "bu sürümde yok" diye işaretle.
3. **HNSW recall ölçümü:** her sorgu için exact brute-force top-K'yı ground
   truth kabul et; HNSW top-K ile kesişimi = recall@K. 14 satırda anlamsız;
   madde 4'teki sentetik ölçekte anlamlı hale gelir.
4. **Sentetik ölçek testi:** ayrı `bench_scale_<dim>` tablolarına (üretim
   tablolarına DOKUNMA) 100K ve 1M satır üret: gerçek smoke embedding'lerin
   etrafına Gauss gürültüsü + gerçekçi filtre kolon dağılımı (ör. satırların
   %5'i bus_count>=1). Ölç: index inşa süresi, index bellek/disk boyutu
   (`system.data_skipping_indices`), dört stratejinin p50/p95'i, HNSW
   recall@10. Kaynak yetiyorsa 10M; yetmiyorsa raporda "yapılmadı" yaz.
   270M ölçeğe ekstrapole etme — mevcut faz kuralı geçerli.
5. **Somut düzeltmeler (deney sonuçlarına göre):**
   - `search/query.py`'ye `strategy` parametresi ekle (varsayılan mevcut
     davranış): `'auto' | 'prefilter' | 'postfilter_rescore' | 'exact'`.
     SQL'i `sql/search_lab/` katalogundan üretsin ki insan/test/kod aynı
     sorguyu kullansın.
   - top_k > 100 için ya `max_limit_for_vector_search_queries`'i bilinçli
     yükselt (ve raporla) ya da top_k'yı stratejiye göre sınırla.
   - Embedding kolonuna `CODEC(NONE)` denemesi: dense vektör sıkışmaz,
     resmî öneri codec'i kapatmak — insert/read süresi farkını ölç,
     kazanç varsa schema.sql'e yorumuyla ekle.
   - Sorgu vektörünü string literal yerine binary bind ile göndermeyi dene
     (`clickhouse_connect` parametre binding); 512-1152 float'lık string
     parse maliyeti p50'de ölçülebilir olabilir. `_fmt_vector` fallback
     olarak kalsın.
   - Sürekli-ingest senaryosu için `materialize_skip_indexes_on_insert=0` +
     zamanlanmış `ALTER TABLE ... MATERIALIZE INDEX` desenini küçük bir
     deneyle göster (insert throughput farkı + sonradan materialize süresi).
6. **Bellek planı notu:** dokümandaki formülle iki tablonun 1M/10M satırdaki
   index bellek ihtiyacını hesapla ve `vector_similarity_index_cache_size`
   önerisini rapora yaz (varsayılan 5 GB; 1152d tabloda 1M satır ≈ 2.6 GB
   mertebesi — iki model + büyüme payıyla yetmez).

**Kanıt:** Genişletilmiş `clickhouse_search_report.{html,json}`; ölçek
tablolarının inşa/sorgu ölçümleri; her strateji için EXPLAIN çıktısı.

**Çıkış kapısı:** "Prefilter gerçekten prefilter mı?" sorusunun EXPLAIN
kanıtlı cevabı; sıkı filtrede LIMIT-altı dönme davranışının belgelenmesi;
100K ve 1M'de dört stratejinin karşılaştırma tablosu; R2 düzeltmesi merge'li.

---

## Faz 3 — YOLO optimizasyonu

Mevcut durum: `yolo26x.pt` ham COCO ağırlıklarıyla, 4 COCO sınıfıyla
(person/car/bus/truck) çalışıyor. Havadan küçük nesnede COCO-pretrained
dedektörlerin zayıf olduğu hem repo dokümanında itiraf edilmiş hem
literatürde belgeli. VisDrone'a fine-tune edilmiş hazır checkpoint'ler
mevcut (Hugging Face: `dronefreak/visdrone-detection-model-zoo` —
YOLOv8/v9/v10/11/26 aileleri, model kartları + benchmark sonuçlarıyla).
Bu, "accuracy VE hız birlikte kazanılabilecek" nadir nokta: küçük ama
fine-tune'lu model, büyük fine-tune'suz modeli geçebilir.

**İş:**

1. **Checkpoint edinimi (offline kuralına uygun):** ağ erişimi bench
   ortamında varsa `huggingface_hub.snapshot_download` ile indir ve
   `weights/` manifest'ine ekle; yoksa dur, kullanıcıdan dosyaları
   yerleştirmesini iste. Her checkpoint'in SHA-256'sını manifest'e yaz.
2. **Sınıf haritası soyutlaması:** `ingest/04_detect.py`'deki sabit
   `COCO_MAP`'i config'e taşı: `detector.class_map` model-varyantı başına
   tanımlanır (COCO id'leri ≠ VisDrone id'leri; VisDrone-native checkpoint
   pedestrian/people/car/van/bus/truck/motor... taksonomisi kullanır ve
   GT üretimiyle birebir hizalanır — bu hizalama başlı başına bir accuracy
   düzeltmesidir). `van` gibi yeni kolon eklemek gerekirse AGENTS.md model
   ekleme prosedürüne paralel bir "kolon ekleme" prosedürü olarak yap:
   config + schema.sql + parser sözlüğü + test.
3. **Dedektör bake-off matrisi (bench harness ile):**
   satırlar = {yolo26x-COCO (mevcut baseline), visdrone-yolo11n,
   visdrone-yolo11s, visdrone-yolo11m veya eldeki en yakın varyantlar,
   varsa visdrone-yolo26}; sütunlar = aşağıdaki metrikler.
   - **Count accuracy:** pencere başına `person_count/car_count/bus_count/
     truck_count` tahminini VisDrone annotation'dan türetilen gerçek
     sayıya karşı ölç: MAE + "eşik doğruluğu" (count>=1 ikili kararının
     precision/recall'u — filtre `>=1` kullandığı için asıl önemli metrik bu).
   - **Hız:** kare/sn (batch=1 ve batch=8/16), model yükleme süresi, VRAM.
   - **Downstream etki:** aynı embedding modeli sabitken dedektör varyantını
     değiştirip filtre-açık retrieval Recall@10/Precision@10 farkını ölç.
     "Yanlış filtre = kalıcı kaçırma" riskinin büyüklüğü bu satırda
     sayısallaşır.
4. **Inference optimizasyonları (her biri ayrı ölçüm):**
   - Batch inference: `window_features` şu an kareleri tek tek işliyor;
     n_sample karelerini tek batch'te ver.
   - Tek decode: video bir kez decode edilip aynı kareler YOLO + embedding
     modellerine paylaştırılsın (mevcutta iki ayrı `VideoCapture` turu var).
   - `imgsz` ablation'ı: 640 vs 960/1280 — VisDrone küçük nesnede yüksek
     çözünürlük mAP'i belirgin artırabilir; hız bedeliyle birlikte raporla.
   - FP16 (GPU varsa): count accuracy değişimini doğrulayarak.
   - `n_sample` — {3, 6, 12}: pencere başına örneklenen kare sayısının
     count kararlılığına ve süreye etkisi (medyan zaten gürültüyü kırpıyor;
     3 yetiyor olabilir).
5. **Karar:** count-eşik-doğruluğu / hız Pareto'suna göre tek varsayılan
   dedektör seç; `config.yaml`'da varyant adı olarak kaydet; kaybeden
   varyantlar registry'de kalır (sektörel ihtiyaç değişebilir kuralı).

**Kanıt:** Dedektör karşılaştırma tablosu (count MAE, eşik P/R, fps, VRAM,
downstream Recall@10); seçim gerekçesi; yeni testler (class_map yükleme,
count hesaplama regresyonu).

**Çıkış kapısı:** VisDrone-tuned en az iki varyant gerçek veride ölçüldü;
varsayılan dedektör kararı sayıyla gerekçelendirildi.

---

## Faz 4 — Embedding bake-off (offline-uyumlu adaylar)

MVEB (Massive Video Embedding Benchmark, MTEB ekosistemi) bulgusu yön
veriyor: retrieval görevinde "multimodal binding" ailesi (LanguageBind tarzı
contrastive modeller) önde; dev MLLM-tabanlı embedding'ler sınıflandırma/QA
tarafında parlıyor. Görevimiz retrieval + lokal/offline cihaz — adaylar
küçük-orta boy contrastive modellerden seçilir. Ayrıca bağımsız bir
benchmark'ta 6B InternVideo2'nin zero-shot retrieval'da 12x küçük X-CLIP'in
gerisinde kaldığı ölçüldü — "büyük model alırız, olur" varsayımı yasak.

**Aday listesi (hepsi lokal ağırlıkla çalışır):**

| Adapter | Model | Boyut | Not |
|---|---|---|---|
| `xclip_hf_zeroshot` | microsoft/xclip-b16-zs | 512 | Mevcut; baseline |
| `siglip2_frameavg` | siglip2-so400m | 1152 | Mevcut; frame-avg baseline; CPU'da en pahalı |
| `videoclip_xl` | VideoCLIP-XL (EMNLP'24) | ~768 | Sunumdaki +29 puan R@1 iddiasının kaynağı; ViT-L/14, HF checkpoint'i lokal indirilebilir |
| `languagebind_video` | LanguageBind_Video | ~768 | MVEB retrieval liderleri ailesinden; HF'de açık checkpoint |
| (ops.) `xclip_ma_aosm` | Ma ve ark. X-CLIP | 512 | Raporun asıl önerisi; resmî repo var ama transformers-dışı, entegrasyon maliyeti yüksek — süre kalırsa |

**İş:**

1. Her aday için `models/<isim>.py` adaptörü (`VideoTextEmbedder` arayüzü,
   L2-normalize, `dim` doğru) + `_REGISTRY` + `schema.sql`'de doğru boyutlu
   `clips_<isim>` tablosu + `eval/run_eval.py::MODELS` — AGENTS.md prosedürü.
   VideoCLIP-XL/LanguageBind resmî ön-işlemeyi (kare sayısı, normalize,
   çözünürlük) model kartından birebir uygula; farklıysa nota yaz.
2. Offline doğrulama: her adaptör `HF_HUB_OFFLINE=1` ile lokal snapshot'tan
   yüklenmeli (Faz 0 mekanizması). Yüklenemeyen aday bench'e girmez,
   "offline paketlenemedi" olarak raporlanır.
3. Bench harness ile aynı subset + aynı sorgu seti üzerinde: model ×  filtre
   × (TR/EN sorgu) matrisi. Metrikler: Recall@{1,5,10}, Precision@10, MRR,
   embed süresi/pencere, embedding MB/video-saat, VRAM.
4. Depolama satırı: her modelin ClickHouse tablo boyutu + HNSW index boyutu
   (`system.data_skipping_indices`) rapora girer — 512 vs 768 vs 1152 boyutun
   depolama/bellek bedeli açıkça görünsün.
5. Karar kapısı: tek ana model + (kanıt varsa) top-N reranker olarak ikinci
   model önerisi. "Klip modeli hareket sorgusunda frame-average'i geçiyor mu?"
   sorusu kategori-kırılımlı (tekli/hareket/bileşik) tabloyla cevaplanır.

**Kanıt:** Model karşılaştırma tablosu (kalite + hız + depolama aynı tabloda);
kategori kırılımı; TR/EN farkı satırı; offline-yükleme logları.

**Çıkış kapısı:** En az bir yeni video-native model gerçek veride ölçüldü ve
mevcut ikiliyle aynı harness'ta karşılaştırıldı.

---

## Faz 5 — Profiller ve nihai rapor

**İş:**
1. `config.yaml`'a `profiles:` bölümü: `fast` (window 8/8, az kare, nano
   YOLO), `balanced` (8/6), `accurate` (8/4, tam kare, seçilen ana model).
   Kesin değerler Faz 3-4 ablation sonuçlarından gelir, tahminle yazılmaz.
2. Kare ablation'ı (Faz 4 kazananı üzerinde): X-CLIP-tipi için 8/16/32,
   frame-tabanlı için 4/8 — accuracy düşüşü ölçülerek en ucuz seçim.
3. Decode/cache/idempotency: video hash + run manifest; değişmeyen video
   yeniden ingest edilmez; kesilen iş kaldığı pencereden devam eder.
4. Retrieval cascade'i `search/`'e opsiyonel akış olarak ekle:
   exact filtre → HNSW top-100 → (varsa) güçlü model top-20 rerank →
   top-10 + interval merge. Cascade açık/kapalı bench satırı.
5. Nihai rapor: `docs/codex/04_KABUL_KRITERLERI_VE_RAPOR.md` şablonu +
   şu ekler: Pareto tablosu (profil × {Recall@10, sn/video-dk, MB/video-saat,
   query p95}), ClickHouse strateji önerisi (hangi seçicilikte hangi strateji),
   dedektör kararı, model kararı, offline paket içeriği (`weights_manifest`),
   üretime açık bağımlılıklar (gerçek telemetri şeması entegrasyonu, kurum
   altın sorgu seti, 10M+ ölçek kararı).

**Çıkış kapısı:** Üç profil gerçek koşuyla ölçüldü; rapor "tamamlandı /
kısmen doğrulandı / doğrulanmadı" ayrımıyla teslim edildi; TASKS.md yalnızca
kanıtlı kutularla güncellendi.

---

## Yasaklar ve sınırlar (tüm fazlar)

- Runtime ağ çağrısı ekleme; `gdown`/HF indirme yalnızca hazırlık
  scriptlerinde ve açık kullanıcı onayıyla.
- Üretim `clips_*` tablolarına ölçek-testi verisi yazma; `bench_scale_*`
  ayrı ve temizleme komutu raporlanır, kullanıcı istemeden silinmez.
- OCR / video-içi metin okuma / ses ekleme (kapsam dışı; telemetri ayrı).
- Küçük-N sonuçtan kesin dil ("kazandı", "en iyi") — N ve güven notu zorunlu.
- Mock inference'ı gerçek başarı yerine sayma.
- Büyük veri/ağırlık dosyası commit'leme (.gitignore zaten kapsıyor; yeni
  dizinler eklenirse .gitignore güncellenir).

## Kabul kriterleri özeti

- [ ] Faz 0: 46 test + offline-load smoke + weights manifest.
- [ ] Faz 1: bench harness, ≥25 sorguluk GT, determinizm kontrolü, tek rapor.
- [ ] Faz 2: 4 strateji × 2 seçicilik EXPLAIN-kanıtlı; 100K & 1M ölçek;
      HNSW recall@10; top_k/limit düzeltmesi.
- [ ] Faz 3: ≥2 VisDrone-tuned YOLO ölçüldü; count-eşik P/R + downstream
      etki tablosu; varsayılan dedektör kararı.
- [ ] Faz 4: ≥1 yeni video-native model aynı harness'ta; kalite+hız+depolama
      tek tabloda; TR/EN satırı.
- [ ] Faz 5: 3 profil ölçüldü; Pareto raporu; TASKS.md kanıtlı güncelleme.
