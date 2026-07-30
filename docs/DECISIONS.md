# Faz 7 kararları

## Faz 11 decisions

- 2026-07-30 — Kurum çalışma yolu kod varsayılanı olarak
  `clickhouse` + `512` + `pushdown` seçildi; araştırma yüzeyi ayrı
  `docker-compose.benchmark.yml` override'ında üç backend ve dört MRL boyutuyla
  korunuyor. Gerekçe: devre dışı servislerin startup/health/schema/ingest
  maliyetine girmemesi ve eski benchmark kabiliyetinin silinmemesi.
- 2026-07-30 — PostgreSQL metadata control-plane olarak her profilde başlıyor;
  pgvector extension/tablo/indexleri yalnız `pgvector` etkinse oluşturuluyor.
  Böylece kurum profili metadata ilişkilerini korurken kullanılmayan vector
  schema yüzeyini yaratmıyor.
- 2026-07-30 — Canonical Compose gerçek secret sağlamıyor: `.env.example`
  yalnız `CHANGE_ME_*` placeholder'ları içeriyor ve boş PostgreSQL/ClickHouse
  parolası Compose interpolation aşamasında fail-fast oluyor. Eski Faz 7 dosyası
  `.env.faz7` akışını compatibility için koruyor; güvenli kurum varsayılanı
  canonical `docker-compose.yml`.
- 2026-07-30 — CapERA kalite protokolü import-time global yerine lazy fonksiyon
  yapıldı. Gerekçe: kurum config'inde `datasets.capera` bulunmaması API import ve
  startup'ını bozmamalı; CapERA kalite komutu çağrılırsa eksiklik açık hata olarak
  kalmalı.
- 2026-07-30 — Manifest parser elle ve fail-closed yazıldı; Pydantic'e ikinci bir
  config modeli eklenmedi. Mutlak path/`..`/symlink escape, bilinmeyen canonical
  alan, eksik altitude reference ve velocity kind parse aşamasında reddediliyor.
  Gerekçe: yanlış semantik ingest başladıktan sonra düzeltilmemeli.
- 2026-07-30 — Pozitif `offset_s` hem absolute hem relative clock'ta telemetriyi
  video timeline'ında erkene taşır (`... - offset_s`). Bu işaret kod docstring'i,
  örnek YAML ve `docs/DATASET_MANIFEST.md` içinde tek anlamla sabitlendi.
- 2026-07-30 — `artifacts/faz11/preflight_example.json` gerçek kurum preflight
  PASS kanıtı değildir. Şablon/config reddetme yolu gerçek çalıştırıldı, fakat bu
  hostta kurum verisi/GPU/model bundle olmadığı için acceptance status `not_run`
  ve gereken tam komutla yayımlandı.
- 2026-07-30 — Streaming decoder birincil olarak PyAV 16.0.1, fallback olarak
  OpenCV headless kullanır. İlk denenmiş PyAV 14.4.0 macOS/Python 3.13 wheel
  sunmayıp FFmpeg 7 source-build önkoşulunda durduğu için 16.0.1'e pinlendi;
  16.0.1 wheel'i kuruldu ve aynı gerçek MP4 fixture testleri PyAV yolunda 7/7
  geçti. Fallback de ayrı koşumda 7/7 geçti.
- 2026-07-30 — Window sahipliği yalnız `t_start` ile belirlenir; decoder chunk
  başına bir kez açılıp `chunk_end + window_size` halo'suna kadar sequential
  ilerler. Generator tamamlanan ilk window'u tüm chunk decode edilmeden yield
  eder; corpus veya chunk frame'leri topluca listelenmez.
- 2026-07-30 — Telemetri bir video için sıralı series olarak tutulur, corpus
  seviyesinde tutulmaz. Continuous alanlar linear+median, circular derece
  alanları shortest-arc+circular mean, categorical alanlar LOCF+mode uygular;
  `extra` ayrı read-only payload olarak kalır.

## Faz 8 decisions

- 2026-07-30 - CapERA quality scope is test only: 1391 videos, 5 captions
  per video, exactly 6955 query/GT rows. Caption index is not treated as
  human/automatic provenance; caption_source remains unknown.
- 2026-07-30 - A0/system and A1/quality are separate readiness profiles.
  Missing A1 never blocks T1-T7.
- 2026-07-30 - Because pgvector 2048d is halfvec, exact cross-backend
  equality is gated only at float32-compatible 1024/512/256 dimensions.
  The 2048d result is a separate quantization experiment.
- 2026-07-30 - A/B/C values are labels, not distinct execution paths.
  T4 skips with reason pattern not implemented until real paths exist.
- 2026-07-30 - Negation and nonsense queries are exploratory rather than
  pass/fail gates. Paired bootstrap resamples video clusters, not query rows.

- 2026-07-30 — GPU/model indirmesi kritik yolu bloke etmesin diye standart servis imajı `synthetic/cached`, ayrı `docker-compose.gpu.yml` ise `real` Qwen modu için ayrıldı.
- 2026-07-30 — Embedding üretmeyen eski notebook 02 silinmedi; kanıt ve geçmiş korunarak `notebooks/_archive/` altına taşındı.
- 2026-07-30 — pgvector HNSW'nin 2000 boyut sınırı nedeniyle 2048d `halfvec`; fp16 cezasını ölçmek için 1024d hem `vector` hem `halfvec` tutulur.
- 2026-07-30 — Dataset görüntüsü indirmeden hazır, doğrulanmış 1.866 AU-AIR segmenti minimum uçtan uca veri yolu seçildi; SeaDronesSee ve MONET kritik yol dışında bırakıldı.
- 2026-07-30 — `cached` mod serbest metin için yalnız önceden üretilmiş `query_embeddings.json` girdilerini kabul eder; gerçek model yüklemeden bilinmeyen sorguya sahte vektör üretmez.
- 2026-07-30 — Kullanıcının açık talimatıyla önceki 7 commit ve Faz 7 çalışması doğrudan `main` üzerinde pushlanır; gece talimatındaki “push atma” kuralı bu teslim için geçersizdir.
- 2026-07-30 — Gradio'da yerleşik çift-tutamaklı range slider bulunmadığı gerçek konteyner importunda doğrulandı; her telemetri alanı aynı min/max semantiğini koruyan yan yana iki slider ile gösterilir.
- 2026-07-30 — Tam L2 matrisi uzun koşum olarak runner'da korunur; teslim artifact'i tüm 150 konfigürasyonu birer sorguyla ölçen ve `settings_json.execution=smoke` diye açık etiketlenen kısa koşumdur.

## Faz 9 (UI redesign) decisions

- 2026-07-30 — Renk/tipografi/grid: koyu "ops dashboard" teması (`--bg #0b1220` vb.), tek
  vurgu rengi (camgöbeği `--accent`), success/warning/danger yalnızca durum rozetlerinde.
  Gerekçe: talimat §1/§3 rengi serbest bırakıyor, öncelik "Jupyter/varsayılan Gradio
  hissi vermemek"; koyu teknik tema bunu en güçlü ayrıştıran seçenekti.
- 2026-07-30 — §0.5 additive alanlar UYGULANDI: `postgres.hydrate()` SELECT'ine
  `event_category, split, person_count, vehicle_count, bus_count` (LEFT JOIN, mevcut
  alanlar korunarak); `postgres.facets()`'e `counts: {person_count,vehicle_count,bus_count}`
  bloğu eklendi. Doğrulama: AU-AIR'de bu alanlar dolu (`event_category` hariç — tüm
  videolarda NULL, kartta otomatik gizleniyor). Testler: `service/tests/test_additive_fields.py`.
- 2026-07-30 — `facets().counts` bilinçli olarak UI'da bir **filtre** olarak bağlanmadı,
  yalnızca sonuç kartlarında rozet (`telemetry_badges`) olarak gösteriliyor. Gerekçe:
  `postgres.filter_segment_ids` yalnızca `event_category/split/video_id` +
  `altitude_m/velocity_mps/gimbal_pitch` destekliyor; person/vehicle/bus_count'u
  filtrelenebilir yapmak `filter_segment_ids`'e yeni SQL koşulu eklemek demek, bu da
  talimat §5'in yasakladığı "retrieval mantığına dokunma" sınırını aşıyor. Aktif
  olmayan bir slider'ı UI'da göstermek de Pattern A/B/C sorununun bir başka biçimi
  olurdu (kullanıcıyı yanıltır) — o yüzden hiç eklenmedi.
- 2026-07-30 — "Open details" (§2.4 madde 8) bir kart-içi accordion yerine ayrı bir
  "Sonuç Detayı" seçici + panel olarak uygulandı. Gerekçe: Gradio 6'nın `gr.HTML`
  bileşeni, kart içindeki bir DOM tıklamasını Python callback'ine bağlamak için özel
  JS-Python köprüsü ister (kırılgan); segment seçici `gr.Dropdown` + `gr.HTML` detay
  paneli aynı işlevi native Gradio olaylarıyla, kırılmadan sağlıyor.
- 2026-07-30 — **Regresyon-öncesi bir hata bulundu ve düzeltildi (backend'e dokunmadan):**
  orijinal UI'da varsayılan arama, gizli telemetri slider'larının (ör. AU-AIR'de
  `gimbal_pitch` — hiçbir satırda değeri yok) `value=0` varsayılanını her zaman
  `[0,0]` aktif filtresi olarak `/search`'e gönderdiği için 0 sonuç döndürüyordu.
  Bunu hem orijinal `service/ui/app.py`'yi ayrı bir portta çalıştırıp hem de ilk
  redesign taslağımda gözlemleyerek doğruladım — mevcut bir hata, benim eklediğim
  bir regresyon değil. Düzeltme UI katmanında: `_sanitize_telemetry()`, bir alanın
  `/facets`'te gerçek bounds'u yoksa o slider'ın değerini backend'e hiç göndermiyor.
  Detay: `docs/UI_REGRESSION_REPORT.md`.
- 2026-07-30 — Gradio 6.20 Slider/Dropdown bug: filtre rozetini canlı güncellemek için
  `.change()` bir slider'ı `inputs=` listesine koyunca, `/facets` yüklemesinin kendi
  `gr.update(minimum=,maximum=,value=)` batch'iyle yarışıyor — tarayıcı henüz eski
  değeri gönderirken sunucu zaten yeni `minimum`'u uygulamış oluyor ve Gradio
  "Value 0 is less than minimum value X" hatası fırlatıyor. Çözüm: rozet
  güncellemesi slider'larda `.release()`, dropdown'larda `.select()` olaylarına
  bağlandı — ikisi de yalnızca gerçek kullanıcı etkileşiminde tetikleniyor,
  `/facets`'in programatik güncellemesiyle asla yarışmıyor.
- 2026-07-30 — `service/Dockerfile.ui`'ye `COPY tests/fixtures ./tests/fixtures`
  eklendi (tek satır, additive). Gerekçe: örnek sorgu çipleri (§2.2)
  `tests/fixtures/queries_semantic.json`'dan okunuyor; bu dosya konteyner imajına
  hiç kopyalanmıyordu. Yol çözümü `app/config.py::_capera_protocol`'deki aynı
  iki-adaylı (repo kökü / konteyner `/app`) desenle yapıldı — yeni bir "dead code
  fallback" icat edilmedi.

## Faz 10 (gerçek embedding'e geçiş) decisions

- 2026-07-30 — §3.4 uygulandı: `vector_provenance` **ayardan değil, veriden**
  okunuyor. `datasets.vector_provenance text NOT NULL DEFAULT 'synthetic'`
  eklendi (idempotent `ADD COLUMN IF NOT EXISTS`); mevcut `auair` satırı ayrı
  bir UPDATE'e gerek kalmadan DEFAULT üzerinden doğru değere düştü (auair
  gerçekten sentetik). `ingest()` provenance'ı `settings.embedding_mode`'dan
  türetip yazıyor (`synthetic`→`synthetic`, diğerleri→`real`); bu tek yazma
  noktası, gelecekte hangi dataset hangi modda ingest edilirse edilsin kolonun
  doğru kalmasını sağlıyor — dataset bazlı hardcode yok.
- 2026-07-30 — `mode_details(dataset_id)` artık önce dataset'in DB'deki
  provenance'ına bakıyor; `synthetic` ise `settings.embedding_mode` ne olursa
  olsun danger banner döner. `dataset_id=None` veya dataset DB'de yoksa eski
  (global `embedding_mode`'a dayalı) davranışa düşer — bu, henüz ingest
  edilmemiş bir dataset için makul bir varsayılan.
- 2026-07-30 — UI'daki iki ayrı "sentetik" uyarı kontrolü de düzeltildi:
  `run_search`'teki sonuç banner'ı ve `_render_comparison`'daki karşılaştırma
  uyarısı artık `response["embedding_mode"]` yerine `response["vector_provenance"]`
  kullanıyor. Gerekçe: karışık bir veritabanında (gerçek CapERA + sentetik
  AU-AIR) global `embedding_mode` (ör. `hybrid_text`) AU-AIR sorgusunun
  vektörlerinin de gerçek olduğunu yanlış iddia ederdi — tam da talimat
  §1'in tarif ettiği risk. `embedding_mode` alanının kendisi (hangi modun
  sorgu vektörünü hesapladığını gösterir) değiştirilmedi, additive olarak
  `vector_provenance` eklendi.
- 2026-07-30 — Bu aşama CapERA verisi olmadan başlatıldı (talimat §3.1 gereği
  A1.1 FAIL iken §3.2+ durduruldu), çünkü §3.4 tamamen dataset-agnostik bir
  şema/kod değişikliği: doğrulaması için gerçek CapERA embedding'i gerekmiyor,
  yalnızca canlı auair verisiyle test edilebiliyor. Amaç, Colab ZIP'i geldiğinde
  §3.2-§3.8'e doğrudan geçebilmek.

## Faz 11 Aşama 4 decisions

- 2026-07-30 — Qwen resmi kaynak `HEAD` değeri doğrudan `git ls-remote` ile
  `393e2978d27852b0d0230d6994f37f9c15bed73c` olarak çözüldü ve mutable branch
  yerine env, Compose, script ve dokümanlarda sabitlendi. Model revision promptun
  verdiği `9f2f7e...7bda` değeridir.
- 2026-07-30 — CUDA image build'i kaynak/model indirmez. Bundle ayrı hazırlanır,
  dosya bazında SHA-256 doğrulanır ve read-only mount edilir. Bu yalnız model
  taşınabilirliğini sağlar; base image, wheel ve NVIDIA runtime ayrıca
  provision edilmeden tam air-gap iddiası kurulmaz.
- 2026-07-30 — Gerçek video batch'i tek `Qwen3VLEmbedder.process()` çağrısıdır;
  adaptör `[batch,2048]` finite float32 ve L2 normunu fail-closed doğrular.
  GPU yokken sentetik/cached fallback yapılmaz; smoke sonucu `not_run` olur.

## Faz 11 Aşama 5 decisions

- 2026-07-30 — Persisted legacy tablolar kanıtsız ALTER/DROP edilmedi. Yeni
  fiziksel storage `run_*`/`*_runs` tablolarıyla additive kuruldu; active pointer
  bulunmayan dataset araması compatibility için legacy yolu kullanır.
- 2026-07-30 — Search aktif run/provenance/model snapshot'ını request başında bir
  kez okur. Aynı değer filter, vector backend ve hydrate boyunca taşınır; request
  ortasında activation olsa bile iki run karışmaz.
- 2026-07-30 — Chunk retry yalnız inactive run + aynı chunk verisini temizler.
  Finalize bütün backend×dimension ve metadata count'ları eşleşmeden active
  pointer'ı değiştirmez. Eski active run immediate GC yapılmaz.
- 2026-07-30 — Legacy Qdrant point'lerinde güvenilir run provenance olmadığı için
  migration bunu yeniden-ID'lemek yerine manifest-driven re-ingest gereksinimi
  raporlar. Volume veya eski point'ler otomatik silinmez.

## Faz 11 Aşama 6 decisions

- 2026-07-30 — Generic manifest ingest yalnız `EMBEDDING_MODE=real` kabul eder;
  model/GPU eksiğinde synthetic fallback yapmaz. Legacy dataset-id loader'ları
  ayrı compatibility yolu olarak korunur.
- 2026-07-30 — Decode iterator chunk boyunca materialize edilmez. Qwen batch
  boyutu `EMBED_BATCH_SIZE`, DB writes aynı batch üzerinden yürür; yalnız enabled
  dimension/backend projeksiyonları üretilir ve her batch sonrası PIL frame'leri
  kapatılır.
- 2026-07-30 — Resume kimliği dataset + manifest hash'ten son incomplete run'dır.
  Committed chunk decode edilmez; incomplete chunk'ın run-scoped metadata ve
  vector satırları temizlenip yeniden yazılır. Report/hata yolları run-scoped'tur.

## Faz 11 Aşama 7 decisions

- 2026-07-30 — `pushdown` active run için gerçekten backend-native execution
  path oldu; Python candidate listesi üretilmez. Active pointer'ı bulunmayan
  legacy dataset açık `legacy_candidate_ids_compatibility` etiketiyle eski yolu
  kullanır.
- 2026-07-30 — Canonical filter registry `(dataset_id,run_id,field_name)`
  anahtarlıdır. ClickHouse gerçek Nullable canonical kolonları, Qdrant registry
  kaynaklı payload indexlerini, pgvector tek JOIN SQL'ini kullanır.
- 2026-07-30 — 350°–10° circular aralık OR, normal aralık AND derlenir.
  Adaptive base aynı predicate'i taşır; yalnız bounded `top_n` rerank ID listesi
  olabilir. Legacy benchmark listesi `LEGACY_CANDIDATE_LIMIT` aşarsa fail olur.

## Faz 11 Aşama 8 decisions

- 2026-07-30 — Medya servisi source URI'yi yalnız PostgreSQL active-run snapshot
  üzerinden çözer; canonical path `DATA_ROOT` dışında kalırsa fail-closed 403 verir.
  ZIP pseudo-path ve uzak URI'ler oynatılabilir dosya sayılmaz. FFmpeg shell string
  değil argüman listesiyle çağrılır; browser uyumluluğu için `libx264`, `yuv420p`,
  AAC ve `faststart` kullanılır. Cache anahtarı source path/stat, zaman aralığı ve
  codec/CRF ayarlarını kapsar; partial çıktı atomik rename olmadan görünür olmaz.
- 2026-07-30 — UI canonical alanları registry+bounds kaynağından gösterir. Mevcut
  altitude/velocity/gimbal slider ve `.release()`/`.select()` yarış önleme kararı
  korunur; diğer numeric/circular alanlar min/max tablosunda, `is_night` boolean
  seçiminde görünür. Manifest `extra` alanları yalnız detail panelinde read-only'dir.
- 2026-07-30 — Backend/dimension/strategy seçeneklerinin tek kaynağı
  `/strategies` endpoint'idir. API erişilemezse UI'nın import edilebilmesi için
  kurum profiliyle aynı dar compatibility varsayımı (`clickhouse`, `512`,
  `prefilter`) kullanılır; disabled benchmark seçenekleri uydurulmaz.

## Faz 11 Aşama 9 decisions

- 2026-07-30 — `API_TOKEN` boşken loopback development backward-compatible;
  doluyken `/health` dışındaki API yüzeyi Bearer ister. Non-loopback bind + boş
  token host preflight'ta config failure'dır. Token Settings repr, response ve
  log detayına konmaz.
- 2026-07-30 — HTML video element'i Authorization header ekleyemediği için media
  auth bypass edilmedi. Authenticated `/media/.../info` cevabı HMAC-SHA256 ile
  imzalı, kısa ömürlü, segment path + run ID scope'lu clip URL üretir. URL token'ı
  içermez; expiry veya signature değişirse 401 olur.
- 2026-07-30 — Resource değerleri donanıma göre env-driven ve opsiyoneldir.
  Boş API/UI limitleri Compose tarafından limitsiz bırakılır; ClickHouse boş
  server limitinde kendi otomatik davranışını kullanır. Evrensel kapasite sayısı
  uydurulmaz; preflight/pilot/GPU artifact ölçümleri esas alınır.
