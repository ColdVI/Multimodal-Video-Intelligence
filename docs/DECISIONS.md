# Faz 7 kararları

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
