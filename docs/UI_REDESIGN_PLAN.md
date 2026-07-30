# UI Redesign Plan (Faz 9)

Kaynak talimat: `UI_REDESIGN_TALIMATI.md` (repo kökü). Bu doküman o talimatın
§6 adım 2'sinde istenen plan çıktısıdır.

## 1. Mevcut durum analizi

`service/ui/app.py` (271 satır, tek dosya):

- Gradio'nun varsayılan teması + iki satırlık CSS (`faz7-banner`) dışında hiçbir
  görsel kimlik yok. `artifacts/ui_smoke.png` bunu doğruluyor: standart Gradio
  mavi/turuncu paleti, notebook hissi veren dikey blok yığını.
- Sonuçlar `gr.Dataframe` — ham tablo. Skor, zaman aralığı, telemetri hepsi aynı
  hücre önceliğinde; ilk sonuç diğerlerinden görsel olarak ayrışmıyor.
- `latency` ve `diagnostics` çıplak `gr.JSON` — okunabilir ama teknik olmayan
  bir kullanıcı için (ya da hızlı tarama için) ağır.
- Medya kavramı hiç yok: `file_path` alanı tabloda ham string olarak duruyor,
  kırık `<img>`/`<video>` denemesi yok ama "medya yok" durumu da açıklanmıyor.
- Filtre paneli statik: `event_category`/`split`/`video_id` dropdown'ları var
  ama boş/`null` alanlar gizlenmiyor, sadece `visible=False` ile slider'lar
  saklanıyor (telemetri için). Aktif filtre sayacı yok, "Clear filters" yok.
- Pattern seçici (`A/B/C`) sanki gerçek bir yürütme yolu seçiyormuş gibi
  duruyor; `docs/DECISIONS.md`'ye göre bu sadece bir etiket (§0.6).
- Empty/error/loading state'i yok: hata durumunda `{"error": "ExceptionType: msg"}`
  ham JSON olarak `latency`/`diagnostics` kutularına yazılıyor.
- T10 (Playwright UI smoke) hiç yazılmamış; `service/requirements-test.txt`
  playwright'i pin'liyor ama tarayıcı hiç kurulmamıştı (bu oturumda kuruldu).

## 2. Bilgi mimarisi

```
Üst bar
  ├─ Ürün adı + embedding_mode rozeti (StatusBadge, /health'ten)
  └─ Backend health noktaları (pg/ch/qdrant)

Ara (ana sekme)
  ├─ Sorgu kutusu + örnek sorgu çipleri (tests/fixtures/queries_semantic.json)
  ├─ Dataset seçimi + top_k          [kullanıcı buraya dokunmadan arayabilir]
  ├─ Filtre paneli (collapsible)
  │    ├─ Metadata: event_category / split / video_id — boş/null → gizli
  │    ├─ Telemetri: altitude / velocity / gimbal_pitch — null → gizli
  │    └─ Aktif filtre rozeti + Clear filters
  ├─ Advanced Search Settings (kapalı accordion)
  │    └─ backend / strategy / dimension / adaptive_mrl / pattern(+rozet) / repeats
  ├─ Sonuçlar (SearchResultCard listesi, ilk kart büyük)
  │    └─ sıra+skor / zaman çubuğu / video_id+t / telemetri rozetleri /
  │       caption(varsa) / segment_id(kopyalanır) / MediaSlot placeholder /
  │       Open details
  ├─ Sonuç detayı (seçili karta göre genişleyen panel)
  └─ Observability/Diagnostics (ayrı, görsel olarak ayrışmış panel)

Karşılaştır (ikinci sekme)
  └─ backend karşılaştırma kartları + interpretable rozetleri + senaryo grupları

Durumlar: empty / no-results / filter-too-narrow / no-media / cached-miss /
          model-loading(progress) / backend-unavailable / synthetic-warning
```

## 3. Görsel yön

- Koyu, teknik "ops dashboard" temeli: nötr gri-lacivert yüzeyler
  (`--bg #0b1220`, `--surface #121a2b`, `--surface-2 #182238`), ince kenarlık
  (`--border`), tek vurgu rengi (`--accent`, mavi-camgöbeği), success/warning/
  danger yalnızca durum rozetlerinde kullanılır (talimat §0.4: placeholder
  hata rengi kullanmaz).
- Tipografi: `--font-sans` (sistem sans, Segoe UI/Inter fallback) gövde için,
  `--font-mono` segment_id / dosya yolu / sayısal metrik için.
- Skor görselleştirmesi: yatay dolgu çubuğu (0-1 arası) + sayısal değer —
  halka yerine çubuk seçildi çünkü liste içinde tarama hızını artırıyor.
- Zaman aralığı çubuğu: video toplam süresi bilinmediği (API'de `duration_s`
  dönmüyor) için `t_end`'in ~%20 fazlası referans alınarak orantılı bir mini
  çubuk çiziliyor; bu bir yaklaşıklık olduğu `title` tooltip'inde belirtiliyor.

## 4. Bileşen listesi (`service/ui/components.py`)

`StatusBadge`, `MetricCard`, `SearchResultCard`, `MediaSlot`, `LatencyRow`,
`EmptyState`, `WarningBanner`, `SectionHeader`, `FilterGroup`, `ScoreIndicator`,
`TimeRangeBar` — talimat §3'te sayılanların birebir aynısı. Hepsi saf Python
fonksiyonu, girdi alır HTML string döner; Gradio state'e dokunmazlar.

## 5. Değiştirilecek / eklenecek dosyalar

- `service/ui/app.py` — yeniden yazılacak (Gradio Blocks iskeleti aynı kalır,
  render mantığı `components.py`'ye taşınır).
- `service/ui/components.py` — yeni.
- `service/ui/static/theme.css` — yeni, tek CSS değişken bloğu + bileşen
  stilleri.
- `service/app/db/postgres.py` — `hydrate()` ve `facets()` additive alanlar
  (§0.5): LEFT JOIN ile `event_category`, `split`, `person_count`,
  `vehicle_count`, `bus_count`.
- `service/tests/test_t10_ui.py` — Playwright smoke + 7 ekran görüntüsü.
- `service/tests/test_additive_fields.py` — §0.5 alanlarının varlığını
  doğrulayan testler.
- `docs/UI_COMPONENTS.md`, `docs/UI_REGRESSION_REPORT.md` — yeni.

## 6. Riskler / notlar

- UI konteyner imajı (`video-search-faz7-ui`) bu oturumda yeniden build
  edilmeyecek; doğrulama canlı API konteynerine (`localhost:8000`) karşı
  yerel Python süreciyle yapılacak. Docker imajı `docs/BLOCKERS.md`'ye not
  düşülecek (rebuild gerektiği için ayrı, kullanıcı onaylı bir adım).
- `embedding_mode` şu an `synthetic` (canlı health kontrolünde doğrulandı).
  `hybrid_text` cold-start ekranı gerçek canlı veriyle üretilemez; bu ekran
  gerçek zamanlama sabitleriyle (`docs/BLOCKERS.md`'de ölçülü: 28.0s/43.2s/0.74s)
  statik olarak, ama kod yolunda gerçek progress state mantığıyla tasarlanacak.
