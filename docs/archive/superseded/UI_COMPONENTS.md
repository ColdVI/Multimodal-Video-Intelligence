# UI Components (Faz 9)

`service/ui/components.py` — saf Python fonksiyonları, girdi olarak `/search` `/facets` `/health`
yanıt sözleşmesindeki (bkz. `UI_REDESIGN_TALIMATI.md` §0.1-§0.3) dict'leri alır, HTML string
döner. Hiçbiri Gradio state'ine dokunmaz; `service/ui/app.py` bunları `gr.HTML(...)` içine koyar.
Stil `service/ui/static/theme.css`'teki tek CSS değişken bloğundan (`--bg`, `--surface`,
`--accent`, `--space-*`, `--radius-*`, `--font-*`, ...) gelir.

## Üst düzey / durum

| Fonksiyon | Amaç |
|---|---|
| `top_bar(product_name, dataset_id, health)` | Ürün adı, aktif dataset, pg/ch/qdrant health noktaları. |
| `status_badge(message, level)` | `/health.embedding` mesajını `danger/success/info` tonunda gösterir — metni **kendisi üretmez**, API'den okur (§2.1). |
| `warning_banner(text, level)` | Tek satır uyarı şeridi (synthetic uyarısı, karşılaştırma uyarısı). |
| `section_header(title, subtitle=None)` | Panel başlığı (ör. "Observability / Diagnostics"). |
| `pattern_not_implemented_badge()` | Pattern A/B/C seçicisinin yanına konan "not implemented" rozeti (§0.6). |
| `filter_group_header(title, active_count)` | "Filtreler · N aktif" rozeti. |

## Metrik / skor / zaman

| Fonksiyon | Amaç |
|---|---|
| `metric_card(label, value, sublabel=None)` / `metric_grid(cards)` | p50/p95/tekrar gibi tekil metrik kutuları. |
| `score_indicator(score)` | Skor sayısı + dolgu çubuğu; `score=None` → "—". |
| `time_range_bar(t_start, t_end)` | Segmentin **yaklaşık** konumunu gösteren mini çubuk. API `duration_s` döndürmediği için referans süre `max(t_end*1.25, t_end+20, span*4)` ile hesaplanır ve `title`/caption'da "yaklaşık" olarak işaretlenir — asla kesin video süresi gibi sunulmaz. |
| `telemetry_badges(result)` | irtifa/hız/gimbal/event_category/person/vehicle/bus rozetleri; `None` olan alan **hiç yazılmaz**. |

## Sonuç kartı / medya / detay

| Fonksiyon | Amaç |
|---|---|
| `search_result_card(result, rank, primary=False)` | Tek sonuç kartı; `primary=True` ilk sonucu büyütür (§2.4). |
| `result_list(results)` | Kart listesi; boşsa `empty_state("no_results")`'a düşer. |
| `media_slot(result, src=None)` | §0.4'teki bilinçli placeholder: dosya adı (son `/` sonrası), zaman aralığı, "medya önizlemesi bu ortamda servis edilmiyor" notu. `src` verilirse `<video>` render eder — **bugün hiçbir çağrı `src` vermiyor**, ileride medya eklenirse tek değişiklik çağıran taraftadır. |
| `result_detail_panel(result, meta)` | §2.5 detay paneli: medya slotu + tüm metadata/telemetry alanları (sadece dolu olanlar) + `backend/strategy/dimension` + `candidate_count → returned_count` daraltma notu. |

## Gecikme / diagnostics

| Fonksiyon | Amaç |
|---|---|
| `latency_row(label, ms, max_ms, emphasize=False)` / `latency_panel(timings, stats)` | 5 aşamalı gecikme çubukları + p50/p95/tekrar kartları. |
| `diagnostics_panel(diagnostics, embedding_mode)` | §2.7'deki 11 alanın tümü (`candidate_count` … `ndcg` + `embedding_mode`); `indexed_vectors_count=None` → **"ölçülmedi"** yazar, asla "0" yazmaz. |

## Durumlar (§2.9)

| Fonksiyon | Amaç |
|---|---|
| `empty_state(kind, detail=None)` | `no_query`, `no_results`, `filter_too_narrow`, `cached_query_missing`, `backend_unavailable`, `quality_unavailable`, `no_selection` — her biri kendi ikon/başlık/gövdesiyle. |
| `loading_state(message, cold_start=False)` | `cold_start=True` → hybrid_text ilk sorgu için ölçülmüş gerçek süreleri (~28s model + ~43s ilk sorgu) yazan metin + CSS animasyonlu ilerleme çubuğu. |
| `error_state(message, raw_detail=None)` | Kullanıcıya dost mesaj; ham exception/traceback yalnızca `<details>` içinde, kapalı. |

## Karşılaştırma

| Fonksiyon | Amaç |
|---|---|
| `comparison_group_header(title)` / `comparison_grid(cards)` / `comparison_card(row)` | §2.8: `embedding_mode`'a göre gruplanmış kartlar, `interpretable` durumu görsel olarak baskın (yeşil/kırmızı rozet), `embedding_mode=synthetic` için üstte uyarı şeridi. |

## Neden bu kadar "pure function"?

Gradio 6'da `gr.HTML` bileşenine basılan string, Python tarafında test edilebilir —
`service/tests/test_ui_components.py` bu fonksiyonları canlı sunucu/Playwright olmadan,
saniyeler içinde doğrular (null-handling, XSS-güvenli escape, placeholder metni). Canlı
tarayıcı gerektiren etkileşim/entegrasyon kanıtı ise `service/tests/test_t10_ui.py`'de.
