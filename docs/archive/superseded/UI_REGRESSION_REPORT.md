# UI Regression Report (Faz 9)

Ortam: yerel Docker Compose (`docker-compose.faz7.yml`), `api`/`ui` imajları bu oturumda
`service/Dockerfile` ve `service/Dockerfile.ui`'den yeniden build edildi (additive alanlar +
yeni UI kodu içerecek şekilde). `embedding_mode=synthetic`, dataset=`auair` (1866 segment).

**Repo geneli test durumu:** `RUN_FAZ8_INTEGRATION=1 pytest -q` (repo kökünden) →
**403 passed, 1 skipped**. Tek skip, önceden bilinen ve beklenen
`test_t4_patterns.py::…` (`docs/DECISIONS.md`: "A/B/C values are labels, not distinct
execution paths"). Redesign öncesi taban 395 passed / 1 skipped idi; bu oturumda eklenen
17 test (`test_additive_fields.py` ×2, `test_ui_components.py` ×8, `test_t10_ui.py` ×7) ile
403'e çıktı — talimattaki "382 passed korunuyor veya artıyor" kriteri sağlanıyor.

| Fonksiyon | Önce | Sonra | Kanıt |
|---|---|---|---|
| Search (3 backend × float32 boyutları 1024/512/256, +2048 ayrı halfvec yolu) | çalışıyor | çalışıyor | `service/tests/test_t1_determinism.py::test_t1_cross_backend_exact_equality_only_for_float32_dimensions` (canlı `/search`, clickhouse/qdrant/pgvector) + `service/tests/test_engine.py::test_every_backend_strategy_combination_runs_on_200_item_corpus` |
| Dinamik facet'ler (null/boş alan gizleme) | çalışıyor | çalışıyor | `service/tests/test_t10_ui.py::test_t10_search_results` (canlı UI, auair: event_category+gimbal_pitch gizli, split+altitude+velocity görünür) + `service/tests/test_additive_fields.py::test_facets_expose_additive_counts_block_without_removing_existing_fields` |
| Telemetri filtreleri | çalışıyor | çalışıyor | `service/tests/test_t2_filters.py::test_t2_negative_filter_is_expected_candidate_shortage` |
| Compare sekmesi | çalışıyor (Dataframe) | çalışıyor (kart grid, aynı `_run_comparison` mantığı) | `service/tests/test_t10_ui.py::test_t10_comparison` → `artifacts/ui_redesign/comparison.png` |
| CSV export | çalışıyor | çalışıyor | `service/tests/test_ui_components.py::test_export_csv_writes_additive_fields_as_columns` |
| Diagnostics alanları (11 alan: candidate_count, returned_count, underfilled, underfilled_reason, plan_used_vector_index, indexed_vectors_count, filter_correctness, quality_vs_groundtruth, r_at_1, ndcg, embedding_mode) | görünüyor (ham `gr.JSON`) | görünüyor (biçimlendirilmiş panel, `null`→"ölçülmedi"/"—") | `service/tests/test_t10_ui.py::test_t10_search_results` + `service/tests/test_ui_components.py::test_diagnostics_panel_uses_olculmedi_for_null_indexed_vectors_not_zero` |
| Embedding mode banner | görünüyor | görünüyor (aynı `/health.embedding.message`, yeniden tasarlanmış rozet) | `service/tests/test_t10_ui.py::test_t10_synthetic_warning` → `artifacts/ui_redesign/synthetic_warning.png` |
| Medya yoksa fallback | yok (ham `file_path` string tabloda) | bilinçli placeholder | `service/tests/test_ui_components.py::test_media_slot_placeholder_matches_talimat_wording` + `artifacts/ui_redesign/no_media_state.png` |

## Bu oturumda bulunup düzeltilen bir regresyon-öncesi hata

Redesign öncesi UI'da varsayılan arama (`backend=clickhouse`, filtresiz) **0 sonuç** döndürüyordu
— hem orijinal `service/ui/app.py`'de hem ilk redesign taslağımda doğrulandı (bkz.
`docs/DECISIONS.md`). Kök neden: gizli/`visible=False` telemetri slider'ları (ör. AU-AIR'de
`gimbal_pitch`, hiç bounds'u olmayan bir alan) yine de sayısal bir `value` taşıyor (Gradio
Slider `None` tutamıyor); `_range(0, 0)` bunu aktif `[0, 0]` filtresi sanıp `/search`'e
gönderiyor, backend de `gimbal_pitch BETWEEN 0 AND 0` koşuluyla (tüm satırlar NULL olduğu için)
sıfır aday buluyordu. Düzeltme UI katmanında (`_sanitize_telemetry`, `service/ui/app.py`):
`/facets`'in o alan için gerçek bounds döndürmediği her durumda slider değeri backend'e
**hiç gönderilmiyor**. Retrieval/filtreleme SQL'ine dokunulmadı — talimatın §5 yasağına uygun.

## Ekran görüntüleri

`artifacts/ui_redesign/` — Playwright ile `service/tests/test_t10_ui.py` tarafından üretildi
(gerçek Chromium, gerçek Docker container, gerçek `/search` yanıtı):

| Dosya | İçerik |
|---|---|
| `home_empty.png` | İlk yükleme, henüz sorgu yok, health noktaları yeşil |
| `search_results.png` | 10 sonuç, ilk kart büyütülmüş, diagnostics+latency panelleri dolu |
| `result_expanded.png` | "Sonuç Detayı" seçiciden #2 seçilmiş, detay paneli güncellenmiş |
| `advanced_settings.png` | Advanced Search Settings açık, Pattern yanında "not implemented" rozeti |
| `comparison.png` | 6 backend×boyut kartı, `NOT INTERPRETABLE` rozetleri (synthetic) |
| `synthetic_warning.png` | Üst status rozeti — `/health.embedding.message` |
| `no_media_state.png` | Tek bir sonuç kartının medya placeholder'ı, kırmızı/turuncu yok |

## Kapsam dışı bırakılanlar (bkz. docs/DECISIONS.md)

- `facets().counts` (person/vehicle/bus) yalnızca **görüntüleme** amaçlı eklendi (sonuç
  kartlarında rozet); aralık filtresi olarak UI'ya bağlanmadı çünkü backend `filter_segment_ids`
  bu alanları desteklemiyor ve bunu eklemek retrieval mantığına dokunmak anlamına gelirdi (§5 yasak).
