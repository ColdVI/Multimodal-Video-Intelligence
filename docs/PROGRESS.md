# Faz 7 ilerleme günlüğü

## Faz 11 progress

| Date | Stage | Result | Evidence |
|---|---|---|---|
| 2026-07-30 | Aşama 0 baseline | Başlangıç SHA/dirty state, root ve service testleri, 192 Python dosyasının syntax kontrolü ve iki mevcut Compose config'i kaydedildi. Root test ortamında eksik `clickhouse_connect`; service koşumunda absent CapERA verisine eager bağlanan iki test açıkça başarısız. | `artifacts/faz11/baseline.json`, `docs/PORTABILITY_AUDIT.md` |
| 2026-07-30 | Aşama 1 profiles | Kurum defaultu ClickHouse/512/pushdown, benchmark override üç backend/dört boyut, lazy CapERA config, enabled-only health/stats/strategies/schema/ingest, loopback bind, internal DB ports ve boş-secret fail-fast eklendi. | Root `307 passed, 42 skipped`; service `52 passed, 17 skipped`; üç Compose config PASS; Docker daemon kapalı olduğundan canlı AU-AIR smoke NOT RUN. |
| 2026-07-30 | Aşama 2 manifest/preflight | Relative-path güvenlikli YAML manifest, filename/CSV pairing, canonical/extra telemetri semantiği, absolute/relative clock formülleri, host+container read-only preflight ve belgeli exit code'lar eklendi. | Contract testleri PASS; `artifacts/faz11/preflight_example.json` gerçek çalıştırıldı ve kurum verisi/GPU/model bundle yokluğu nedeniyle `not_run`. |
| 2026-07-30 | Aşama 3 streaming | PyAV-first/OpenCV-fallback probe+decode, chunk/halo ownership, drop/pad partial window, erken-yield bounded generator, deterministic segment ID, `WindowRecord` iterator ve continuous/circular/categorical telemetri aggregation eklendi. | Sentetik küçük MP4 yalnız contract fixture: OpenCV 7/7, pinli PyAV 16.0.1 gerçek decoder yolu 7/7; performans kanıtı değil. |

## Faz 8 progress

| Date | Stage | Result | Evidence |
|---|---|---|---|
| 2026-07-30 | Protocol | Local CapERA test JSON counted directly as 1391 items, five captions each and 6955 total. | Real source-file read |
| 2026-07-30 | Ingest/GT | Loader uses CapERAAdapter test surface, split-qualified IDs and unknown caption source. | 46 target tests passed; 5 readiness skips |
| 2026-07-30 | Readiness | system/quality, JSON and strict profiles implemented; A1 does not affect system. | Unit tests and CLI contract |
| 2026-07-30 | Hybrid text | Atomic process-safe revision+text cache, lazy lock, CPU fp32 fallback and warm-p50 decision. | Real CPU: load 28.030 s, cold query 43.173 s, warm p50 0.739 s; hybrid_text selected |
| 2026-07-30 | Quality | NPY+Parquet bulk queries, video-cluster bootstrap, exploratory controls and separate halfvec experiment. | Code/unit evidence; real T8 waits for A1 |
| 2026-07-30 | Final validation | System readiness passed; quality readiness remained explicitly blocked on real CapERA artifacts. | 382 passed, 5 skipped; matrix 318 PASS, 1 EXPLORATORY, 3 SKIP, 0 FAIL |

| Zaman (Europe/Istanbul) | Aşama | Sonuç | Kanıt türü |
|---|---|---|---|
| 2026-07-30 | Başlangıç | `main` dalının `origin/main` önünde olan önceki 7 commiti başarıyla pushlandı. | GERÇEK git remote işlemi |
| 2026-07-30 | Aşama 1 | Ayrık `service/` iskeleti, üç embedding modu, CPU/GPU Docker ayrımı ve compose topolojisi oluşturuldu; eski notebook 02 arşivlendi. | Kod/config doğrulaması |
| 2026-07-30 | Aşama 2 | pgvector/pg16, ClickHouse 25.8, Qdrant 1.12.4 ve FastAPI gerçek Docker healthcheck'leri geçti. | GERÇEK Docker servisleri |
| 2026-07-30 | Aşama 3 | Hazır AU-AIR parquet'inden 1.866 segment; dört boyutta pgvector, ClickHouse ve Qdrant'a yüklendi. `/stats` her store için 1.866 döndürdü. | GERÇEK veri + SENTETİK embedding |
| 2026-07-30 | Aşama 4 | ClickHouse/Qdrant/pgvector sorguları, negatif filtre ve `top_k=200` canlı API'de geçti; sentetik kalite alanları NULL kaldı. | GERÇEK DB/API + SENTETİK embedding |
| 2026-07-30 | Aşama 5 | Dinamik facet/min-max kontrolleri, gecikme/diagnostics, CSV ve karşılaştırma sekmeli Gradio UI HTTP 200; 1440×1200 smoke görüntüsü üretildi. | GERÇEK UI smoke |
| 2026-07-30 | Aşama 6 | Tam 150-konfigürasyon matrisi smoke ölçümü yazıldı; 150/150 `embedding_mode=synthetic`, 150/150 kalite NULL, hata 0; exact stratejiler float32 stable NumPy referansına karşı recall@10=1,00. | SENTETİK sistem/gecikme smoke |
| 2026-07-30 | Aşama 7 | 9 kod hücreli Colab üretim notebook'u nbformat ile geçerli; GPU olmayan ilk hücre doğru ve açık mesajla duruyor. | Notebook sözleşme doğrulaması |
| 2026-07-30 | Aşama 8 | Windows tam verify 47,2 sn'de geçti; repo-geneli 347/347 pytest, compileall, compose config ve notebook CPU kapısı geçti. | GERÇEK teslim doğrulaması |

## Faz 10 (gerçek embedding'e geçiş) progress

| Date | Stage | Result | Evidence |
|---|---|---|---|
| 2026-07-30 | Ön kapı | `readiness_check.py --profile quality` çalıştırıldı: `A1.1 FAIL` — Colab ZIP'i (`artifacts/embeddings/`) yerelde yok. Notebook 07 kodu zaten hem video hem caption/query embedding'ini doğru formatta üretiyor (ek script gerekmiyor); eksik olan tek şey kullanıcının §2 Colab koşumu. | Gerçek CLI çıktısı |
| 2026-07-30 | §3.4 provenance | `datasets.vector_provenance` kolonu eklendi (idempotent migration); `ingest()`, `mode_details()`, `/health`, `/stats`, `/search`, UI banner'ları (arama sonucu + karşılaştırma) dataset'in gerçek provenance'ını kullanacak şekilde güncellendi. | Kod + canlı Docker doğrulaması: `/stats`, `/health?dataset_id=auair`, `/search` üçü de `vector_provenance=synthetic` döndü |
| 2026-07-30 | Regresyon | `RUN_FAZ8_INTEGRATION=1 pytest -q` (repo kökünden, canlı `api`+`ui`+3 DB container'a karşı): `404 passed, 1 skipped` (önceki `403 passed, 1 skipped`'den regresyon yok, +1 yeni provenance testi). `test_engine.py`'deki `fake_corpus` fixture'ı yeni `vector_provenance` alanını içerecek şekilde güncellendi (KeyError düzeltmesi). | Gerçek pytest çıktısı |
| 2026-07-30 | G3 kapısı | Kısmen geçti: auair yarısı (`synthetic`+danger banner) canlı doğrulandı. capera yarısı (`real`+success) CapERA henüz ingest edilmediği için doğrulanamadı — §2 tamamlanınca yapılacak. | `test_additive_fields.py::test_mixed_provenance_database_reports_auair_as_synthetic_with_danger_banner` |

## Ölçüm kapsamı

| Kapsam | Kullanılabilecek iddialar |
|---|---|
| GERÇEK embedding / gerçek araştırma ölçümleri | Yalnız daha önce üretilmiş ve kaynağı belirtilen artifact sonuçları; Faz 7 sentetik arama sıralamasıyla karıştırılmaz. |
| SENTETİK embedding | Yalnız DB/index/API/UI bütünlüğü ve gecikme; kalite kolonları daima NULL. |
