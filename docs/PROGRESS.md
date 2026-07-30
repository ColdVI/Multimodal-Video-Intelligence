# Faz 7 ilerleme günlüğü

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

## Ölçüm kapsamı

| Kapsam | Kullanılabilecek iddialar |
|---|---|
| GERÇEK embedding / gerçek araştırma ölçümleri | Yalnız daha önce üretilmiş ve kaynağı belirtilen artifact sonuçları; Faz 7 sentetik arama sıralamasıyla karıştırılmaz. |
| SENTETİK embedding | Yalnız DB/index/API/UI bütünlüğü ve gecikme; kalite kolonları daima NULL. |
