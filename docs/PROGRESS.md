# Faz 7 ilerleme günlüğü

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
