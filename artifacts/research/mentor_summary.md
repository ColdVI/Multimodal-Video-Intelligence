# Mentor özeti — Faz 7

Faz 7, araştırma notebook'larından bağımsız `service/` ürünü ekledi: FastAPI, Gradio, AU-AIR ingestion, pgvector/ClickHouse/Qdrant adapter'ları, exact/ANN/filter stratejileri, MRL 2048/1024/512/256 ve sentetik/cached/real embedding router'ı. Mevcut `docker-compose.yml`, `schema.sql` ve `config.yaml` değiştirilmedi.

## Gerçek veri veya gerçek yürütmeyle doğrulananlar

| Kanıt | Sonuç | Kapsam |
|---|---:|---|
| Hazır AU-AIR parquet'i | 1.866 segment + 1.866 telemetri | Önceki Faz 6 gerçek çıktısı; irtifa metre sözleşmesi loader'da savunmacı doğrulanır |
| Service saf-mantık testleri | 20/20 geçti | 200 öğelik corpus, tüm zorunlu strateji sözleşmeleri, exact referans, negatif kontrol, `top_k=200` |
| UI HTTP/render smoke | 200 + gerçek PNG | Banner, arama, diagnostics, CSV ve Karşılaştır sekmesi; API kapalı olduğundan veri alanları boş |
| Compose config | geçti | `docker-compose.faz7.yml` geçerli |

Canlı üç-DB ingest/search ölçümü bu oturumda **yoktur**: host Docker daemon kapalıydı. Bu durum `docs/BLOCKERS.md` ve verify çıktısında açıkça kayıtlıdır.

## Sentetik embedding ile sistem/gecikme çıktıları

| Artifact | Mod | Satır | Gecikme/kalite durumu |
|---|---|---:|---|
| `vector_database_results.csv` | `synthetic` | 150 | API çalışmadığı için latency NULL; kalite alanları zorunlu olarak NULL; her satır `benchmark_status=blocked` |
| UI/API tasarımı | `synthetic` varsayılan | — | Kırmızı banner ve API `embedding_mode`; semantik kalite iddiası yok |

Sonuç: ürün kodu ve saf-mantık kapıları hazırdır; canlı kabul için Docker Engine açıldıktan sonra `bash scripts/verify_faz7.sh` çalıştırılmalıdır. Gerçek Qwen üretimi `notebooks/07_colab_embedding_production.ipynb` ile checkpoint/resume biçiminde hazırlanmıştır.
