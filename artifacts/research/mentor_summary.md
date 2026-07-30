# Faz 7 mentor özeti

Sistem `docker compose -f docker-compose.faz7.yml up -d --build` ile çalışır: UI `:7860`, API `:8000`. Varsayılan mod **synthetic**; bu modda sıralama anlamsızdır ve kalite iddiası yoktur.

## (a) GERÇEK ölçümler — sentetik Faz 7 sıralamasıyla karıştırılmaz

| Kanıt | Ortam/kapsam | Sonuç |
|---|---|---|
| AU-AIR parquet | Gerçek doğrulanmış veri | 1.866 segment; irtifa 2,838–30,364 m; mm→m dönüşümü yapılmış |
| Qwen3-VL-Embedding-2B | Colab NVIDIA L4, 10 pencere (`artifacts/colab_gpu_bench_l4.json`) | video p50 2,557 s; text p50 0,045 s |
| Qwen3-VL-Embedding-2B | Colab Tesla T4, 10 pencere (`artifacts/colab_gpu_bench_t4.json`) | video p50 349,447 s; text p50 0,117 s |

## (b) SENTETİK embedding — yalnız sistem/gecikme

| Kanıt | Sonuç |
|---|---|
| Docker health | pgvector/pg16 + ClickHouse 25.8 + Qdrant 1.12.4 + API + UI healthy |
| Store eşitliği | AU-AIR × 4 boyut; her backend/boyutta 1.866 satır |
| Canlı örnek | ClickHouse prefilter 512d, 3 tekrar: p50 160,175 ms; 10/10 sonuç |
| Canlı örnek | Qdrant ANN 512d: 62,444 ms; pgvector exact 512d: 37,354 ms |
| Negatif kontrol | `altitude_m=[-100,-50]`: 0 sonuç, `underfilled=true`, kalite NULL |
| Matris smoke | 150 satır, hata 0; p50 medyan 56,235 ms; exact stratejilerde recall@10=1,00; tüm ground-truth kalite kolonları NULL |

Üretim notebook'u `notebooks/07_colab_embedding_production.ipynb`; eksikler `docs/BLOCKERS.md`; tam işletim `docs/RUNBOOK.md`.
