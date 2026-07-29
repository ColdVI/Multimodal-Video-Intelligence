# Faz 7 işletim rehberi

## Sıfırdan başlatma

Önkoşullar: çalışan Docker Desktop/Engine, Compose v2, en az 12 GB boş disk. Varsayılan `synthetic` mod GPU ve veri görüntüsü gerektirmez; AU-AIR'in repodaki doğrulanmış parquet'lerini kullanır.

```bash
cp .env.faz7.example .env.faz7
docker compose -f docker-compose.faz7.yml up -d --build
docker compose -f docker-compose.faz7.yml exec -T api python -m app.ingestion.load_dataset --dataset auair
curl -sf http://localhost:8000/health
curl -sf http://localhost:8000/stats
```

UI `http://localhost:7860`, OpenAPI `http://localhost:8000/docs` adresindedir. Tam kabul akışı `bash scripts/verify_faz7.sh` ile çalışır ve kanıtı `artifacts/verify_faz7_output.txt` dosyasına yazar.

Örnek sorgu:

```bash
curl -sf -X POST http://localhost:8000/search -H 'Content-Type: application/json' \
  -d '{"query":"kalabalık trafik","dataset_id":"auair","backend":"clickhouse","strategy":"prefilter","dimension":512,"top_k":10,"repeats":10}'
```

UI'da dataset seçimi facet ve telemetri sınırlarını `/stats` ve `/facets` üzerinden gerçek veriden yeniler. `Karşılaştır` sekmesi aynı sorguyu backend×strateji×boyut matrisinde 10 tekrarla koşturur. Görsel kanıt için UI'ı açın, AU-AIR ve ClickHouse/prefilter/512 seçin, Search'e basın ve tam sayfa ekran görüntüsü alın.

## Gerçek cached embedding'e geçiş

1. `notebooks/07_colab_embedding_production.ipynb` dosyasını Colab'da açın.
2. Runtime → Change runtime type → GPU seçin ve tüm hücreleri çalıştırın.
3. Oluşan `embeddings_<dataset>.zip` dosyasını indirin ve içeriğini `artifacts/embeddings/` altına açın.
4. `.env.faz7` içinde `EMBEDDING_MODE=cached` yapın.
5. `docker compose -f docker-compose.faz7.yml restart api ui` çalıştırın.

Banner yeşile döner. Cached mod yalnız notebook'un ürettiği `{dataset}_queries.npz` içindeki gerçek Qwen metin sorgularını kabul eder; yeni ve serbest bir metin için `real` mod gerekir. Bu kural gerçek item vektörünü sentetik query ile karşılaştırıp yanıltıcı sonuç üretmeyi engeller.

## Real/GPU modu

Qwen repo'sunu proje köküne klonlayın, ağırlıkları `weights/` altında hazırlayın ve NVIDIA Container Toolkit bulunan hostta:

```bash
docker compose -f docker-compose.faz7.yml -f docker-compose.gpu.yml up -d --build api ui
```

T4/Turing için `TORCH_DTYPE=float16` ve SDPA; Ampere+ için `bfloat16` ve Flash Attention 2 kullanın. Real bağımlılıklar GPU imajına ayrıca kurulmalıdır; temel sentetik imaj Torch/Qwen ağırlığı taşımaz.

## Benchmark ve kapatma

```bash
docker compose -f docker-compose.faz7.yml exec -T api \
  python -m app.bench.runner --level L2 --out /workspace/artifacts/research/vector_database_results.csv
docker compose -f docker-compose.faz7.yml down
```

Volume'ları da silmek isterseniz bunun veri kaybı yaratacağını bilerek ayrıca `down -v` kullanın.

Sorun giderme: `/health` degraded ise `docker compose -f docker-compose.faz7.yml ps` ve `logs api ui pg ch qdrant` çıktısını inceleyin. Qdrant `indexed_vectors_count=0` küçük koleksiyonun brute-force çalıştığını gösterir; latency raporunda ANN diye yorumlanmamalıdır.
