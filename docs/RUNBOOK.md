# Faz 7 runbook

## Ayağa kaldırma

```bash
cp .env.faz7.example .env.faz7
docker compose -f docker-compose.faz7.yml up -d --build
docker compose -f docker-compose.faz7.yml exec -T api python -m app.ingestion.load_dataset --dataset auair
```

API: <http://localhost:8000/docs>  
UI: <http://localhost:7860>

Sağlık ve örnek sorgu:

```bash
curl -sf http://localhost:8000/health
curl -sf -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"kalabalık trafik","dataset_id":"auair","backend":"clickhouse","strategy":"prefilter","dimension":512,"top_k":10,"repeats":10}'
```

UI ekran görüntüsü: UI açıldıktan ve AU-AIR yüklendikten sonra tarayıcıyı `http://localhost:7860` adresinde 1440×1000 boyuta getir; kırmızı sentetik banner ile gecikme/diagnostics/sonuç panellerinin tamamını `artifacts/ui_smoke.png` olarak kaydet.

## Cached gerçek embedding'e geçiş

Colab'ı aç → GPU seç → Tümünü çalıştır → ZIP'i indir → `artifacts/embeddings/` altına aç → `.env.faz7`'de `EMBEDDING_MODE=cached` → `docker compose -f docker-compose.faz7.yml restart api ui` → UI banner yeşile döner.

`cached` mod model yüklemez. Serbest metin sorguları, Colab'ın ürettiği `artifacts/embeddings/query_embeddings.json` içinde bulunmalıdır; bilinmeyen sorgu sentetik vektöre sessizce düşmez.

## GPU/real modu

NVIDIA Container Toolkit bulunan makinede:

```bash
docker compose -f docker-compose.faz7.yml -f docker-compose.gpu.yml up -d --build
```

## Teardown

```bash
docker compose -f docker-compose.faz7.yml down
# Verileri de silmek açıkça isteniyorsa ayrıca: docker compose -f docker-compose.faz7.yml down -v
```

