# Faz 7 runbook

## Faz 8 - Gercek veriye gecis (zorunlu sira)

On kosul: Qwen kaynak kodu ve model snapshot'i .env.faz7 icindeki
QWEN_REPO_HOST_PATH / QWEN_MODEL_HOST_PATH yollarinda hazir olmalidir. Bu
hazirlik 10 dakikayi asiyorsa sistem testlerini bloke etmez; A1 eksik kalir.

1. Colab'da notebooks/07_colab_embedding_production.ipynb calistirilir.
   Yalniz CapERA test split uretilir: 1391 item, 6955 caption query;
   provenance unknown.
2. Indirilen capera_embeddings_faz8.zip dosyasi artifacts/embeddings/
   altina acilir. ZIP'i acmak tek basina A1 hazirligi degildir.
3. .env.faz7 icinde gecici olarak EMBEDDING_MODE=cached ayarlanir ve cached
   ingest gercekten calistirilir:

   docker compose -f docker-compose.faz7.yml up -d --build
   docker compose -f docker-compose.faz7.yml exec -T api python -m app.ingestion.load_dataset --dataset capera

4. GET /stats ile CapERA segments=1391; 2048/1024/512/256 boyutlarinin her
   birinde pgvector, ClickHouse ve Qdrant sayilarinin ayri ayri 1391 oldugu
   ve GT sayisinin tam 6955 oldugu dogrulanir.
5. API/UI hybrid_text imajiyla yeniden baslatilir; ardindan cold model load,
   cold query ve warm query ayri olculur:

   docker compose -f docker-compose.faz7.yml -f docker-compose.hybrid.yml up -d --build api ui
   docker compose -f docker-compose.faz7.yml -f docker-compose.hybrid.yml exec -T api python -m app.embedding.text_cpu

   Karar sadece warm_p50_ms uzerindendir. Esik asilirsa artifact
   selected_mode=cached_only yazar; .env.faz7 cached moda alinir ve servis
   yeniden baslatilir. Hicbir durumda sentetik fallback yoktur.
6. Quality readiness strict calistirilir:

   .venv/Scripts/python.exe scripts/readiness_check.py --profile quality --json --strict

7. Yalniz quality profili hazirsa T8 calistirilir:

   PYTHONPATH=service .venv/Scripts/python.exe -m app.bench.matrix --suite T8 --out artifacts/research/test_matrix_T8.csv

Sistem profili ayri bir kapidir: --profile system --strict. CapERA/A1
eksikligi T1-T7'nin calismasini engellemez.

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

Yukarıdaki zorunlu sırayı kullan. ZIP'i açtıktan sonra cached ingest ve
`1391 x 4 boyut x 3 backend` ile `6955` GT doğrulaması yapılmadan A1 hazır
sayılmaz.

`cached` mod model yüklemez. `query_embeddings.json` yalnız sabit demo
sorgularını tutar; 6955 quality sorgusu NPY + Parquet dosyalarındadır.
Bilinmeyen sorgu sentetik vektöre sessizce düşmez.

## GPU/real modu

NVIDIA Container Toolkit bulunan makinede:

```bash
cp .env.example .env
# Secret'ları ve MODEL_BUNDLE_ROOT'u gerçek değerlerle değiştir.
python scripts/prepare_model_bundle.py \
  --model-id Qwen/Qwen3-VL-Embedding-2B \
  --model-revision 9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda \
  --source-repo https://github.com/QwenLM/Qwen3-VL-Embedding.git \
  --source-commit 393e2978d27852b0d0230d6994f37f9c15bed73c \
  --bundle-root /opt/mvi-model-bundle
docker compose --env-file .env -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
python scripts/gpu_smoke.py --dataset datasets/kurum.yaml --data-root /kurum/data \
  --output artifacts/faz11/gpu_smoke.json --windows 10
```

Bundle manifest ve tam offline kapsamı için `docs/MODEL_BUNDLE.md` esas alınır.

## Faz 11 generic manifest ingest

Preflight başarılı olmadan generic ingest başlamaz. Kurum container'ında:

```bash
python -m app.preflight --dataset /workspace/datasets/kurum.yaml
python -m app.ingestion.ingest --dataset /workspace/datasets/kurum.yaml
```

Kesilen veya failed olmuş aynı manifest-hash run'ını chunk ledger üzerinden
devam ettirmek için:

```bash
python -m app.ingestion.ingest --dataset /workspace/datasets/kurum.yaml --resume
```

`--resume`, tamamlanmış chunk'ları decode etmez; incomplete chunk için yalnız
aynı inactive run/chunk satırlarını temizleyip yeniden yazar. Run raporu
`artifacts/faz11/ingest_runs/<run_id>/report.json`, satır/video hataları aynı
dizindeki `errors.jsonl` dosyasındadır. Legacy yollar korunur:

```bash
python -m app.ingestion.ingest --dataset-id auair
python -m app.ingestion.load_dataset --dataset auair
```

## Teardown

```bash
docker compose -f docker-compose.faz7.yml down
```
