# FAZ 11 operatör hızlı başlangıç

Bu, [USER_GUIDE.md](USER_GUIDE.md)'un adım-adım komut özetidir — açıklama
için oraya, sorun giderme için [OPERATIONS.md](OPERATIONS.md)'a bakın.
NVIDIA Linux host, Docker Engine + Compose v2 ve NVIDIA Container Toolkit
kurulu olduğu varsayılır.

```bash
# 1) Repo ve env
git clone <repo-url>
cd Multimodal-Video-Intelligence
cp .env.example .env
# .env içindeki her CHANGE_ME_* değeri, MODEL_BUNDLE_ROOT ve gerekiyorsa
# BIND_HOST/API_TOKEN'ı düzenleyin.

# 2) Host ön kontrol
nvidia-smi
docker run --rm --gpus all nvidia/cuda:${CUDA_IMAGE_TAG:-12.1.1-runtime-ubuntu22.04} nvidia-smi

# 3) Model bundle (ayrı, internete çıkabilen bir makinede hazırlanıp kopyalanabilir)
python scripts/prepare_model_bundle.py \
  --model-id Qwen/Qwen3-VL-Embedding-2B \
  --model-revision 9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda \
  --source-repo https://github.com/QwenLM/Qwen3-VL-Embedding.git \
  --source-commit 393e2978d27852b0d0230d6994f37f9c15bed73c \
  --bundle-root "$MODEL_BUNDLE_ROOT"

# 4) Dataset manifesti
cp datasets/example_uav.yaml datasets/kurum.yaml
# videos_glob/telemetry_glob/time_alignment/telemetry.fields'ı düzenleyin.
# Video ve telemetri dosyalarını .env'deki DATA_ROOT altına yerleştirin.

# 5) Preflight (status=pass olmadan devam etmeyin)
python scripts/preflight.py --dataset datasets/kurum.yaml --env-file .env \
  --json-out artifacts/faz11/preflight.json

# 6) Sistemi başlat
docker compose --env-file .env -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
docker compose --env-file .env ps
curl -fsS http://${BIND_HOST:-127.0.0.1}:8000/health

# 7) İlk ingest (run_id'yi kaydedin)
docker compose --env-file .env exec api \
  python -m app.ingestion.ingest --dataset /workspace/datasets/kurum.yaml --resume

# 8) UI
# tarayıcıda http://<BIND_HOST>:7860

# 9) Kabul kanıtı (opsiyonel ama önerilir)
python scripts/run_faz11_acceptance.py --dataset datasets/kurum.yaml --env-file .env \
  --output artifacts/faz11/target_acceptance.json
```

## Canonical (kurum) varsayılan ayarlar

```env
ENABLED_VECTOR_BACKENDS=clickhouse
ENABLED_DIMENSIONS=512
FILTER_EXECUTION_MODE=pushdown
```

Bunlar `.env.example`'ın zaten varsayılanıdır — değiştirmeyin. Benchmark
karşılaştırması gerekiyorsa **ayrı** bir compose override kullanın:

```bash
docker compose --env-file .env \
  -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.benchmark.yml \
  up -d --build
```

bu Qdrant ve ek boyutları (2048/1024/512/256) yalnız o override aktifken
açar; normal deployment'ı etkilemez.

## Bir sonraki adım

- Video/telemetri eklemeye devam: [DATASET_ONBOARDING_GUIDE.md](DATASET_ONBOARDING_GUIDE.md)
- Günlük işletim, resume, migration, GC, sorun giderme: [OPERATIONS.md](OPERATIONS.md)
- Son kullanıcıları yönlendirmek için: [END_USER_GUIDE.md](END_USER_GUIDE.md)
