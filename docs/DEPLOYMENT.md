# Faz 11 kurum deployment rehberi

Bu rehber tek makinede NVIDIA Linux + Docker Compose kurulumunu tarif eder.
Repo NVIDIA driver, Docker Engine, Compose v2 veya NVIDIA Container Toolkit
kurmaz. Kurum bunları kendi güvenlik/yama politikasıyla provision eder.

## 1. Host hazırlığı

Önce host ve container GPU görünürlüğünü doğrulayın:

```bash
nvidia-smi
docker version
docker compose version
docker run --rm --gpus all nvidia/cuda:${CUDA_IMAGE_TAG} nvidia-smi
```

Kurum registry/proxy, firewall, disk mount, Docker group/rootless ve log
retention kararlarını deployment öncesi tamamlamalıdır. `DATA_ROOT` yalnız
okunur mount edilir; `ARTIFACTS_ROOT` rapor ve media cache için yazılabilir
olmalıdır.

## 2. Repo ve kurum ayarları

```bash
git clone <repo-url>
cd Multimodal-Video-Intelligence
cp .env.example .env
cp datasets/example_uav.yaml datasets/kurum.yaml
```

`.env` içinde bütün `CHANGE_ME` değerlerini değiştirin:

- `DATA_ROOT`, `MODEL_BUNDLE_ROOT`, PostgreSQL/ClickHouse parolaları;
- `CUDA_IMAGE_TAG`, `ENABLED_VECTOR_BACKENDS`, `ENABLED_DIMENSIONS`;
- `BIND_HOST` ve gerekiyorsa `API_TOKEN`;
- decode/embed/write batch değerleri ve opsiyonel resource limitleri.

`datasets/kurum.yaml` yalnız `DATA_ROOT` altındaki relative glob/pairing
yollarını taşır. Clock/anchor/offset ve canonical telemetry mapping ayrıntıları
[DATASET_MANIFEST.md](DATASET_MANIFEST.md) içindedir. Proprietary binary
telemetry için yalnız canonical record üreten bir adapter eklenir.

## 3. Model bundle

İnternete çıkabilen kontrollü hazırlık makinesinde:

```bash
python scripts/prepare_model_bundle.py \
  --model-id Qwen/Qwen3-VL-Embedding-2B \
  --model-revision 9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda \
  --source-repo https://github.com/QwenLM/Qwen3-VL-Embedding.git \
  --source-commit 393e2978d27852b0d0230d6994f37f9c15bed73c \
  --bundle-root /kurum/model-bundle
```

Bundle hedef hosta kopyalandıktan sonra manifest hash'leri yeniden doğrulanır.
Docker build model/weight indirmez; base image ve Python wheel'leri için tam
air-gap registry/wheelhouse hazırlığı kurum sorumluluğudur. Ayrıntı:
[MODEL_BUNDLE.md](MODEL_BUNDLE.md).

## 4. Preflight kapısı

```bash
python scripts/preflight.py \
  --dataset datasets/kurum.yaml \
  --env-file .env \
  --json-out artifacts/faz11/preflight.json
```

Exit `0` ve artifact `status=pass` olmadan ingest başlatmayın. Exit kodları:
`2` config, `3` data/manifest, `4` GPU/runtime, `5` model bundle, `6`
disk/resources. Preflight DB/data yazmaz.

## 5. Servisleri başlatma

Kurum profili PostgreSQL + ClickHouse + API + UI açar; DB portlarını hosta
publish etmez:

```bash
docker compose --env-file .env \
  -f docker-compose.yml -f docker-compose.gpu.yml \
  up -d --build
docker compose --env-file .env ps
curl -fsS http://${BIND_HOST:-127.0.0.1}:8000/health
```

Benchmark gerektiğinde yalnız kontrollü ortamda Qdrant ve dört boyutu ekleyin:

```bash
docker compose --env-file .env \
  -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.benchmark.yml \
  up -d --build
```

`docker-compose.debug.yml` DB portlarını loopback'e publish eder; üretim
başlatma komutuna eklenmez.

## 6. Ingest ve UI

```bash
docker compose --env-file .env exec api python -m app.ingestion.ingest \
  --dataset /workspace/datasets/kurum.yaml --resume
```

CLI başlangıç/report çıktısındaki `run_id` saklanır. Tamamlanıp bütün count/hash
kapıları geçmeden active pointer değişmez. UI varsayılan olarak
`http://127.0.0.1:7860` adresindedir. `API_TOKEN` doluysa UI aynı env değerini
Bearer header olarak kullanır; medya oynatma için API kısa ömürlü, path/run
scope'lu imzalı URL üretir. Token log veya response içine yazılmaz.

## 7. Kaynak limitleri

`API_MEMORY_LIMIT` ve `UI_MEMORY_LIMIT`, Compose memory limit formatını (`8g`
gibi) kabul eder; boş değer limitsiz davranışı korur. `CLICKHOUSE_MAX_MEMORY_BYTES`
server memory byte limitidir; boş değer ClickHouse otomatik davranışıdır.
`MEDIA_CACHE_MAX_GB` ve `MEDIA_CACHE_RETENTION_HOURS` cache'i sınırlar.

Uydurma evrensel limit yoktur. Başlangıç değeri preflight'ın estimated segment
ve vector bytes çıktısı, gerçek GPU smoke peak VRAM'i ve pilot ingest RSS/DB
ölçümlerinden türetilmelidir. Deployment sonrası işlemler ve rollback için
[OPERATIONS.md](OPERATIONS.md) kullanılır.
