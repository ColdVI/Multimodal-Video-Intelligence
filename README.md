# Multimodal Video Intelligence

Doğal dil sorgularını yapısal telemetri filtreleriyle birleştirerek büyük video arşivlerinde **video kimliği + zaman aralığı** döndüren hibrit multimodal arama sistemi.

```text
"100 metrenin altında, insanların görüldüğü gece uçuşları"
        ↓
flight_0042   00:14:08–00:14:16   score=0.812
flight_0117   01:02:24–01:02:32   score=0.774
```

Video embedding'leri ingest sırasında bir kez üretilir. Arama sırasında videolar tekrar işlenmez; yalnız sorgu metni embed edilir.

> **Durum:** `implementation_complete_hardware_acceptance_pending`
>
> Son bağlayıcı kabul matrisi `d3fef0e` kodu için **28 PASS, 0 FAIL, 12 NOT RUN** sonucunu kaydetmiştir. Daha sonraki kanonik Docker/requirements düzeni henüz gerçek NVIDIA hedef host ve kurum verisiyle yeniden kabul edilmemiştir. Hedef GPU kabulü tamamlanmadan `fully_accepted_on_target_environment` durumu kullanılmamalıdır. Ayrıntılar: [FAZ11 final raporu](docs/reports/faz11/FINAL_REPORT.md) ve [Kurumsal GPU Kurulum ve İşletim Kılavuzu](docs/operations/GPU_DEPLOYMENT_GUIDE_TR.md).

## En kolay başlangıç

Kendi video/CSV datasetinizi hazırlamak, kolonları eşlemek, manifest üretmek ve uygun ortamda ingest başlatmak için:

**[VideoSearch Unified Runner notebook](notebooks/production/VideoSearch_Unified_Runner.ipynb)**

Notebook veri kaynağı olarak yerel klasör, ZIP, opsiyonel Google Drive, HTTPS/presigned URL veya küçük sentetik örnek dataset kullanabilir. Google Drive zorunlu değildir.

| Mod | Amaç | Docker |
|---|---|---|
| `prepare_dataset` | CSV profiling, mapping, manifest ve preflight | Gerekmez |
| `portable_embedding` | Qwen window embedding ve 2048/1024/512/256 export | Gerekmez |
| `docker_ingest` | Aynı hosttaki FAZ11 servisine `ingest --resume` | Gerekir |

Colab ile uzak kurum sunucusu aynı dosya sistemini paylaşmaz. Colab yolu kalıcı full-stack deployment değil, dataset/embedding artifact üretim yoludur.

---

## FAZ11 kurum kurulumu

### Ön koşullar

- NVIDIA Linux veya WSL2 Ubuntu
- çalışan NVIDIA sürücüsü
- Docker Engine / Docker Desktop ve Compose v2
- NVIDIA Container Toolkit
- Python 3.11+
- kurum video dosyaları ve varsa telemetry CSV'leri
- pinlenmiş Qwen3-VL-Embedding model bundle'ı

```bash
nvidia-smi
docker version
docker compose version
docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi
```

### 1. Repository ve ortam

```bash
git clone https://github.com/ColdVI/Multimodal-Video-Intelligence.git
cd Multimodal-Video-Intelligence

# Container build'lerinde kök Dockerfile, Python bağımlılıklarında kök
# requirements.txt kanonik kaynaktır. Compose profilleri aynı Dockerfile'ın
# api, ui, hybrid ve gpu target'larını kullanır.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cp .env.example .env
```

`.env` içinde bütün `CHANGE_ME` değerlerini ve yolları düzenleyin:

```dotenv
POSTGRES_PASSWORD=<guvenli-parola>
CLICKHOUSE_PASSWORD=<guvenli-parola>
DATA_ROOT=/absolute/path/to/institution-dataset
MODEL_BUNDLE_ROOT=/absolute/path/to/mvi-model-bundle

ENABLED_VECTOR_BACKENDS=clickhouse
DEFAULT_VECTOR_BACKEND=clickhouse
ENABLED_DIMENSIONS=512
FILTER_EXECUTION_MODE=pushdown
EMBEDDING_MODE=real
CUDA_IMAGE_TAG=12.1.1-runtime-ubuntu22.04
BIND_HOST=127.0.0.1
API_TOKEN=
```

Canonical kurum profili ClickHouse + 512d + native pushdown'dır. Qdrant, pgvector ve diğer boyutlar benchmark profiline aittir.

### 2. Dataset

```text
DATA_ROOT/
├── videos/
│   ├── flight_001.mp4
│   └── flight_002.mp4
├── telemetry/
│   ├── flight_001.csv
│   └── flight_002.csv
└── pairing/
    └── institution.csv
```

Docker, `DATA_ROOT` klasörünü container içinde `/workspace/data` olarak salt okunur mount eder.

```bash
cp datasets/example_institution.yaml datasets/kurum.yaml
```

Kurum CSV'sini bizim kolon adlarımıza dönüştürmek gerekmez. Manifest source kolonlarını canonical veya extra alanlara eşler:

```yaml
telemetry:
  format: generic_csv
  timestamp_column: FlightTime
  fields:
    altitude_m:
      source: RelAltitude
      unit: m
      reference: AGL
      type: continuous
    velocity_mps:
      source: GroundSpd
      unit: m/s
      kind: ground_speed
      type: continuous
    compass_heading:
      source: Heading
      unit: deg
      type: circular_deg
  extra:
    battery_v:
      source: BatteryVoltage
      unit: V
      type: continuous
```

Ayrıntı: [Dataset onboarding](docs/datasets/DATASET_ONBOARDING_GUIDE.md).

### 3. Model bundle

```bash
python scripts/prepare_model_bundle.py \
  --model-id Qwen/Qwen3-VL-Embedding-2B \
  --model-revision 9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda \
  --source-repo https://github.com/QwenLM/Qwen3-VL-Embedding.git \
  --source-commit 393e2978d27852b0d0230d6994f37f9c15bed73c \
  --bundle-root /absolute/path/to/mvi-model-bundle
```

### 4. Read-only preflight

```bash
python scripts/preflight.py \
  --dataset datasets/kurum.yaml \
  --env-file .env \
  --json-out artifacts/faz11/preflight.json
```

`status=pass` ve exit code `0` olmadan ingest başlatmayın.

### 5. Servisleri başlatın

```bash
docker compose \
  --env-file .env \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  up -d --build

docker compose --env-file .env -f docker-compose.yml -f docker-compose.gpu.yml ps
curl -fsS http://127.0.0.1:8000/health
```

- UI: <http://127.0.0.1:7860>
- API/OpenAPI: <http://127.0.0.1:8000/docs>
- Health: <http://127.0.0.1:8000/health>

### 6. Gerçek GPU smoke

20 GB kurum verisinin tamamı işlenmeden önce 1-2 gerçek MP4 veya 10 pencere ile GPU smoke çalıştırın:

```bash
docker compose \
  --env-file .env \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  exec -T api python scripts/gpu_smoke.py \
  --dataset /workspace/datasets/kurum.yaml \
  --data-root /workspace/data \
  --output /workspace/artifacts/faz11/gpu_smoke.json \
  --windows 10
```

Smoke çıktısında sentetik fallback bulunmamalı; GPU, model revision, embedding shape ve sonlu değer kontrolleri başarılı olmalıdır.

### 7. İlk ingest

```bash
docker compose \
  --env-file .env \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  exec -T api python -m app.ingestion.ingest \
  --dataset /workspace/datasets/kurum.yaml \
  --resume
```

`--resume` tamamlanmış chunk'ları tekrar decode veya embed etmez. Başarısız yeni run tamamen doğrulanmadan eski active run değişmez.

Generic ingest sırasında video ve segment metadata'sı PostgreSQL'deki run kapsamlı tablolara, 512 boyutlu vektörler ClickHouse'a yazılır. PostgreSQL segment sayıları ile ClickHouse vektör sayıları doğrulanmadan yeni run aktif edilmez. Arama sonuçlarındaki `video_id`, `segment_id`, `t_start`, `t_end`, dosya yolu ve telemetry alanları PostgreSQL'den hydrate edilir.

`EMBEDDING_MODE=real` altında video embedding'leri yalnız ingest sırasında üretilir. Daha önce tanımlanmamış her sorgu metni istek anında GPU'da embed edilir; production query cache'i okunmaz veya yazılmaz.

Kurum profili ClickHouse üzerinde `prefilter` stratejisini kullanır. Desteklenen diğer stratejiler ve backend seçenekleri için [Kurumsal GPU Kurulum ve İşletim Kılavuzu](docs/operations/GPU_DEPLOYMENT_GUIDE_TR.md) esas alınmalıdır.

### 8. Hedef ortam kabulü

```bash
python scripts/run_faz11_acceptance.py \
  --dataset datasets/kurum.yaml \
  --env-file .env \
  --live \
  --output artifacts/faz11/target_acceptance.json
```

---

## Normal kullanıcı

1. UI'ı açar.
2. Dataset seçer.
3. Doğal dil sorgusu yazar.
4. Gerekirse filtreleri seçer.
5. Sonuç video zaman aralığını açar.

Normal kullanıcının SQL yazması, embedding üretmesi veya ClickHouse/Qwen yapılandırması gerekmez.

---

## Benchmark profili

```bash
docker compose \
  --env-file .env \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  -f docker-compose.benchmark.yml \
  up -d --build
```

Bu profil normal deployment'ın zorunlu parçası değildir.

---

## Güvenlik ve işletim

- Dataset ve model bundle salt okunur mount edilir.
- `DATA_ROOT` dışına çıkan path'ler reddedilir.
- Public bind için güçlü `API_TOKEN` gerekir.
- DB portları canonical profilde hosta açılmaz.
- Normal kapatmada `down -v` kullanmayın.

```bash
docker compose --env-file .env -f docker-compose.yml -f docker-compose.gpu.yml ps
docker compose --env-file .env -f docker-compose.yml -f docker-compose.gpu.yml logs -f api
docker compose --env-file .env -f docker-compose.yml -f docker-compose.gpu.yml logs -f ui
docker compose --env-file .env -f docker-compose.yml -f docker-compose.gpu.yml down
```

---

## Bilinen sınırlar

- GT 1030 hedef kurum kabulü için temsilî değildir.
- Colab kalıcı PostgreSQL + ClickHouse + API + UI production ortamı değildir.
- Küçük dataset mekanik smoke için yeterli olabilir; retrieval kalitesi kararı için yeterli değildir.
- Dimension seçimi gerçek kurum golden set'iyle doğrulanmalıdır.
- `fully_accepted_on_target_environment` yalnız gerçek hedef host kabulü geçince kullanılmalıdır.

---

## Dokümantasyon

- [Tüm dokümanlar](docs/README.md)
- [Operatör hızlı başlangıç](docs/getting-started/OPERATOR_QUICKSTART.md)
- [Son kullanıcı kılavuzu](docs/getting-started/END_USER_GUIDE.md)
- [Dataset onboarding](docs/datasets/DATASET_ONBOARDING_GUIDE.md)
- [Güncel mimari](docs/architecture/CURRENT_SYSTEM.md)
- [Deployment](docs/operations/DEPLOYMENT.md)
- [Operasyonlar](docs/operations/OPERATIONS.md)
- [Hedef ortam kabulü](docs/operations/TARGET_ENVIRONMENT_ACCEPTANCE.md)
- [FAZ11 final raporu](docs/reports/faz11/FINAL_REPORT.md)
- [Notebook rehberi](notebooks/README.md)

Coding agent girişi: [AGENTS.md](AGENTS.md).
