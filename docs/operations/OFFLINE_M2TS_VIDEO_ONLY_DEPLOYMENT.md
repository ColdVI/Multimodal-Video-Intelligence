# Air-Gapped M2TS Video-Only Dağıtım Yönergesi

## 1. Kapsam ve üretim mimarisi

Bu yönerge, yalnızca M2TS video dosyalarının internetsiz bir NVIDIA Linux/amd64 bilgisayarda işlenmesi içindir. Üretim profili dört servisten oluşur: PostgreSQL (`pg`), ClickHouse (`ch`), GPU kullanan API (`api`) ve UI (`ui`). Qdrant, Milvus ve benchmark servisleri bu profile dahil değildir.

Veri akışı şöyledir:

```text
M2TS -> 8 saniyelik temporal window -> Qwen3-VL GPU video embedding
     -> PostgreSQL run/video/segment metadata
     -> ClickHouse 512d vector
     -> active run
     -> yeni text query -> Qwen GPU text embedding -> ClickHouse search
     -> video_id + segment_id + t_start + t_end + score
```

PostgreSQL; `ingest_runs`, `ingest_chunks`, `dataset_active_runs`, `run_videos`, `run_segments`, `run_segment_metadata`, sonuç hydration, resume ve active-run doğrulaması için zorunludur. Telemetry bulunmaması PostgreSQL gereksinimini ortadan kaldırmaz. ClickHouse 512 boyutlu embedding saklama, arama ve run satır sayısı doğrulaması için zorunludur.

## 2. Ön koşullar

İnternetsiz hedefte aşağıdakiler önceden kurulu ve çalışır olmalıdır:

- Linux/amd64 Docker Engine ve Docker Compose v2;
- uyumlu NVIDIA sürücüsü;
- NVIDIA Container Toolkit;
- doğrulama betiği için yalnızca Python 3 standart kütüphanesi;
- yeterli disk alanı ve GPU belleği.

Image bundle Docker, sürücü veya NVIDIA Container Toolkit kurmaz. Hedef bilgisayarda `pip install`, `apt-get`, `docker build` veya `docker pull` çalıştırılmaz. Eksik bir image varsa işlem açık hata ile durmalıdır.

## 3. İnternetli staging bilgisayarında bundle üretimi

Önce pinlenmiş model/source bundle'ını bir defa hazırlayın veya mevcut doğrulanmış bundle'ı kullanın:

```bash
python scripts/prepare_model_bundle.py \
  --bundle-root /staging/mvi-model-bundle
```

Mevcut bundle kullanılıyorsa exporter ağırlıkları tekrar indirmez; manifest ve dosya hash zincirini doğrular. Linux/amd64 image bundle'ını üretmeden önce tahmini planı yazdırın; bu komut Docker build/pull/save veya model indirme çalıştırmaz:

```bash
python scripts/export_offline_bundle.py \
  --model-bundle /staging/mvi-model-bundle \
  --output-dir /staging/offline_bundle \
  --target-platform linux/amd64 \
  --estimate-only
```

Beklenen image seti yalnız `mvi-app-gpu:<sha>`, `pgvector/pgvector:pg16` ve `clickhouse/clickhouse-server:25.8` olmalıdır. Muhafazakâr başlangıç bütçesi download için 7–11 GB, build cache için 15–25 GB, model için 4.5–6.5 GB, transfer için 13–20 GB ve boş disk için en az 60 GB'dır. Bunlar tahmindir; gerçek TAR boyutu yalnız `docker save` sonrasında ölçülür.

Exporter başlamadan `docker version`, `docker compose version`, `docker info`, `git rev-parse HEAD` ve `git status --short` kontrollerini çalıştırır. Dirty checkout varsayılan olarak reddedilir; SHA ile kaynak içeriğinin ayrışmaması esastır. Staging GPU zorunlu değildir; `--gpu-smoke` verilmezse manifestte `gpu_runtime_smoke=NOT_RUN` kalır.

Linux/amd64 image bundle'ını üretin:

```bash
python scripts/export_offline_bundle.py \
  --model-bundle /staging/mvi-model-bundle \
  --output-dir /staging/offline_bundle \
  --target-platform linux/amd64
```

Exporter, mevcut Git SHA ile yalnız `mvi-app-gpu:<sha>` application image'ını `gpu` target'ından build eder; `pgvector/pgvector:pg16` ile `clickhouse/clickhouse-server:25.8` image'larını staging hostta pull eder; toplam üç image'ı `images/mvi-images-linux-amd64.tar` içine kaydeder. API ve UI ayrı container'lar olarak aynı application image ID'sini kullanır. UI `python3 -m ui.app` command'ıyla başlar ve GPU almaz; yalnız API `gpus: all` ister. `bundle_manifest.json` image ID, digest, OS/mimari, gerçek image size, TAR boyutu/hash'i ve model pinlerini içerir. `SHA256SUMS` tüm taşıma dosyalarını kapsar. Docker save manifestindeki bütün layer dosyaları exporter tarafından kontrol edilir.

Bundle'a gerçek parola, token veya kurum videosu eklemeyin. `.env.offline.example` yalnız placeholder içerir.

## 4. Hedef dizin yerleşimi

Beklenen çalışma düzeni:

```text
folder/
├── videos/
│   ├── *.m2ts
│   └── alt_klasorler/**/*.M2TS
└── multi-model/
    ├── docker-compose.offline-gpu.yml
    ├── .env.offline
    ├── datasets/video_only_m2ts.yaml
    ├── artifacts/
    └── offline/
        ├── images/mvi-images-linux-amd64.tar
        └── model-bundle/
```

Transfer bundle'ındaki `model-bundle/` dizinini `multi-model/offline/model-bundle/`, TAR dosyasını `multi-model/offline/images/` altına kopyalayın. Compose, dataset manifesti ve doğrulama betiğini bundle'dan repository köküne kopyalayabilirsiniz. Kaynak dosyaları koruyun.

`.env.offline.example` dosyasını `.env.offline` olarak kopyalayın; bütün `CHANGE_ME` değerlerini değiştirin:

```env
DATA_ROOT=../videos
MODEL_BUNDLE_ROOT=./offline/model-bundle
ARTIFACTS_ROOT=./artifacts
```

Relative bind yolları Compose dosyasının proje dizinine göre çözülür. Production için özellikle farklı mount/drive düzenlerinde absolute Linux yolları önerilir. Windows üzerinde Docker Desktop kullanılıyorsa drive sharing açık olmalıdır. WSL içinde Windows yolu (`C:\...`) yerine `/mnt/c/...` biçimi kullanılmalıdır. Dataset manifestindeki `**/*.m2ts` glob'u container içindeki `/workspace/data` köküne göredir; host absolute yolu değildir.

Yolu başlamadan doğrulayın:

```bash
docker compose --env-file .env.offline \
  -f docker-compose.offline-gpu.yml config --format json
```

API volume kaynağının gerçek `folder/videos` dizini, hedefinin `/workspace/data` ve `read_only` değerinin `true` olduğunu kontrol edin. Model mount'u da read-only olmalıdır. PostgreSQL ve ClickHouse portları hosta publish edilmez.

## 5. Bundle doğrulama, image yükleme ve offline başlatma

Bundle henüz transfer dizinindeyken checksum/model/image sözleşmesini doğrulayıp image'ları yükleyin:

```bash
python scripts/verify_offline_bundle.py /transfer/offline_bundle \
  --env-file /absolute/path/to/multi-model/.env.offline \
  --json-out /absolute/path/to/multi-model/artifacts/offline_bundle_verification.json
```

Varsayılan davranış `docker load --input ...` çalıştırır ve üç tag'in local store'da doğru ID ve linux/amd64 platformuyla bulunduğunu doğrular. `--skip-load` yalnız statik inceleme içindir. Doğrulayıcı Docker Engine, Compose, NVIDIA sürücüsü ve `nvidia` container runtime'ı yoksa durur; internetten image çekmez.

Çalışma dizini kurulmuşsa stack'i doğrudan registry erişimi olmadan başlatın:

```bash
python scripts/verify_offline_bundle.py /transfer/offline_bundle \
  --env-file /absolute/path/to/multi-model/.env.offline

docker compose --env-file .env.offline \
  -f docker-compose.offline-gpu.yml \
  up -d --no-build --pull never
```

`--pull never` ve her servisteki `pull_policy: never`, eksik image durumunda pull yerine fail edilmesini sağlar. Sağlığı kontrol edin:

```bash
docker compose --env-file .env.offline -f docker-compose.offline-gpu.yml ps
docker compose --env-file .env.offline -f docker-compose.offline-gpu.yml exec -T api \
  python3 -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
curl -fsS -H "Authorization: Bearer ${API_TOKEN}" http://127.0.0.1:8000/health
```

## 6. M2TS preflight ve küçük GPU smoke

Tam ingest öncesinde video-only preflight çalıştırın:

```bash
docker compose --env-file .env.offline -f docker-compose.offline-gpu.yml exec -T api \
  python3 -m app.preflight \
  --dataset /workspace/datasets/video_only_m2ts.yaml \
  --data-root /workspace/data \
  --json-out /workspace/artifacts/m2ts_preflight.json
```

`status=pass`, `telemetry_enabled=false`, beklenen `video_count`, `container=mpegts`, codec, raw start timestamp, normalized origin ve monotonicity bilgisini doğrulayın. Unsupported codec/container veya timestamp discontinuity hata olarak ele alınmalıdır; M2TS dosyaları varsayılan olarak MP4'e transcode edilmez.

En az 10 window üzerinde gerçek GPU embedding smoke çalıştırın:

```bash
docker compose --env-file .env.offline -f docker-compose.offline-gpu.yml exec -T api \
  python3 scripts/gpu_smoke.py \
  --dataset /workspace/datasets/video_only_m2ts.yaml \
  --data-root /workspace/data \
  --output /workspace/artifacts/m2ts_gpu_smoke.json \
  --windows 10
```

Bu iki adım geçmeden 20 GB ingest başlatmayın.

## 7. Ingest, PostgreSQL ve ClickHouse doğrulaması

```bash
docker compose --env-file .env.offline -f docker-compose.offline-gpu.yml exec -T api \
  python3 -m app.ingestion.ingest \
  --dataset /workspace/datasets/video_only_m2ts.yaml \
  --data-root /workspace/data --resume
```

Çıktıdaki `run_id` değerini kaydedin. Aşağıdaki sayılar birbirleriyle ve ingest raporuyla uyumlu olmalıdır:

```bash
docker compose --env-file .env.offline -f docker-compose.offline-gpu.yml exec -T pg \
  psql -U mvi -d uav_search -c \
  "SELECT r.run_id,r.status,count(DISTINCT v.video_id) AS videos,count(DISTINCT s.segment_id) AS segments FROM ingest_runs r LEFT JOIN run_videos v USING(run_id) LEFT JOIN run_segments s USING(run_id) WHERE r.dataset_id='institution_m2ts_video_only' GROUP BY r.run_id,r.status ORDER BY r.started_at DESC LIMIT 1;"

docker compose --env-file .env.offline -f docker-compose.offline-gpu.yml exec -T pg \
  psql -U mvi -d uav_search -c \
  "SELECT dataset_id,active_run_id,activated_at FROM dataset_active_runs WHERE dataset_id='institution_m2ts_video_only';"

docker compose --env-file .env.offline -f docker-compose.offline-gpu.yml exec -T ch \
  clickhouse-client --user mvi --password "$CLICKHOUSE_PASSWORD" --query \
  "SELECT run_id,count() AS vectors FROM uav_search.seg_ch_512_runs WHERE dataset_id='institution_m2ts_video_only' GROUP BY run_id ORDER BY vectors DESC;"
```

Başarısız veya eksik run active pointer'ı değiştirmemelidir. PostgreSQL segment sayısı ile active run'a ait ClickHouse vector sayısı eşit olmalıdır.

## 8. Cache dışı canlı text query

`EMBEDDING_MODE=real` yolu query cache'i okumaz veya yazmaz; metin her request sırasında Qwen `embed_text` ile GPU'da embed edilir. Daha önce kullanılmamış benzersiz bir metin gönderin:

```bash
curl -fsS -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"query":"offline-m2ts-acceptance-UNIQUE-20260801 hareketli araç","dataset_id":"institution_m2ts_video_only","backend":"clickhouse","strategy":"exact","dimension":512,"top_k":10,"repeats":1,"diagnose":true,"explain":true}' \
  http://127.0.0.1:8000/search > artifacts/offline_m2ts_query.json
```

Yanıtta `embedding_mode=real`, `model_revision`, `timings_ms.embed`, `timings_ms.vector_search`, `timings_ms.total` ve her sonuçta `video_id`, `segment_id`, `t_start`, `t_end`, `score` bulunmalıdır. Request öncesi ve sonrası query-cache dosyası inventory/hash'inin değişmediğini ayrıca kaydedin.

## 9. Video-only strateji smoke

Global ClickHouse defaultu `prefilter` olarak kalır. Aynı benzersiz query ve `top_k` ile `exact`, `ann` ve `prefilter` request'lerini çalıştırın; her biri için aşağıdakileri JSON artifact'e kaydedin:

- `returned_count`;
- `timings_ms.vector_search` ve `timings_ms.total`;
- exact top-k segment ID kümesiyle agreement;
- `diagnostics.plan_used_vector_index`;
- `diagnostics.explain`/query plan.

Gerçek active corpus üzerinde ANN yeterli agreement ve index kullanımı göstermeden defaultu değiştirmeyin. Bu kanıt sağlanırsa yalnız bu video-only runbook için request bazında `strategy=ann` önerilebilir; aksi durumda mevcut `prefilter` defaultu korunur.

## 10. Kabul ve durdurma ölçütleri

20 GB ingest kararı ancak şu kanıtların tamamı mevcutsa `READY_FOR_20GB_INGEST` olabilir: registry-free temiz-host start, gerçek NVIDIA container GPU, doğrulanmış model bundle, native M2TS preflight, en az 10 gerçek Qwen window, tamamlanmış generic ingest, eşit PostgreSQL/ClickHouse sayıları, değişmiş active run, cache dışı canlı text embedding ve zaman aralıklı sonuç. Bunlardan biri eksikse karar en fazla `READY_AFTER_TARGET_SMOKE`; kod veya veri hatası varsa `NO_GO` olmalıdır.
