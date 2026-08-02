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

## 2. Neden tek TAR ve hangi bağımlılıklar hostta kalır?

Kurum aktarım prosedürü yalnız hazırlanmış Docker image TAR'ının yüklenmesini garanti eder. Bu nedenle Qwen source, model ağırlıkları ve üç model manifesti `mvi-app-gpu:<git-sha>` image'ının `/opt/mvi-model-bundle` yoluna gömülür. Ayrı model klasörü kopyalama veya model bind mount'u yoktur. Kurum videoları ise büyük ve kuruma özgü olduğu için image'a girmez; `DATA_ROOT` salt okunur biçimde `/workspace/data` yoluna mount edilir.

Image içine konamayan ve hedef hostta önceden hazır olması gerekenler:

- Linux/amd64 Docker Engine ve Docker Compose v2;
- uyumlu NVIDIA sürücüsü ve `nvidia-smi`;
- NVIDIA Container Toolkit ve Docker `nvidia` runtime'ı;
- doğrulama scripti için Python 3 standart kütüphanesi ve health bekleme için `curl`;
- yeterli disk alanı ve GPU belleği.

Bundle Docker Engine, NVIDIA sürücüsü veya Container Toolkit kurmaz. Hedef bilgisayarda `pip install`, `apt-get`, `docker build` veya `docker pull` çalıştırılmaz.

## 3. MacBook M4 üzerinde tek komutla üretim

Bu akış `agent/offline-single-load-bundle` çalışma branch'inde, `feat/advanced-retrieval-evidence-gated` branch'inin `ec3963fef230...` commit'i üzerinden geliştirilmiştir. Release checkout'u bu ileri branch'ten gelmeli ve temiz olmalıdır. Builder branch/commit'i yazdırır, merge-base kontrolü yapar ve üretilen her gerçek bundle'ın kesin commit'ini `bundle_manifest.json` içindeki `git_sha` ile kaydeder. Apple Silicon host `darwin/arm64` olsa da kurum hedefi değişmez:

```text
linux/amd64
```

Buildx emülasyon/çapraz-build mekanizması nedeniyle her build ve pull bu platformla açıkça çalışır. ARM64 olarak inspect edilen tek bir image bile export'u durdurur.

Docker Desktop çalışırken repository kökünde yalnız şu komutu çalıştırın:

```bash
./scripts/build_offline_bundle_macos.sh
```

Script en az 60 GB boş alanı denetler, `.runtime/` altında küçük bir venv kurar, pinned modeli/source'u `scripts/prepare_model_bundle.py` ile hazırlar veya mevcut bundle'ın bütün hash zincirini yeniden doğrular. Ardından BuildKit named context ile `gpu-bundled` target'ını build eder. Model dosyaları Git'e veya normal repository build context'ine kopyalanmaz.

Sabit pinler:

- model: `Qwen/Qwen3-VL-Embedding-2B`;
- revision: `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`;
- source commit: `393e2978d27852b0d0230d6994f37f9c15bed73c`.

Tek TAR tam olarak şu image'ları içerir:

```text
mvi-app-gpu:<git-sha>
pgvector/pgvector:pg16
clickhouse/clickhouse-server:25.8
```

API ve UI aynı application image ID'sini kullanır; yalnız API `gpus: all` ister. Exporter container'ı `--network none --pull never` ile açıp gömülü model/source dizinlerini, pinleri, hash manifestlerini, symlink yokluğunu ve Python importlarını doğrular. Mac'te NVIDIA runtime yoksa GPU smoke `SKIPPED: host has no NVIDIA runtime` olarak kaydedilir; build bundan dolayı başarısız olmaz.

Varsayılan `.runtime/offline-bundles/` altında her koşu timestamp ve SHA içeren yeni bir dizin üretir; mevcut output üzerine yazılmaz. `OUTPUT_ROOT`, `MODEL_CACHE_ROOT` ve `TARGET_PLATFORM` environment değişkenleri opsiyoneldir, fakat `TARGET_PLATFORM` yalnız `linux/amd64` olabilir.

## 4. Nihai transfer dizini

```text
mvi-offline-bundle-<sha>-<timestamp>/
├── images/mvi-images-linux-amd64.tar
├── docker-compose.offline-gpu.yml
├── .env.offline.example
├── datasets/video_only_m2ts.yaml
├── scripts/verify_offline_bundle.py
├── install-and-start-offline.sh
├── bundle_manifest.json
└── SHA256SUMS
```

`model-bundle/` yoktur; model yalnız application image'ının `/opt/mvi-model-bundle/{source,model}` yollarındadır. `bundle_manifest.json` build hostunu, hedef platformu, image ID'lerini, TAR'ın gerçek boyut/hash'ini ve `embedded_model_bundle` pinlerini içerir. `SHA256SUMS`, TAR dahil bütün taşıma dosyalarını kapsar. Model boyutu transfer toplamına ikinci kez eklenmez. Gerçek TAR/dizin boyutu builder sonunda `du` ve SHA-256 ile yazdırılır; build yapılmadan verilen 9–16 GB aralığı yalnız planlama tahminidir.

Video dizini bundle'ın yanında veya başka bir absolute Linux yolunda kalabilir:

```text
/transfer/mvi-offline-bundle-.../
/kurum/videos/**/*.m2ts
```

`.env.offline.example` dosyasını kopyalayın ve bütün `CHANGE_ME` değerleriyle `DATA_ROOT` yolunu değiştirin:

```bash
cd /transfer/mvi-offline-bundle-...
cp .env.offline.example .env.offline
```

```env
DATA_ROOT=/kurum/videos
ARTIFACTS_ROOT=./artifacts
```

Model için host path ayarı yoktur. Compose'ta `build:` bulunmaz, her serviste `pull_policy: never` bulunur ve video mount'u read-only'dir.

## 5. Offline yükleme ve başlatma

Yalnız hash, `docker load`, image ID/platform ve gömülü model doğrulaması gerekiyorsa:

```bash
bash install-and-start-offline.sh --load-only
```

Stack'i de başlatmak için:

```bash
bash install-and-start-offline.sh
```

Starter sırasıyla SHA256 inventory'yi, Docker/Compose'u, linux/amd64 engine'i, `nvidia-smi` ve NVIDIA runtime'ı denetler; placeholder credential varsa durur; TAR'ı `docker load --input` ile yükler; local image ID'lerini manifestle karşılaştırır; application container'ını ağsız açarak gömülü bundle'ı yeniden doğrular; Compose config'i doğrular. Start modunda kullanılan son komut sözleşmesi şudur:

```bash
docker compose --env-file .env.offline \
  -f docker-compose.offline-gpu.yml \
  up -d --no-build --pull never
```

API ve UI health endpointleri hazır olana kadar beklenir. Hiçbir registry pull girişimi yapılmaz.

Sorun halinde, sırayla `sha256sum -c SHA256SUMS`, `docker info`, `nvidia-smi`, `docker info --format '{{json .Runtimes}}'`, `docker image inspect <tag>` ve `docker compose --env-file .env.offline -f docker-compose.offline-gpu.yml config` çıktılarını inceleyin. Schema v1/eski bundle, ARM64 image, eksik layer/tag, yanlış image ID, bozuk hash, model mount'u veya `CHANGE_ME` credential açık hata ile reddedilir.

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
