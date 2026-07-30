# Faz 11 final implementation report

Durum: **`implementation_complete_hardware_acceptance_pending`**  
Test edilen kod SHA: `045ea81b83de366b3597c84a051d5c5f039603d5`  
Başlangıç SHA: `23eb2a894c9b24f998b05a93c6a33262a860796d`

Kod, otomatik test, additive migration, profile, preflight, ingest,
run-versioning, pushdown, UI/media, security ve deployment dokümantasyonu
tamamlandı. Bu macOS/arm64 hostta Docker daemon, NVIDIA GPU, kurum verisi ve
verified model bundle bulunmadığından hedef ortam kabulü başarılı gösterilmedi.
Makine-okunur matris: `artifacts/faz11/final_acceptance.json`.

## 1. Baseline

Baseline temiz `main` üzerinde alındı. `artifacts/faz11/baseline.json`, SHA,
dirty state, Python/pytest sürümleri, exact komutlar/exit code'lar, Compose ve
requirements hash'lerini taşır. İlk ortamda root collection
`clickhouse_connect` eksikliğiyle durdu; service sonucu `41 passed, 2 failed,
15 skipped` idi ve iki failure eager CapERA dosya erişimiydi. Syntax ve iki
mevcut Compose config'i başlangıçta geçti.

İzole `.testdeps` ortamına test bağımlılıkları kuruldu; kullanıcı dosyası veya
tracked dependency lock'u bu amaçla değiştirilmedi. Nihai sonuç:

- root: `314 passed, 42 skipped`;
- service: `104 passed, 17 skipped`;
- bütün `.py` dosyaları `py_compile`: PASS;
- `git diff --check`: PASS;
- canonical/GPU/benchmark/benchmark+debug Compose config: PASS.

Skip'ler mevcut gerçek dataset/integration readiness kapılarıdır; FAIL yoktur.

## 2. Mimari değişiklikler

- Kurum defaultu ClickHouse + 512d + native pushdown; Qdrant/pgvector ve dört
  dimension yalnız benchmark override ile açılır. DB portları canonical profilde
  hosta publish edilmez.
- Relative path güvenlikli dataset manifesti ve iki katmanlı read-only preflight,
  absolute/relative clock ve canonical/extra telemetry sözleşmesini doğrular.
- PyAV streaming decoder chunk+halo ownership ile bounded iterator üretir;
  continuous/categorical/circular telemetry pencereye hizalanır.
- Qwen kaynak/model revision'ları pinlidir; hash manifestli bundle read-only
  mount edilir; batch embedding GPU yoksa fail-closed davranır.
- Additive run-scoped storage, chunk ledger, finalize/atomic active pointer,
  resume, recovery, migration planı ve güvenli GC eski active run'ı korur.
- Generic ingest streaming decode → Qwen batch → enabled dimensions → enabled
  backends hattını çalıştırır; legacy loader compatibility korunur.
- Canonical filter registry ClickHouse kolonlarına, Qdrant payload/indexlerine
  ve pgvector JOIN'e projekte edilir. Default pushdown Python candidate listesi
  taşımaz; legacy path limitli ve açık etiketlidir.
- API dataset/run/filter-schema/media uçlarını ve run/model/provenance
  diagnostics'i additive sunar. UI backend/dimension/strategy seçeneklerini
  `/strategies`'ten alır; canonical alanları registry'ye göre gösterir.
- Media source PostgreSQL active snapshot'tan çözülür, `DATA_ROOT` dışı path
  reddedilir, H.264 clip arg-list ffmpeg ile atomik ve bounded cache'e yazılır.
- Optional Bearer token, public-bind preflight kapısı ve token içermeyen HMAC
  signed media URL uygulanmıştır. `/health` açık kalır.

## 3. Değişen dosyalar: neden ve doğrulama

| Dosya/grup | Neden | Doğrulama |
|---|---|---|
| `.env.example`, `service/app/config.py`, `docker-compose*.yml`, `Makefile` | Enabled profile, secure defaults, resource/media/model ayarları | profile tests + dört Compose parse |
| `datasets/example_uav.yaml`, `service/app/ingestion/manifest.py`, `service/app/preflight.py`, `scripts/preflight.py` | Portable veri/clock/path/preflight sözleşmesi | manifest/preflight root+service tests |
| `service/app/ingestion/video.py`, `telemetry.py`, `generic_loader.py` | Streaming decode, window ownership, interpolation/aggregation | `test_faz11_streaming.py` |
| `service/app/embedding/{bundle,qwen}.py`, `scripts/{prepare_model_bundle,gpu_smoke}.py`, `service/Dockerfile.gpu` | Pinned offline bundle ve gerçek batch API | model bundle + Qwen batch + GPU not-run testleri |
| `service/app/db/{ingest_runs,migrations,registry,telemetry_registry}.py` | Control plane, active snapshot ve capability registry | run-versioning/profile tests |
| `service/app/db/{postgres,clickhouse,qdrant}.py` | Run-scoped physical storage ve native predicates | recovery/pushdown/engine tests |
| `service/app/ingestion/{ingest,gc_runs}.py`, `scripts/migrate_faz11_schema.py` | Resume/finalize/report/GC ve additive migration CLI | generic ingest/run-versioning tests |
| `service/app/search/{filter_schema,filter_projection,pushdown,equivalence,engine}.py` | Canonical compile, backend pushdown, equivalence ve snapshot search | pushdown + engine tests |
| `service/app/{main,media,auth}.py` | Additive API, safe media ve optional token | media/UI/security tests |
| `service/ui/{app,components}.py`, `service/Dockerfile` | Dynamic UI, provenance/extra detail, real clip, ffmpeg | UI component/media tests |
| `docs/{PORTABILITY_AUDIT,DATASET_MANIFEST,MODEL_BUNDLE,RUN_VERSIONING,FILTER_PUSHDOWN,DEPLOYMENT,OPERATIONS}.md` | Kurum onboarding, karar, migration ve operation runbook | command/path review + docs links |
| `artifacts/faz11/*` | Baseline ve pass/not-run kabul kanıtları | JSON parse, mandatory artifact audit |
| `service/tests/test_faz11_*.py`, `tests/test_faz11_*.py` | Yeni pure/contract regression kapsamı | final pytest sonuçları |

Toplam Faz 11 farkı (final evidence dosyaları hariç): 81 dosya, yaklaşık 7.6k
ekleme ve 253 silme; silmeler refactor/compatibility düzeltmeleridir, kullanıcı
verisi/volume/legacy tablo silinmemiştir.

## 4. Çalıştırılan komutlar

Başlangıç ve final boyunca kullanılan exact ana komutlar:

```bash
git rev-parse HEAD
git status --short
git log -5 --oneline
make test PYTHON=.testdeps/bin/python
PYTHONPATH=service .testdeps/bin/python -m pytest service/tests/ -q -p no:cacheprovider
rg --files -g '*.py' -0 | xargs -0 .testdeps/bin/python -m py_compile
git diff --check
docker compose --env-file .env.example -f docker-compose.yml config
env MODEL_BUNDLE_ROOT=/private/tmp/mvi-model-bundle docker compose --env-file .env.example -f docker-compose.yml -f docker-compose.gpu.yml config
docker compose --env-file .env.example -f docker-compose.yml -f docker-compose.benchmark.yml config
docker compose --env-file .env.example -f docker-compose.yml -f docker-compose.benchmark.yml -f docker-compose.debug.yml config
env API_MEMORY_LIMIT=4g UI_MEMORY_LIMIT=2g CLICKHOUSE_MAX_MEMORY_BYTES=8589934592 docker compose --env-file .env.example -f docker-compose.yml config
.testdeps/bin/python scripts/write_ui_not_run_artifact.py
docker info --format '{{.ServerVersion}}'
nvidia-smi
```

İlk on aşamanın commit kapıları:

```text
4ee65d0 baseline/portability
15376b6 config/profiles
c2537c2 manifest/preflight
e51a149 streaming video/telemetry
df4df12 model bundle/Qwen batch
f5d7578 run-versioned storage
1a72169 generic ingest/resume
e5ed25f native pushdown
8e3b527 UI/media
045ea81 security/deployment/operations
```

## 5. Kabul matrisi

| Alan | Durum | Kanıt / gerekçe |
|---|---|---|
| Baseline, portability, profile/config | PASS | baseline + portability artifact/docs; Compose hashes |
| Manifest, path, clock ve preflight contracts | PASS | manifest/preflight tests |
| Streaming video/telemetry/window contracts | PASS | streaming tests |
| Model bundle/hash ve batch API contract | PASS | bundle/Qwen tests |
| Gerçek 10-window Qwen GPU inference | NOT RUN | `gpu_smoke.json`: bu hostta CUDA yok |
| Run storage, recovery, finalize/activation, GC | PASS | run-versioning tests |
| Canlı legacy volume migration | NOT RUN | `schema_migration_report.json`: live DB/driver yok |
| Generic ingest ve injected crash/resume contract | PASS | generic ingest tests |
| Gerçek kurum verisiyle interrupted resume | NOT RUN | `ingest_resume_smoke.json` |
| Filter compiler, circular wrap, active snapshot | PASS | pushdown/engine tests |
| Canlı backend equivalence/scale/index planı | NOT RUN | `filter_equivalence.json`, `search_scale_smoke.json` |
| Dynamic UI, run/provenance/extra detail | PASS | media/UI pure tests |
| Safe clip cache ve token security | PASS | media/security tests |
| Canlı UI screenshot ve local MP4 playback | NOT RUN | açık etiketli `ui_smoke.png` + `ui_smoke.json` |
| Deployment/operations/migration docs | PASS | `docs/DEPLOYMENT.md`, `docs/OPERATIONS.md` |
| Hedef kurum full acceptance | NOT RUN | Docker/GPU/credentials/kurum verisi gerekli |

Nihai sayım: **19 PASS, 0 FAIL, 0 BLOCKED, 9 NOT RUN**. Her NOT RUN satırının
`reason`, `required_command` ve `expected_environment` alanı
`artifacts/faz11/final_acceptance.json` içinde yer alır.

## 6. Hedef ortamda tamamlanacak kabul sırası

```bash
python scripts/preflight.py --dataset datasets/kurum.yaml --env-file .env \
  --json-out artifacts/faz11/preflight.json
docker compose --env-file .env -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
docker compose --env-file .env exec api python -m app.ingestion.ingest \
  --dataset /workspace/datasets/kurum.yaml --resume
PYTHONPATH=service python scripts/gpu_smoke.py --dataset datasets/kurum.yaml \
  --data-root /kurum/data --output artifacts/faz11/gpu_smoke.json --windows 10
RUN_FAZ8_INTEGRATION=1 UI_URL=http://127.0.0.1:7860 \
  PYTHONPATH=service pytest service/tests/test_t10_ui.py -q
```

Bu sıra tamamlanana kadar durum bilinçli olarak
`implementation_complete_hardware_acceptance_pending` kalır.
