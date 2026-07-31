# FAZ 11 kullanım kılavuzu — teknik operatör

Bu kılavuz, sistemi hiç görmemiş bir teknik ekip üyesinin sıfırdan kurup
çalıştırabilmesi için yazıldı. Her komut bu repodaki gerçek CLI/Compose
tanımlarıyla eşleşir; hiçbir bayrak veya endpoint uydurulmadı. Derin
referans için: [DEPLOYMENT.md](DEPLOYMENT.md) (kurulum), [OPERATIONS.md](OPERATIONS.md)
(günlük işletim/kurtarma), [DATASET_MANIFEST.md](DATASET_MANIFEST.md) (veri
sözleşmesi), [MODEL_BUNDLE.md](MODEL_BUNDLE.md) (model pinleme),
[RUN_VERSIONING.md](RUN_VERSIONING.md) (run/aktivasyon), [FILTER_PUSHDOWN.md](FILTER_PUSHDOWN.md)
(arama filtreleri). Bu dosya bunları tek bir uçtan uca operatör yolculuğuna
dizer.

Hızlı başlangıç için [OPERATOR_QUICKSTART.md](OPERATOR_QUICKSTART.md)'a bakın.

## 1. Sistem ne yapar?

Doğal dil sorgusuyla İHA video arşivinde arama yapılmasını sağlar: video,
`Qwen/Qwen3-VL-Embedding-2B` ile pencere bazlı embedding'e dönüştürülür,
telemetri (konum/irtifa/hız/yön/gimbal) her pencereye hizalanır, ve arama
zamanı sorgu embedding'i + isteğe bağlı telemetri/metadata filtreleri
backend-native olarak (Python'a candidate ID listesi taşınmadan) çalıştırılır.
UI'da sonuç, ilgili video kesiti, aktif run/model bilgisi ve provenance
(gerçek mi sentetik mi) birlikte görünür.

## 2. Mimari özeti

```text
gerçek video (+ opsiyonel telemetri CSV)
  → chunk + window streaming decode (PyAV/OpenCV)
  → Qwen3-VL-Embedding-2B batch embedding (2048d, MRL truncate → enabled dimensions)
  → PostgreSQL (metadata, telemetri, run/control-plane)
  → ClickHouse (kurum varsayılanı vector store) [+ opsiyonel Qdrant/pgvector benchmark]
  → FastAPI (/search, /health, /stats, /media/...)
  → Gradio UI (arama, filtreler, sonuç detay, video oynatma)
```

Kurum varsayılan profili: `ENABLED_VECTOR_BACKENDS=clickhouse`,
`ENABLED_DIMENSIONS=512`, `FILTER_EXECUTION_MODE=pushdown`. Bu üçü değişmeden
sistemin ana yoludur; Qdrant/pgvector ve ek boyutlar yalnız
`docker-compose.benchmark.yml` ile açılır.

## 3. Desteklenen deployment biçimleri

| Profil | Compose dosyaları | Ne zaman |
|---|---|---|
| Kurum (varsayılan) | `docker-compose.yml` + `docker-compose.gpu.yml` | Üretim |
| Kurum, GPU'suz CPU smoke | yalnız `docker-compose.yml` (`EMBEDDING_MODE=synthetic/cached`) | Donanım gelmeden entegrasyon testi |
| Benchmark | + `docker-compose.benchmark.yml` | Backend/boyut karşılaştırması |
| Debug | + `docker-compose.debug.yml` | DB portlarını loopback'e açmak (yalnız yerel debug) |
| Colab portable | [COLAB_RUNBOOK.md](COLAB_RUNBOOK.md) | Yalnız GPU embedding üretimi + küçük ölçek değerlendirme; kalıcı üretim deploy'u değil |

## 4. Kurum makinesi minimum gereksinimleri

- NVIDIA GPU (Qwen3-VL-Embedding-2B için pratik anlamda en az Ampere sınıfı,
  ör. A10/A100/L4/T4; `flash_attention_2` seçilecekse compute capability
  ≥ 8.0 zorunlu — `sdpa` varsayılanı daha geniş donanımla çalışır).
- Docker Engine + Compose v2, NVIDIA Container Toolkit.
- `DATA_ROOT` için video corpus'unuza yetecek disk (bkz. §28 kapasite tahmini).
- `MODEL_BUNDLE_ROOT` için pinlenmiş Qwen kaynak+model bundle'ı (§9).
- İşletim sistemi: bu repo Linux/Docker hedefler; geliştirme/test bu depoda
  Windows'ta da doğrulanmıştır ama üretim runbook'u Linux host varsayar.

## 5. NVIDIA driver ve Docker ön koşulları

```bash
nvidia-smi
docker version
docker compose version
docker run --rm --gpus all nvidia/cuda:${CUDA_IMAGE_TAG:-12.1.1-runtime-ubuntu22.04} nvidia-smi
```

Bunlardan biri başarısızsa `scripts/preflight.py` de aynı gerekçeyle FAIL
verecektir — host taraf sorunu önce burada çözülmelidir.

## 6. Repository clone

```bash
git clone <repo-url>
cd Multimodal-Video-Intelligence
```

## 7. `.env.example` → `.env` hazırlama

```bash
cp .env.example .env
```

`.env.example`'daki tüm anahtarlar (gerçek dosyadan):

```text
POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
CLICKHOUSE_DB, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD
ENABLED_VECTOR_BACKENDS, ENABLED_DIMENSIONS, FILTER_EXECUTION_MODE, LEGACY_CANDIDATE_LIMIT
EMBEDDING_MODE
DECODE_CHUNK_S, DECODE_PREFETCH_WINDOWS, EMBED_BATCH_SIZE, DB_WRITE_BATCH_SIZE
BIND_HOST, DATA_ROOT, ARTIFACTS_ROOT, MODEL_BUNDLE_ROOT, API_URL, API_TOKEN
QWEN_REPO_PATH, QWEN_MODEL_PATH, QWEN_MODEL_ID, QWEN_MODEL_REVISION, QWEN_SOURCE_COMMIT, ATTN_IMPL
CUDA_IMAGE_TAG, CLICKHOUSE_MAX_MEMORY_BYTES, API_MEMORY_LIMIT, UI_MEMORY_LIMIT
MEDIA_MAX_CLIP_S, MEDIA_CACHE_MAX_GB, MEDIA_CACHE_RETENTION_HOURS, MEDIA_H264_CRF, MEDIA_URL_TTL_S
```

`CHANGE_ME_*` ile başlayan her değeri değiştirmeden preflight/Compose fail-fast
verir (`REQUIRE_SECURE_CREDENTIALS=true` API container'ında sabittir).

## 8. Güvenli credential üretimi

```bash
openssl rand -base64 32   # POSTGRES_PASSWORD
openssl rand -base64 32   # CLICKHOUSE_PASSWORD
openssl rand -hex 32      # API_TOKEN (boş bırakılırsa yalnız loopback bind'da auth kapalı kalır)
```

`.env` dosyasını asla commit etmeyin; `git status` ile kontrol edin.
`BIND_HOST` loopback (`127.0.0.1`) dışında bir değere ayarlanacaksa
`API_TOKEN` doldurulmadan `scripts/preflight.py` `api_exposure` kontrolünde
FAIL verir.

## 9. Qwen model bundle hazırlama

İnternete çıkabilen ayrı bir hazırlık makinesinde:

```bash
python scripts/prepare_model_bundle.py \
  --model-id Qwen/Qwen3-VL-Embedding-2B \
  --model-revision 9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda \
  --source-repo https://github.com/QwenLM/Qwen3-VL-Embedding.git \
  --source-commit 393e2978d27852b0d0230d6994f37f9c15bed73c \
  --bundle-root /kurum/model-bundle
```

Bundle'ı hedef hosta kopyaladıktan sonra `MODEL_BUNDLE_ROOT=/kurum/model-bundle`
olarak `.env`'e yazın. Detay ve offline hazırlık: [MODEL_BUNDLE.md](MODEL_BUNDLE.md).
`ATTN_IMPL=flash_attention_2` seçerseniz preflight, `flash_attn` paketinin
kurulu olduğunu ve GPU compute capability'sinin ≥8.0 olduğunu ayrıca
doğrular (`flash_attention_2_support` kontrolü); karşılanmazsa varsayılan
`sdpa`'da kalın.

## 10. Dataset dizin yapısı

`DATA_ROOT` altında (host path `.env`'de, container'da salt-okunur
`/workspace/data` olarak mount edilir):

```text
DATA_ROOT/
  videos/
    flight-001.mp4
    flight-002.mp4
  telemetry/
    flight-001.csv
    flight-002.csv
```

Manifestteki `videos_glob`/`telemetry_glob` bu köke görelidir; mutlak yol veya
`..` reddedilir (bkz. [DATASET_MANIFEST.md](DATASET_MANIFEST.md)).

## 11. Dataset manifesti oluşturma

```bash
cp datasets/example_uav.yaml datasets/kurum.yaml
```

En az düzenlenecek alanlar: `dataset_id`, `source.videos_glob`,
`pairing.telemetry_glob`, `time_alignment` (clock/anchor/offset),
`telemetry.fields` (kaynak kolon adlarınız). Ayrıntılı alan-alan rehber:
[DATASET_ONBOARDING_GUIDE.md](DATASET_ONBOARDING_GUIDE.md).

## 12. Preflight çalıştırma

```bash
python scripts/preflight.py \
  --dataset datasets/kurum.yaml \
  --env-file .env \
  --json-out artifacts/faz11/preflight.json
```

Exit code'lar: `0` başarı, `2` configuration, `3` data/manifest, `4` GPU/runtime,
`5` model bundle, `6` disk/resources. `status=pass` olmadan ingest başlatmayın.
Preflight hiçbir DB/data yazımı yapmaz (bkz.
[preflight_no_write_audit.json](../artifacts/faz11/preflight_no_write_audit.json)).

## 13. Compose ile sistemi başlatma

```bash
docker compose --env-file .env -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
docker compose --env-file .env ps
curl -fsS http://${BIND_HOST:-127.0.0.1}:8000/health
```

Kurum profili yalnız `pg`, `ch`, `api`, `ui` servislerini başlatır; DB portları
hosta publish edilmez (yalnız `expose`). Benchmark ihtiyacında:

```bash
docker compose --env-file .env \
  -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.benchmark.yml \
  up -d --build
```

## 14. Migration plan/apply

Bu ilk kurulumda gerekmez (fresh deployment migration istemez). Var olan bir
Faz 7-10 volume'unu Faz 11 run-scoped şemasına taşıyorsanız:

```bash
# 1) Önce backup/snapshot alın (operatör sorumluluğu)
docker compose --env-file .env exec -T api \
  python scripts/migrate_faz11_schema.py --plan --output artifacts/faz11/schema_migration_report.json
# 2) Planı ve row count'ları inceleyin, sonra:
docker compose --env-file .env exec -T api \
  python scripts/migrate_faz11_schema.py --apply --output artifacts/faz11/schema_migration_report.json
```

`--apply`, `--plan`'dan ayrı ve zorunlu bir bayraktır; migration legacy
tabloları asla DROP/TRUNCATE etmez. Detay: [RUN_VERSIONING.md](RUN_VERSIONING.md#existing-volume-migration).

## 15. İlk ingest

```bash
docker compose --env-file .env exec api \
  python -m app.ingestion.ingest --dataset /workspace/datasets/kurum.yaml --resume
```

Container içindeki path `/workspace/datasets/...`'dir (host `./datasets`
mount'u). Çıktı JSON'ı `run_id`, `status`, `segments`, `chunks` alanlarını
taşır — `run_id`'yi saklayın.

## 16. Kesilmiş ingest'i resume etme

Aynı komutu `--resume` ile tekrar çalıştırmak yeterlidir; committed chunk'lar
yeniden decode edilmez, yarım kalan chunk yalnız kendi (henüz aktif olmayan)
run'ı içinde temizlenip yeniden yazılır. Eski aktif run bu sırada aramada
kullanılabilir kalır.

## 17. Ingest run durumunu izleme

```bash
curl -fsS -H "Authorization: Bearer ${API_TOKEN}" http://${BIND_HOST:-127.0.0.1}:8000/stats
curl -fsS -H "Authorization: Bearer ${API_TOKEN}" http://${BIND_HOST:-127.0.0.1}:8000/ingest-runs/<run_id>
```

Run raporu ayrıca `artifacts/ingest/<run_id>/report.json` içindedir; hata
manifesti aynı dizinde `errors.jsonl`.

## 18. Active run mantığı

Her dataset'in tek bir "active run"ı vardır (`dataset_active_runs`). Yeni bir
ingest, tüm metadata + enabled backend×dimension satır sayıları doğrulanmadan
bu pointer'ı değiştirmez — başarısız/yarım bir ingest asla mevcut aramayı
bozmaz. Detay: [RUN_VERSIONING.md](RUN_VERSIONING.md).

## 19. UI'a erişme

```text
http://<BIND_HOST>:7860
```

`API_TOKEN` doluysa UI aynı token'ı API çağrılarına Bearer header olarak
otomatik ekler; kullanıcıdan ayrıca giriş istenmez.

## 20. Yeni video/dataset ekleme

```text
1) yeni .mp4 dosyasını DATA_ROOT/videos/ altına koy
2) varsa telemetry CSV'sini DATA_ROOT/telemetry/ altına koy
3) manifesti gerekiyorsa güncelle (yeni glob deseni, yeni pairing kuralı)
4) python scripts/preflight.py --dataset datasets/kurum.yaml --env-file .env
5) docker compose exec api python -m app.ingestion.ingest --dataset /workspace/datasets/kurum.yaml --resume
6) yeni run tamamlanınca /stats üzerinden active_run_id'nin güncellendiğini doğrula
7) UI'dan smoke bir arama yap
```

Başarısız yeni ingest eski active run'ı **bozmaz** — §18'e bakın.

## 21. Dataset güncelleme

Var olan videoları değiştirmek yerine yeni video ekleyip §20'yi tekrarlayın;
manifest hash değiştiyse yeni bir run oluşur, `--resume` eski run'ı resume
etmeye çalışmaz (resume anahtarı `dataset_id + manifest_hash`'tir).

## 22. Eski run'ları güvenli temizleme

```bash
docker compose --env-file .env exec api \
  python -m app.ingestion.gc_runs --dry-run --retain-previous-completed 1 --min-age-hours 24
```

Dry-run raporunu inceleyin; gerçek silme için `--dry-run`'ı kaldırın. Aktif,
`ingesting`/`validating` durumundaki run'lar hiçbir zaman aday değildir.
`docker compose down -v`, `git clean`, manuel tablo silme bu prosedürün
**yerine geçmez** — kullanmayın.

## 23. Backup ve recovery

Bu repo bir backup aracı sağlamaz; PostgreSQL/ClickHouse volume'larının
(`pg_data`, `ch_data`) kurumun standart snapshot/backup politikasıyla
yedeklenmesi operatör sorumluluğudur — özellikle migration `--apply`'dan
önce. Rollback prosedürü: [OPERATIONS.md](OPERATIONS.md#migration-backup-ve-rollback).

## 24. Log ve artifact konumları

```text
artifacts/ingest/<run_id>/report.json      ingest run raporu
artifacts/ingest/<run_id>/errors.jsonl     video/chunk hataları
artifacts/faz11/*.json                      preflight/migration/acceptance kanıtları
artifacts/media_cache/                      ffmpeg clip cache
docker compose logs -f api                  API stdout/stderr
docker compose logs -f ui                   UI stdout/stderr
```

## 25. Sık hatalar ve çözümü

Bkz. [OPERATIONS.md#sık-arızalar](OPERATIONS.md) — preflight exit code'ları,
media 403/404/422/503 anlamları, search underfilled/candidate shortage
ayrımı, disk pressure prosedürü.

## 26. Sistemi kapatma

```bash
docker compose --env-file .env stop
```

Volume silmez. `down -v` bu runbook kapsamında yasaktır.

## 27. Sistemi güncelleme

```bash
git pull
docker compose --env-file .env -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Şema değişikliği içeren bir sürüme geçerken önce §14'teki migration
akışını izleyin.

## 28. Kurum acceptance komutları

Tek makine-okunur sonuç için: [TARGET_ENVIRONMENT_ACCEPTANCE.md](TARGET_ENVIRONMENT_ACCEPTANCE.md)
ve `scripts/run_faz11_acceptance.py`. Kapasite tahmini formülü
([OPERATIONS.md](OPERATIONS.md#capacity-tahmini)):

```text
vector_bytes ≈ segment_count × enabled_dimension_toplamı × 4 byte × enabled_backend_sayısı
decoder_RAM  ≈ DECODE_PREFETCH_WINDOWS × EMBED_BATCH_SIZE × sampled_frames × decoded_frame_size (pilot ölçümü)
```

## Kullanıcıdan Beklenenler

### Teknik operatörden beklenenler

- NVIDIA Linux makine (veya Colab GPU ortamı — yalnız embedding üretimi için, §3).
- Docker Engine + Compose v2, NVIDIA Container Toolkit.
- MP4 video dosyaları ve varsa telemetri CSV'leri.
- Dataset manifest YAML'ı (§11, [DATASET_ONBOARDING_GUIDE.md](DATASET_ONBOARDING_GUIDE.md)).
- Güvenli PostgreSQL/ClickHouse parolaları ve isteğe bağlı `API_TOKEN`.
- Pinlenmiş Qwen model bundle'ı (§9).
- Yeterli disk alanı ve `DATA_ROOT` tanımı.
- `BIND_HOST`/network erişim kararı.

### Normal son kullanıcıdan beklenenler

- UI'a tarayıcıdan erişmek, dataset seçmek, doğal dil sorgusu yazmak,
  gerekirse filtre seçmek, sonuçları incelemek. Ayrıntı:
  [END_USER_GUIDE.md](END_USER_GUIDE.md).

### Normal son kullanıcıdan beklenmeyenler

- Model kurmak veya GPU/Docker yönetmek.
- SQL yazmak veya embedding üretmek.
- ClickHouse/Qdrant/pgvector yapılandırması bilmek.
- Her sorguda videoları elle yeniden işlemek — ingest bir kez yapılır, arama
  saniyeler içinde önceden üretilmiş embedding'ler üzerinden çalışır.
