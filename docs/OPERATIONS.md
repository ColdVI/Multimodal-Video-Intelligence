# Faz 11 operations, recovery ve troubleshooting

## Günlük sağlık ve görünürlük

```bash
docker compose --env-file .env ps
curl -fsS http://${BIND_HOST:-127.0.0.1}:8000/health
curl -fsS -H "Authorization: Bearer ${API_TOKEN}" \
  http://${BIND_HOST:-127.0.0.1}:8000/stats
```

`/stats` dataset başına active run, status, segment count, provenance,
model/source revision, enabled backend/dimension ve backend count'larını verir.
Token boşsa Authorization header gerekmez. Secret'ı shell history/process
listesinde taşımamak için kurum secret manager veya izinleri sınırlı env-file
tercih edilmelidir.

## Ingest resume ve hata inceleme

```bash
docker compose --env-file .env exec api python -m app.ingestion.ingest \
  --dataset /workspace/datasets/kurum.yaml --resume
```

Run raporu `artifacts/ingest/<run_id>/report.json`, satır/hata manifesti aynı
run dizinindedir. `failed` veya yarım run active dataset'i değiştirmez. Resume
committed chunk'ları decode etmez; incomplete chunk'ı yalnız aynı inactive run
içinde temizleyip tekrar yazar.

## Migration, backup ve rollback

Önce DB/backend snapshot/backup alın. Ardından salt-okunur plan:

```bash
PYTHONPATH=service python scripts/migrate_faz11_schema.py --plan \
  --output artifacts/faz11/schema_migration_report.json
```

Count'lar ve Qdrant re-ingest gereksinimi operator tarafından incelendikten
sonra:

```bash
PYTHONPATH=service python scripts/migrate_faz11_schema.py --apply \
  --output artifacts/faz11/schema_migration_report.json
```

Migration legacy tabloları drop/truncate etmez. Uygulama rollback'i önceki
image tag/commit'e dönerek yapılır. Data rollback'i, doğrulanmış önceki
`completed` run'ın `dataset_active_runs` pointer'ına kontrollü PostgreSQL
transaction ile yeniden atanmasıdır; hedef run'ın metadata ve bütün enabled
backend×dimension count'ları tekrar doğrulanmadan pointer değiştirilmez. Ayrıntı:
[RUN_VERSIONING.md](RUN_VERSIONING.md).

## Güvenli run GC

```bash
PYTHONPATH=service python -m app.ingestion.gc_runs --dry-run \
  --retain-previous-completed 1 --min-age-hours 24
```

Dry-run raporu review edilmeden apply yapılmaz. Active, ingesting ve validating
run'lar korunur. Volume silme, `docker compose down -v`, repo `clean/reset` veya
legacy tablo temizliği recovery prosedürü değildir.

## Capacity tahmini

Bir dimension'ın sıkıştırılmamış float32 alt sınırı yaklaşık
`segment_count × dimension × 4 byte`'tır. Toplamı yalnız enabled dimension'lar
için hesaplayın; backend index/replication ve metadata overhead'ini gerçek pilot
ölçümüyle ekleyin. Staging sırasında active + staging, bir önceki completed run
retain ediliyorsa geçici olarak üçüncü corpus alanı gerekebilir. Media cache üst
sınırı ayrıca `MEDIA_CACHE_MAX_GB`'dır.

Decoder RAM'i corpus toplamından değil yaklaşık
`DECODE_PREFETCH_WINDOWS × EMBED_BATCH_SIZE × sampled_frames × decoded_frame_size`
pilotundan ölçülür. GPU VRAM için `artifacts/faz11/gpu_smoke.json` gerçek hedef
GPU'da `pass` olmadan üretim batch artırılmaz.

## Sık arızalar

- **Preflight exit 2:** `.env` placeholder/boş parola, bind veya Compose config.
- **Exit 3:** absolute/traversal glob, pairing, clock anchor, timestamp overlap
  ya da decoder problemi. Manifesti düzeltin; anchor tahmin etmeyin.
- **Exit 4/5:** driver/toolkit/CUDA veya bundle hash/import uyumsuzluğu. Synthetic
  fallback ile üretim ingest'i sürdürmeyin.
- **Search underfilled:** diagnostics'te candidate shortage ile ANN filter loss'u
  ayırın. Legacy candidate limit hatasında pushdown kullanın.
- **Media 403/404/422/503:** sırasıyla `DATA_ROOT` sınırı, eksik source, unsupported
  source/codec veya ffmpeg yokluğu. API image'ında ffmpeg bulunur; host path'in
  read-only mount altında gerçekten mevcut olduğunu doğrulayın.
- **Disk pressure:** yeni ingest'i durdurun, dry-run GC ve run durumlarını inceleyin;
  active volume veya tabloları manuel silmeyin.

## Durdurma

```bash
docker compose --env-file .env stop
```

Bu komut volume silmez. `down -v` bu runbook kapsamında yasaktır.
