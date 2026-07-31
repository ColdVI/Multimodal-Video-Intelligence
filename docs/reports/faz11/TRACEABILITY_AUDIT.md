# FAZ 11 bağımsız traceability audit (final)

Bu denetim iki oturumda yürütüldü ve `d3fef0e`'de (baz kod `045ea81`) sona
erer. `docs/FAZ11_FINAL_REPORT.md` ve önceki `artifacts/faz11/final_acceptance.json`
kaynak gerçek kabul edilmedi; her satır kod okuma, gerçek test çalıştırma
veya statik kaynak taraması ile bağımsız doğrulandı.

- **Oturum 1** (`02a6c10`, `548698b`): baseline yeniden doğrulama, ilk
  traceability audit, iki kritik streaming defekti (bounded pipeline,
  frame lifecycle) tespit ve düzeltme.
- **Oturum 2** (`2ebc476` .. `d3fef0e`): oturum 1'de "spot-checked, deep-dive
  pending" bırakılan beş alanın derinlemesine denetimi, run-versioning
  10-nokta fault matrix, pushdown adapter audit, gerçek streaming memory
  smoke kanıtı, ve tam kullanım kılavuzu seti.

## Bağımsız test yeniden-çalıştırma

| Komut | Sonuç (bu oturum, `d3fef0e`) |
|---|---|
| `PYTHONPATH=service pytest service/tests/ -q` | **150 passed, 16 skipped** |
| `pytest tests/ -q` | **375 passed** |
| `py_compile` (tüm `.py`, `.venv`/`.testdeps` hariç) | PASS |
| `git diff --check` | PASS |
| `docker compose --env-file .env.example -f docker-compose.yml config` | PASS (sha256 `3b7cb51a...`) |
| `+ -f docker-compose.gpu.yml` (`MODEL_BUNDLE_ROOT=/private/tmp/mvi-model-bundle`) | PASS (sha256 `aff9f957...`) |
| `+ -f docker-compose.benchmark.yml` | PASS (sha256 `73c5f134...`) |
| `+ -f docker-compose.debug.yml` | PASS (sha256 `2b7437f1...`) |

16 service skip'i sınıflandırıldı: 15'i `RUN_FAZ8_INTEGRATION=1` ile açılan
canlı Docker/Playwright entegrasyon testleri (bu host'ta Docker daemon
kapalı), 1'i bu host/kullanıcının symlink oluşturma izni olmaması
(`test_media_rejects_symlink_escape`, host izin verirse otomatik geçer). Hiç
biri gizlenmiş bir regresyon değildir; `test_t4_patterns.py`'deki
"pattern not implemented" skip'i önceden var olan, dürüst etiketli bir
skip'tir (`PATTERN_EXECUTION_IMPLEMENTED = False` ile tutarlı).

Bu host'ta (orijinal macOS build host'unun aksine) Docker CLI/Compose ve bir
GPU (GeForce GT 1030, 4GB VRAM) mevcut, ancak Docker daemon'ı bu oturumlarda
kapalıydı ve GT 1030 temsili kurum donanımı değil — canlı Docker/GPU kabul
adımları yine de tetiklenmedi.

## Kritik bulgular ve düzeltmeler (iki oturum toplamı)

| # | Bulgu | Ciddiyet | Durum | Commit |
|---|---|---|---|---|
| F1 | `DECODE_PREFETCH_WINDOWS`/`DB_WRITE_BATCH_SIZE` ölü konfigürasyondu — üç ayar tek `EMBED_BATCH_SIZE`'a indirgenmişti | high | fixed | `548698b` |
| F2 | `iter_chunk_windows` kaynak frame'i ve `GeneratorExit`'i açıkça kapatmıyordu | medium | fixed | `548698b` |
| F3 | `ATTN_IMPL=flash_attention_2` için `flash_attn`/GPU capability kontrolü hiç yoktu | medium | fixed | `2ebc476` |
| M1 | `_copy_clickhouse_legacy`'de idempotency guard yoktu — ikinci `--apply` satırları duplike ederdi | high | fixed | `9c8fd73` |
| M2 | Migration retry-after-failure asla activate() edemezdi (`status='validating'` reset eksikti) | high | fixed | `9c8fd73` |
| T1 | `telemetry.py` naive iso8601 zaman damgasını manifest.timezone yerine hep UTC sayıyordu | medium | fixed | `6571306` |
| T2 | Categorical alan için interpolation/aggregation hiç doğrulanmıyordu (circular_deg'in aksine) | medium | fixed | `6571306` |

Media/auth (§2.3) ve pushdown adapter (§4) denetimlerinde **hiçbir gerçek
defekt bulunmadı** — yalnız test kapsamı genişletildi.

## Alan-alan durum matrisi

Önceki auditte "spot-checked, deep-dive pending" bırakılan beş alan artık
tam durumdadır:

### Model bundle hash/provenance zinciri

| implementation_files | test_files | production_call_path | artifact | test_command | status |
|---|---|---|---|---|---|
| `service/app/embedding/bundle.py`, `scripts/prepare_model_bundle.py` | `tests/test_faz11_model_bundle.py`, `service/tests/test_faz11_preflight_model.py` | `service/app/preflight.py::_append_model_checks` → `verify_bundle()` (gerçek preflight yolunda, ayrı script değil) | — | `pytest tests/test_faz11_model_bundle.py service/tests/test_faz11_preflight_model.py -q` | **pass** |

Kanıt: hash zinciri (source/model manifest + bundle manifest üç katmanlı
SHA-256), revision/commit/model_id mismatch reddi, tamper edilmiş detay
manifest reddi, `docker-compose.gpu.yml`'in bundle'ı `:ro` mount ettiği ve
`QWEN_REPO_PATH`/`QWEN_MODEL_PATH` ile eşleştiği, `Dockerfile.gpu`'nun
build-time clone/download yapmadığı, `flash_attention_2` seçilince
dependency+GPU capability kontrolünün artık çalıştığı — hepsi gerçek
production fonksiyonları çağıran testlerle doğrulandı. Kalan iş: gerçek
bundle indirme ve gerçek GPU'da doğrulama (donanım yok, `not_run`).

### Migration idempotency ve legacy koruması

| implementation_files | test_files | production_call_path | artifact | test_command | status |
|---|---|---|---|---|---|
| `service/app/db/migrations.py`, `scripts/migrate_faz11_schema.py` | `service/tests/test_faz11_migrations.py` (önceden **hiç yoktu**) | `apply_migration()`/`plan_migration()` doğrudan (script bunları çağırır) | `artifacts/faz11/migration_contract_audit.json` | `pytest service/tests/test_faz11_migrations.py -q` | **pass** (2 gerçek bug bulundu ve düzeltildi: M1, M2) |

Kalan iş: canlı PostgreSQL/ClickHouse'a karşı gerçek `--plan`/`--apply`
(`not_run` — bu host'ta `psycopg2` kurulu değil, Docker daemon kapalı).

### Media path traversal ve signed URL güvenliği

| implementation_files | test_files | production_call_path | artifact | test_command | status |
|---|---|---|---|---|---|
| `service/app/media.py`, `service/app/auth.py` | `service/tests/test_faz11_media_ui.py`, `service/tests/test_faz11_security.py` | `app/main.py::media()`/`media_information()` → `get_clip()`/`media_info()`; `TokenAuthMiddleware` | — | `pytest service/tests/test_faz11_media_ui.py service/tests/test_faz11_security.py -q` | **pass** (defekt bulunmadı) |

Parent traversal, symlink escape (host izin verirse), encoded-traversal
segment_id (yalnız opak DB anahtarı olduğu kanıtlandı), negatif/ters zaman
aralığı reddi, ffmpeg arg-list, atomic cache publish, HMAC constant-time
compare, signed URL'de token yokluğu, expiry, cross-path/cross-endpoint
signature reuse reddi — hepsi gerçek fonksiyonlara karşı test edildi.

### Preflight no-write garantisi

| implementation_files | test_files | production_call_path | artifact | test_command | status |
|---|---|---|---|---|---|
| `service/app/preflight.py` | `service/tests/test_faz11_preflight_no_write.py` (önceden **hiç yoktu**) | `run_data_preflight()` doğrudan | `artifacts/faz11/preflight_no_write_audit.json` | `pytest service/tests/test_faz11_preflight_no_write.py -q` | **pass** (defekt bulunmadı) |

Gerçek before/after dosya sistemi snapshot'ları (boyut+mtime+sha256) ile
kanıtlandı, statik kaynak taraması `app.db.postgres`/`app.db.clickhouse`
referansı olmadığını doğruladı. Kalan iş: `scripts/preflight.py`'nin
host-seviyesi subprocess çağrıları (docker/nvidia-smi) canlı denenmedi
(`not_run` — komutlar inceleme ile salt-okunur).

### Manifest ve telemetry semantiği

| implementation_files | test_files | production_call_path | artifact | test_command | status |
|---|---|---|---|---|---|
| `service/app/ingestion/manifest.py`, `service/app/ingestion/telemetry.py` | `service/tests/test_faz11_manifest.py` | `load_manifest()`, `TelemetrySeries.from_csv()`/`align_timestamp()` (ingest ve preflight'ın ikisinden de çağrılır) | — | `pytest service/tests/test_faz11_manifest.py -q` | **pass** (2 gerçek bug bulundu ve düzeltildi: T1, T2) |

Heading/yaw cross-map olmadığı (tasarım gereği — hiç auto-map kodu yok),
altitude/velocity semantiğinin parse edilen alanda korunduğu, extra
telemetry'nin kaybolmadığı/canonical ile karışmadığı, offset işaretinin
docstring ile eşleştiği testlerle kanıtlandı.

## Run-versioning fault matrix (10/10)

`artifacts/faz11/run_versioning_fault_matrix.json` — 9 nokta gerçek injected-
failure testleriyle `pass`, 1 nokta (`active pointer transaction'ı sırasında`)
gerçek PostgreSQL/psycopg2 close-without-commit rollback semantiğine
dayandığından `pass_by_code_inspection_not_live_run` (fake store bunu
gerçekten kanıtlayamaz — canlı DB gerektirir).

## Pushdown adapter audit

`artifacts/faz11/pushdown_adapter_audit.json` — ClickHouse/Qdrant/pgvector'ın
üçü de gerçek backend-native sorgu/filter nesneleri üretiyor (Python
candidate-ID listesi taşınmıyor); `test_active_pushdown_never_materializes_candidate_ids`
gerçek `app.search.engine.search()` çağrı yolunu test ediyor. NULL semantiği
tutarlılığı `pass_by_code_inspection` (canlı cross-backend equivalence
koşulmadı).

## Streaming memory smoke

`artifacts/faz11/streaming_memory_smoke.json` — gerçek 60s/10fps sentetik
video ile `DECODE_PREFETCH_WINDOWS=5` sınırının 59 pencere boyunca hiç
aşılmadığı ölçüldü (`max_live_window_records=5`, `max_live_pil_images=20`).
`status=pass_synthetic_smoke` — kurum acceptance olarak sunulmuyor.

## Doküman-kod tutarlılığı

`tests/test_faz11_docs_and_notebook.py` (11 test) — `.env.example`'ın her
anahtarının gerçekten okunduğunu (iki yönlü), iki örnek manifest'in gerçek
parser'dan geçtiğini, Colab notebook'unun geçerli nbformat olduğunu ve
`.env.example` ile aynı model revision/source commit'i pinlediğini, her
`docs/*.md` script/manifest referansının gerçekten var olduğunu,
`run_faz11_acceptance.py --help`'in dokümante edilen her bayrağı
içerdiğini, `gc_runs.py`/`preflight.py`'nin gerçek CLI/exit-code
sözleşmesinin dokümanla eşleştiğini doğruluyor.

## Kabul matrisi güncellemesi

`final_acceptance.json` yeniden üretildi (bkz. `docs/FAZ11_FINAL_REPORT.md`).
Nihai durum: **`implementation_complete_hardware_acceptance_pending`** —
bu oturumlar kodu iyileştirdi, yedi gerçek defekti kapattı ve beş alanı
yüzeysellikten derin doğrulamaya taşıdı; hedef kurum donanımı/verisi/canlı
Docker olmadan tam kabul hâlâ mümkün değil.

## Sonraki tur için öncelik sırası

1. `scripts/run_faz11_acceptance.py --live` ile gerçek hedef host'ta canlı
   compose/ingest/health/active-run adımlarını çalıştırmak.
2. `psycopg2-binary` kurulup gerçek PostgreSQL'e karşı migration `--plan`
   denemek (bu host'ta artık mümkün olabilir — Docker Desktop başlatılırsa).
3. Interrupted-resume, pushdown equivalence/scale, UI search (Playwright),
   media playback — beşi de yalnız gerçek hedef ortamda mümkün.
4. F1'in bilinen kalan sınırı: decode-side istisna anında partial
   prefetch-group frame temizliği.
