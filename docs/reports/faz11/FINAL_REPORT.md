# Faz 11 final implementation report (audit + hardening + documentation)

Durum: **`implementation_complete_hardware_acceptance_pending`**
Test edilen kod SHA: `d3fef0e6826c626381a4a2b76bbd397d971b58dc`
Orijinal Faz 11 teslim SHA'sı: `045ea81b83de366b3597c84a051d5c5f039603d5`
Başlangıç SHA (Faz 10): `23eb2a894c9b24f998b05a93c6a33262a860796d`

Bu rapor, orijinal FAZ11 teslimini (`045ea81`) kaynak gerçek kabul etmeyen
iki bağımsız denetim oturumunun sonucudur. Oturum 1 (`02a6c10`, `548698b`)
baseline'ı yeniden doğruladı ve iki kritik streaming defekti buldu/düzeltti.
Oturum 2 (`2ebc476`..`8d0ae5c`) önceki oturumda "spot-checked, deep-dive
pending" bırakılan beş alanı derinlemesine denetledi, run-versioning
10-nokta fault matrix'i ve pushdown adapter audit'ini tamamladı, gerçek
streaming memory smoke kanıtı üretti ve eksiksiz kullanım kılavuzu setini
yazdı. Toplam yedi gerçek defekt bulundu ve düzeltildi (aşağıda §5).

Bu macOS/Windows geliştirme hostlarında Docker daemon, temsili NVIDIA GPU,
kurum verisi ve doğrulanmış model bundle bulunmadığından hedef ortam kabulü
başarılı gösterilmedi. Makine-okunur matris: `artifacts/faz11/final_acceptance.json`.

## 1. Baseline ve denetim ortamları

| Oturum | Host | Docker | GPU | Sonuç |
|---|---|---|---|---|
| Orijinal FAZ11 teslim | macOS/arm64 | daemon kapalı | yok | `045ea81` |
| Denetim oturumu 1 | Windows | daemon kapalı | GT 1030 (4GB, temsili değil) | `02a6c10`, `548698b` |
| Denetim oturumu 2 | Windows (aynı host) | daemon kapalı | GT 1030 (4GB, temsili değil) | `2ebc476`..`8d0ae5c` |

Denetim oturumları farklı bir host'ta (macOS değil, Windows) çalıştığı için
bağımsızlık gerçekti — testler kör kopyalanmadı, gerçekten yeniden
çalıştırıldı. Windows `.venv`'de `clickhouse-connect`, `psycopg` (v3, ama
`psycopg2` değil), `qdrant-client`, `torch` kurulu olduğundan bazı testler
macOS oturumunda skip iken burada gerçekten çalıştı.

## 2. Bulunan ve düzeltilen yedi gerçek defekt

| # | Defekt | Ciddiyet | Commit |
|---|---|---|---|
| F1 | `DECODE_PREFETCH_WINDOWS`/`DB_WRITE_BATCH_SIZE` ölü konfigürasyondu; üç ayar tek `EMBED_BATCH_SIZE`'a indirgenmişti | high | `548698b` |
| F2 | `iter_chunk_windows` kaynak frame'i ve `GeneratorExit`'i açıkça kapatmıyordu (implicit GC'ye güveniyordu) | medium | `548698b` |
| F3 | `ATTN_IMPL=flash_attention_2` seçilince `flash_attn`/GPU capability kontrolü hiç yoktu | medium | `2ebc476` |
| M1 | `_copy_clickhouse_legacy`'de idempotency guard yoktu — ikinci `--apply` ClickHouse satırlarını duplike ederdi | high | `9c8fd73` |
| M2 | Migration retry-after-failure asla activate() edemezdi (`ON CONFLICT DO NOTHING` status'u `validating`'e resetlemiyordu) | high | `9c8fd73` |
| T1 | `telemetry.py` naive iso8601 zaman damgasını `manifest.timezone` yerine hep UTC sayıyordu | medium | `6571306` |
| T2 | Categorical telemetry alanı için interpolation/aggregation hiç doğrulanmıyordu (circular_deg'in aksine) | medium | `6571306` |

Detaylı kanıt: `docs/FAZ11_TRACEABILITY_AUDIT.md`.

## 3. Mimari değişiklikler (orijinal Faz 11 teslimi, oturum 1-2 tarafından korunmuş)

- Kurum defaultu ClickHouse + 512d + native pushdown; Qdrant/pgvector ve dört
  dimension yalnız benchmark override ile açılır. DB portları canonical profilde
  hosta publish edilmez.
- Relative path güvenlikli dataset manifesti ve iki katmanlı read-only preflight;
  bu oturumda preflight'ın **gerçekten** hiçbir yazı yapmadığı gerçek
  dosya sistemi snapshot'larıyla kanıtlandı (`preflight_no_write_audit.json`).
- PyAV/OpenCV streaming decoder chunk+halo ownership ile bounded iterator üretir;
  bu oturumda üç bağımsız ayarın (decode prefetch/embed batch/DB write batch)
  gerçekten bağımsız çalıştığı ve frame lifecycle'ın explicit olduğu düzeltildi.
- Qwen kaynak/model revision'ları pinlidir; hash zincirinin gerçek preflight
  yoluna bağlı olduğu ve `flash_attention_2` seçiminin artık dependency/GPU
  capability kontrolünden geçtiği bu oturumda doğrulandı.
- Run-scoped storage, chunk ledger, finalize/atomic active pointer; bu
  oturumda 10 noktalı injected-failure matrix'i tamamlandı (9/10 gerçek test,
  1/10 kod incelemesiyle).
- Migration'ın gerçekten additive/idempotent olduğu bu oturumda **iki gerçek
  bug bulunup düzeltilerek** kanıtlandı (önceden hiç test yoktu).
- Canonical filter registry ClickHouse/Qdrant/pgvector'a projekte edilir; bu
  oturumda üç adaptörün de gerçek backend-native sorgu inşa ettiği (Python
  candidate listesi taşımadığı) kaynak seviyesinde doğrulandı.
- Media path containment, ffmpeg arg-list, signed URL — bu oturumda hiçbir
  defekt bulunmadı; yalnız test kapsamı (symlink escape, encoded traversal,
  token leak, cross-path signature reuse) genişletildi.
- **Yeni bu oturumda:** `docs/USER_GUIDE.md`, `docs/OPERATOR_QUICKSTART.md`,
  `docs/END_USER_GUIDE.md`, `docs/DATASET_ONBOARDING_GUIDE.md`,
  `datasets/example_institution.yaml`, `docs/COLAB_RUNBOOK.md`,
  `notebooks/08_colab_portable_runner.ipynb`,
  `docs/TARGET_ENVIRONMENT_ACCEPTANCE.md`, `scripts/run_faz11_acceptance.py`
  — hepsi gerçek CLI/config/UI ile çapraz doğrulanan (`tests/test_faz11_docs_and_notebook.py`)
  eksiksiz kullanım kılavuzu seti.

## 4. Değişen/eklenen dosyalar (bu iki denetim oturumu)

| Dosya/grup | Neden | Doğrulama |
|---|---|---|
| `service/app/ingestion/ingest.py`, `video.py` | Bounded 3-aşamalı pipeline (F1) ve frame lifecycle (F2) | `service/tests/test_faz11_pipeline_bounds.py` (yeni, 7 test) |
| `service/app/preflight.py` | `flash_attention_2` dependency/GPU gate (F3) | `service/tests/test_faz11_preflight_model.py` (yeni) |
| `service/app/db/migrations.py` | ClickHouse idempotency + retry-after-failure (M1, M2) | `service/tests/test_faz11_migrations.py` (yeni — önceden hiç yoktu) |
| `service/app/ingestion/telemetry.py`, `manifest.py` | Timezone-naive timestamp fix + categorical validation (T1, T2) | `service/tests/test_faz11_manifest.py` (genişletildi) |
| `service/tests/test_faz11_media_ui.py`, `test_faz11_security.py` | Test kapsamı genişletme (defekt yok) | 5 yeni test |
| `service/tests/test_faz11_preflight_no_write.py` | Read-only kanıtı (yeni) | 5 test, gerçek fs snapshot |
| `service/tests/test_faz11_run_versioning_fault_matrix.py` | 10-nokta fault matrix'i tamamlama | 6 yeni test |
| `scripts/streaming_memory_smoke.py`, `tests/test_faz11_streaming_memory_smoke.py` | Gerçek bounded-memory ölçümü | Gerçek instrumented çalıştırma |
| `docs/USER_GUIDE.md`, `OPERATOR_QUICKSTART.md`, `END_USER_GUIDE.md`, `DATASET_ONBOARDING_GUIDE.md` | Eksiksiz kullanım kılavuzu | `tests/test_faz11_docs_and_notebook.py` |
| `docs/COLAB_RUNBOOK.md`, `notebooks/08_colab_portable_runner.ipynb` | Portable Colab embedding yolu | nbformat validate + pin match testi |
| `docs/TARGET_ENVIRONMENT_ACCEPTANCE.md`, `scripts/run_faz11_acceptance.py` | Tek-komut hedef ortam kabul zinciri | Bu host'ta çalıştırıldı (graceful not_run) |
| `artifacts/faz11/migration_contract_audit.json`, `preflight_no_write_audit.json`, `run_versioning_fault_matrix.json`, `pushdown_adapter_audit.json`, `streaming_memory_smoke.json`, `traceability_audit.json` | Yeni denetim kanıtları | JSON şema doğrulaması + `final_artifact_audit` adımı |

## 5. Çalıştırılan komutlar (bu iki oturum boyunca)

```bash
git rev-parse HEAD
git status --short
git log -15 --oneline
PYTHONPATH=service .venv/Scripts/python -m pytest service/tests/ -q -p no:cacheprovider
.venv/Scripts/python -m pytest tests/ -q -p no:cacheprovider
find . -name "*.py" -not -path "./.venv/*" -not -path "./.testdeps/*" -print0 | xargs -0 .venv/Scripts/python -m py_compile
git diff --check
docker compose --env-file .env.example -f docker-compose.yml config
docker compose --env-file .env.example -f docker-compose.yml -f docker-compose.gpu.yml config
docker compose --env-file .env.example -f docker-compose.yml -f docker-compose.benchmark.yml config
docker compose --env-file .env.example -f docker-compose.yml -f docker-compose.benchmark.yml -f docker-compose.debug.yml config
python scripts/streaming_memory_smoke.py --decode-prefetch-windows 5 --embed-batch-size 2 --db-write-batch-size 8
python scripts/run_faz11_acceptance.py --env-file .env.example --output artifacts/faz11/target_acceptance_test.json
docker info --format {{.ServerVersion}}
nvidia-smi
```

Commit kapıları (oturum 1 + oturum 2):

```text
02a6c10 faz11-audit: add independent traceability audit
548698b faz11-streaming: enforce bounded decode/embed/write stages and explicit frame ownership
2ebc476 faz11-model: verify bundle hash/provenance chain and add flash_attention_2 preflight gate
9c8fd73 faz11-migration: prove additive idempotent migration contracts
184afae faz11-security: close test-coverage gaps in media path containment and signed URLs
a70a58e faz11-preflight: prove read-only preflight behavior with real filesystem snapshots
6571306 faz11-telemetry: fix timezone-naive timestamp handling and validate categorical interpolation
9dc7b92 faz11-runs: complete the 10-point injected-failure matrix
d52f0d3 faz11-search: complete backend pushdown adapter audit
347eb52 faz11-streaming: add real streaming_memory_smoke evidence
7cf9b9b faz11-docs: add operator, end-user, and dataset onboarding guides
d3fef0e faz11-colab: add portable Colab embedding workflow / faz11-acceptance / faz11-docs (doc-code tests)
8d0ae5c faz11-final: regenerate traceability audit with per-area status matrix
```

## 6. Kabul matrisi

Bkz. `artifacts/faz11/final_acceptance.json` — **28 PASS, 0 FAIL, 0 BLOCKED,
12 NOT RUN**. Her NOT RUN satırının `reason`, `required_command` ve
`expected_environment` alanı doludur. Yeni bu oturumda eklenen PASS
satırları: `flash_attention_2_dependency_gate`, `run_versioning_fault_matrix_10_point`,
`migration_idempotency_contract`, `streaming_bounded_memory_contract`,
`pushdown_adapter_production_call_path`, `operator_end_user_dataset_onboarding_documentation`,
`colab_portable_workflow_documentation`, `target_environment_acceptance_runner`,
`doc_code_consistency`.

## 7. Hedef ortamda tamamlanacak kabul sırası

Artık tek bir orkestre edilmiş komut var:

```bash
python scripts/run_faz11_acceptance.py \
  --dataset datasets/kurum.yaml --env-file .env --live \
  --output artifacts/faz11/target_acceptance.json
```

Detay: `docs/TARGET_ENVIRONMENT_ACCEPTANCE.md`. Bu sıra tamamlanana kadar
durum bilinçli olarak `implementation_complete_hardware_acceptance_pending`
kalır.

## 8. Doğrulanmayanlar

- Gerçek GPU inference, gerçek kurum videosu/telemetrisi, canlı PostgreSQL/
  ClickHouse/Qdrant, Playwright UI testi, gerçek Colab GPU çalıştırması —
  hiçbiri bu iki oturumda mevcut değildi; hiçbiri mock/tahminle "geçti"
  gösterilmedi.
- Run-versioning fault matrix'inin 10. noktası (active pointer transaction
  sırasında hata) yalnız kod incelemesiyle doğrulandı — gerçek PostgreSQL
  rollback-on-close semantiğine dayanır, fake store bunu kanıtlayamaz.
- Pushdown NULL semantiği tutarlılığı kod incelemesiyle doğrulandı; canlı
  cross-backend equivalence koşulmadı.

## 9. Rollback/migration

Değişiklik yok — `docs/RUN_VERSIONING.md` ve `docs/OPERATIONS.md`'deki
mevcut rollback/GC prosedürleri geçerlidir. Migration artık idempotent
retry'i destekliyor (M1/M2 düzeltmesi) — bu OPERATIONS.md'ye eklendi.

## 10. Son durum

```text
implementation_complete_hardware_acceptance_pending
```

Yalnız `scripts/run_faz11_acceptance.py --live` gerçek hedef NVIDIA Linux
ortamında tüm adımları PASS verince `fully_accepted_on_target_environment`e
yükseltilebilir.
