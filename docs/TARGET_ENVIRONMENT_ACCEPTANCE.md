# Hedef ortam kabul runbook'u

`scripts/run_faz11_acceptance.py`, tam kurum kabul zincirini sıralı çalıştırıp
tek bir makine-okunur sonuç üretir: `artifacts/faz11/target_acceptance.json`.
Hiçbir adım gerçekten denenmeden `pass` işaretlenmez; donanım/veri/Docker
eksikse adım `not_run` olur (`reason`/`required_command`/`expected_environment`
ile). Script hiçbir yıkıcı işlem yapmaz — volume silmez, `down -v` veya
`git reset/clean` çalıştırmaz.

## Çalıştırma

Yalnız salt-okunur/statik kontroller (Docker servislerini başlatmadan):

```bash
python scripts/run_faz11_acceptance.py \
  --dataset datasets/kurum.yaml --env-file .env \
  --output artifacts/faz11/target_acceptance.json
```

Servisleri gerçekten başlatıp canlı ingest/health/media adımlarını da
denemek için (yalnız gerçek hedef host'ta, operatör kararıyla):

```bash
python scripts/run_faz11_acceptance.py \
  --dataset datasets/kurum.yaml --env-file .env --live \
  --output artifacts/faz11/target_acceptance.json
```

`--live` verilmezse Compose başlatma, health check, gerçek ingest ve
active-run doğrulama adımları güvenli biçimde `not_run` kalır — script
varsayılan çalıştırmada hiçbir container başlatmaz.

## Adımlar

| # | id | Ne yapar |
|---|---|---|
| 1 | `git_state` | `git status --short` + `git rev-parse HEAD` |
| 2 | `host_info` | platform/Python sürümü |
| 3 | `nvidia_driver` | `nvidia-smi --query-gpu=...` |
| 4 | `docker_daemon` | `docker info` |
| 5 | `compose_config` | 4 Compose kombinasyonunu `config` ile parse eder |
| 6 | `secure_credentials` | `.env`'de kalan `CHANGE_ME_*` placeholder'ı arar |
| 7 | `model_bundle_hash` | `verify_bundle()` ile gerçek hash zincirini doğrular |
| 8 | `dataset_preflight` | `scripts/preflight.py --dataset ...` çalıştırır |
| 9 | `migration_plan` | `scripts/migrate_faz11_schema.py --plan` çalıştırır |
| 10 | `compose_up` | (`--live`) Compose'u `up -d --build` ile başlatır |
| 11 | `health_check` | (`--live`) `GET /health` |
| 12 | `gpu_smoke` | `scripts/gpu_smoke.py --windows 10` çalıştırır |
| 13 | `real_ingest` | (`--live`) container içinde gerçek ingest |
| 14 | `interrupted_resume` | operatör-gözetimli kesinti gerektirir; her zaman `not_run` (otomatikleştirilmedi — canlı bir run'ı denetimsiz kesmek riskli) |
| 15 | `active_run_verification` | (`--live`) `GET /stats` |
| 16 | `pushdown_equivalence` | temsili ölçekte aktif run gerektirir |
| 17 | `scale_diagnostics` | temsili ölçekte aktif run gerektirir |
| 18 | `ui_search` | canlı UI/API + Playwright Chromium gerektirir |
| 19 | `media_playback` | aktif run + gerçek local MP4 gerektirir |
| 20 | `final_artifact_audit` | tüm zorunlu `artifacts/faz11/*.json` dosyalarının var olduğunu doğrular |

14, 16, 17, 18, 19 kasıtlı olarak `--live` ile bile otomatik `not_run`
kalır — bunlar ya operatör gözetimi (14) ya da bu script'in kapsamadığı ayrı
araçlar (Playwright, `app.search.equivalence` CLI) gerektirir; gerekli tam
komut her birinin `required_command` alanında yazılıdır.

## Çıktı şeması

Her adım:

```json
{
  "id": "...", "status": "pass|fail|blocked|not_run",
  "started_at": "...", "finished_at": "...", "command": "...",
  "evidence": "...", "reason": "...", "expected_environment": "..."
}
```

`overall_status`:

```text
implementation_incomplete                            — herhangi bir adım fail
implementation_complete_hardware_acceptance_pending   — fail yok, en az bir not_run
fully_accepted_on_target_environment                  — tüm adımlar pass
```

**Not:** bu `overall_status`, o **tek çalıştırmanın** sonucudur — örneğin
`.env.example` gibi kasıtlı placeholder içeren bir dosyayla çalıştırılırsa
`secure_credentials` haklı olarak `fail` verir ve genel durum
`implementation_incomplete` görünür; bu kod eksikliği değil, ortam henüz
yapılandırılmadığı anlamına gelir. Gerçek kabul iddiası için script'i her
zaman gerçek, tamamlanmış bir `.env` ve gerçek kurum verisiyle, `--live` ile
çalıştırın.

## Kabul iddiası kuralları

- Bir adımın `pass` olması için script'in o komutu **gerçekten çalıştırmış**
  olması gerekir; asla tahminle işaretlenmez.
- GPU/Docker/veri yoksa ilgili adım dürüstçe `not_run` kalır — `pass` veya
  `pass_synthetic` gibi bir etiketle gizlenmez.
- `docs/FAZ11_FINAL_REPORT.md`'deki nihai durum yalnız bu script'in gerçek
  hedef ortamda ürettiği `target_acceptance.json` ile
  `fully_accepted_on_target_environment` olarak güncellenebilir.
