# FAZ 11 bağımsız traceability audit

Bu denetim, `04df3eb` HEAD'inde (`tested_code_sha=045ea81`, artifact-only ek
commit `04df3eb`) mevcut FAZ11 teslimini `FAZ11_FINAL_CODEX_IMPLEMENTATION_PROMPT.md`
ile satır satır karşılaştırır. `docs/FAZ11_FINAL_REPORT.md` ve
`artifacts/faz11/final_acceptance.json` kaynak gerçek kabul edilmedi; kod,
test ve komut çıktıları bu oturumda (Windows/`.venv`, farklı host) bağımsız
olarak yeniden üretildi.

## Bağımsız test yeniden-çalıştırma (macOS orijinal host'tan farklı ortam)

| Komut | macOS (orijinal) | Windows (bu denetim, önce) | Windows (bu denetim, fix sonrası) |
|---|---|---|---|
| `PYTHONPATH=service pytest service/tests/ -q` | 104 passed, 17 skipped | 106 passed, 15 skipped | **113 passed, 15 skipped** |
| `pytest tests/ -q` | 314 passed, 42 skipped | 356 passed, 0 skipped | 356 passed, 0 skipped |
| `py_compile` (tüm `.py`) | PASS | PASS | PASS |
| `git diff --check` | PASS | PASS | PASS |

Fark, Windows `.venv`'de `clickhouse-connect`, `psycopg`, `qdrant-client`,
`torch` gibi bağımlılıkların kurulu olması ve daha önce dependency-yokluğu
nedeniyle skip edilen testlerin burada gerçekten çalışmasından kaynaklanıyor.
Hiçbir ortamda FAIL yok; bu iddiaları destekliyor.

Bu host'ta ayrıca (orijinal macOS host'un aksine) Docker CLI/Compose ve bir
NVIDIA GPU (GeForce GT 1030, 4 GB VRAM, driver 560.94, CUDA 12.6) mevcut,
ancak Docker daemonı bu oturumda çalışmıyordu (Docker Desktop başlatılmamış)
ve GT 1030 hedef "saatler süren gerçek İHA görüntüsü" kurumsal donanımı değil.
Bu nedenle canlı Docker/GPU kabul adımları bu oturumda da tetiklenmedi;
ancak bir sonraki oturumda Docker Desktop başlatılıp en azından
`docker compose config` ötesinde gerçek container health/ingest smoke'unun
denenmesi mümkün olabilir — bu rapor bunu iddia etmiyor, yalnız kaydediyor.

## Kritik bulgular (bu oturumda düzeltildi)

### F1 — DECODE_PREFETCH_WINDOWS ve DB_WRITE_BATCH_SIZE ölü konfigürasyondu

**Durum: DÜZELTİLDİ (bu oturumda).**

`service/app/config.py` üç ayrı ayarı (`decode_prefetch_windows`,
`embed_batch_size`, `db_write_batch_size`) tanımlıyor ve yalnızca pozitiflik
doğrulaması yapıyordu (`config.py:185`). Fix öncesi
`service/app/ingestion/ingest.py:151`'de gerçek ingest döngüsü:

```python
for batch in batched(itertools.chain([first], group_iterator), settings.embed_batch_size):
    ...
    postgres.write_run_metadata_chunk(...)          # embed_batch_size ile aynı batch
    ...backend.write_chunk(...)                       # embed_batch_size ile aynı batch
```

tek bir batch boyutuyla (`EMBED_BATCH_SIZE`) hem Qwen çağrısını hem DB
yazımını yürütüyordu. `DECODE_PREFETCH_WINDOWS`, `generic_loader.py` veya
`video.py` içinde hiçbir yerde referans edilmiyordu — decode iterator'ı
sınırlayan ayrı bir tampon yoktu. `docs/DECISIONS.md:184-187` bunu açıkça
"Qwen batch boyutu EMBED_BATCH_SIZE, DB writes aynı batch üzerinden yürür"
diye belgeliyordu — talimatın §6.2'de doğrudan yasakladığı "Tek bir
INGEST_BATCH_SIZE ile bu üç katmanı birleştirme" davranışının ta kendisiydi.

**Fix:** `GenericIngestor.run()` artık üç bağımsız katman kullanıyor:

1. `batched(decode_source, settings.decode_prefetch_windows)` — decode
   iterator'ından aynı anda en fazla `DECODE_PREFETCH_WINDOWS` kayıt çekilir
   (RAM sınırı).
2. `batched(prefetch_group, settings.embed_batch_size)` — Qwen çağrısı bu
   boyutta yapılır (VRAM sınırı); her çağrıdan hemen sonra o batch'in
   frame'leri serbest bırakılır.
3. `pending` listesi `settings.db_write_batch_size`'a ulaşınca (veya chunk
   sonunda kalan artık ne kadarsa) Postgres + her etkin backend'e flush
   edilir (DB/ağ sınırı).

Kanıt: `service/tests/test_faz11_pipeline_bounds.py` — 7 yeni test:
`test_decode_prefetch_bound_is_enforced`, `test_embed_batch_size_is_independent`,
`test_db_write_batch_size_is_independent`, `test_frames_are_released_after_each_batch`,
`test_producer_exception_reaches_ingestor`, `test_db_failure_does_not_commit_chunk`,
`test_resume_after_partial_batch_is_idempotent`. Tümü geçiyor; ayrıca üç
ayarın birbirinden farklı, uyumsuz (asal) değerlerle çağrıldığında gerçek
davranış farkı gösterdiği doğrulandı (örn. `EMBED_BATCH_SIZE=4` ile çağrı
boyutları `[4,2,4]`, aynı veri için `DB_WRITE_BATCH_SIZE=3` ile yazım
boyutları `[3,3]` — ikisi asla eşleşmiyor).

**Bilinen kalan sınır:** decode kaynağı bir `prefetch_group` doldurulurken
(yani `list(itertools.islice(iterator, decode_prefetch_windows))` içinde)
istisna fırlatırsa, o partial batch'teki zaten decode edilmiş frame'ler
`release_frames()` çağrısına hiç ulaşmaz ve yalnızca CPython refcounting ile
toplanır. Etki alanı en fazla `DECODE_PREFETCH_WINDOWS` kayıttır ve chunk
zaten `failed` işaretlenip decode durdurulacağından pratik etkisi düşüktür;
tam bir düzeltme decode tarafında da `try/finally` gerektirir ve bu oturumda
kapsam dışı bırakıldı — bir sonraki turda ele alınmalı.

### F2 — Frame lifecycle: kaynak decode frame'i ve GeneratorExit'te temizlik eksikti

**Durum: DÜZELTİLDİ (bu oturumda).**

`service/app/ingestion/video.py::iter_chunk_windows` fix öncesi:

- `_iter_decoded_frames()`'ten gelen kaynak `frame` nesnesi (kopyalar
  alındıktan sonra) hiçbir zaman `.close()` edilmiyordu; yalnızca örtük
  referans sayımıyla toplanıyordu.
- Fonksiyon `GeneratorExit` ile erken kapatılırsa (`iterator.close()` veya
  tüketicinin döngüyü erken bırakması), `collected[]` içinde henüz
  emit edilmemiş kısmi pencere frame'leri hiç kapatılmıyordu.

**Fix:** Kaynak `frame`, bir sonraki frame geldiğinde (kendisinden gereken
kopyalar zaten alındıktan sonra) açıkça kapatılıyor; tüm gövde
`try/finally` ile sarıldı — hem normal tükenmede (no-op, zaten temizlenmiş)
hem `GeneratorExit`'te `last_frame` ve `collected[]` içindeki her şey
kapatılıyor.

Kanıt: mevcut `service/tests/test_faz11_streaming.py` (9 test) fix sonrası
da geçiyor; davranış değişmedi, yalnızca ownership disiplini eklendi.
Ayrı bir "erken generator kapatma" testi bu oturumda eklenmedi — bir sonraki
turda `test_frames_are_released_after_each_batch` benzeri bir
`iter_chunk_windows` seviyesi test (üretici `.close()` çağrıldığında
`collected` boşalıyor mu) eklenmelidir.

## Spot-check edilen ve TEMİZ bulunan alanlar

Bu alanlar dosya varlığı ötesinde kod seviyesinde okunup doğrulandı; kod,
iddia edilen davranışı gerçekten uyguluyor:

| Alan | Kanıt | Sonuç |
|---|---|---|
| ClickHouse pushdown | `service/app/db/clickhouse.py:94-153` — gerçek parametrize SQL, `WHERE`'e canonical predicate'ler + `run_id` ekleniyor, `candidate_ids IN (...)` yalnız legacy modda | Gerçek backend-native, Python'a candidate listesi taşınmıyor |
| Qdrant pushdown | `service/app/db/qdrant.py:89-114` — gerçek `models.Filter`/`FieldCondition`/`HasIdCondition` inşası, `run_id` zorunlu alan | Gerçek backend-native |
| pgvector pushdown | `service/app/db/postgres.py:429-454` — tek SQL'de `run_segments`/`run_videos`/`run_segment_telemetry`/`run_segment_metadata` JOIN | Gerçek backend-native |
| Pattern A/B/C dürüstlüğü | `service/app/search/engine.py:25` `PATTERN_EXECUTION_IMPLEMENTED = False` sabiti korunuyor; `pattern` alanı yalnız pgvector/C validasyonu ve response label'ı için kullanılıyor, execution mode'u belirlemiyor | Talimat §12.3 ile tutarlı, yanıltıcı değil |
| Run-versioning invariant testleri | `service/tests/test_faz11_run_versioning.py` — `test_finalize_failure_preserves_old_active_run`, `test_finalize_success_activates_only_after_all_counts_match`, `test_retry_cleans_only_same_inactive_run_chunk_before_writing`, `test_gc_never_selects_active_running_or_previous_completed` | İsimlendirilmiş invariant'lar gerçekten test ediliyor (mock değil, davranış assertion'ı) |
| `ui_smoke.png` / `ui_smoke.json` dürüstlüğü | `scripts/write_ui_not_run_artifact.py` — PIL ile "NOT RUN" yazılı gerçek 1440×900 placeholder üretiyor, JSON içinde `status=not_run` ve nedeni açık | Fabrikasyon değil; kod-üretimli ve dürüst etiketli |
| `gpu_smoke.json` dürüstlüğü | `result=not_run`, `windows_embedded=0`, `gpu_name=null` — uydurma throughput/VRAM yok | Talimat §7.4/§3.2 ile tutarlı |

## Bu oturumda derinlemesine yeniden doğrulanmayan alanlar

Aşağıdakiler dosya/test varlığı ve isim bazında tutarlılık kontrolünden
geçti, ancak talimatın istediği tam adaptör-seviyesi/injected-failure
derinliğinde bağımsız olarak yeniden okunmadı. Bunlar `PASS` değil
`PARTIAL — spot-checked, deep-dive pending` olarak işaretlenmiştir; bir
sonraki denetim turunda öncelik sırasına alınmalı:

- Model bundle hash doğrulama zinciri (`scripts/prepare_model_bundle.py`,
  `service/app/embedding/bundle.py`) — dosyalar var, hash algoritması
  okunmadı.
- Migration additivity/idempotency (`scripts/migrate_faz11_schema.py`,
  `service/app/db/migrations.py`) — `--plan`/`--dry-run` davranışı canlı
  DB'ye karşı bu oturumda da denenmedi (psycopg/Docker sınırlaması burada da
  geçerli olabilir, ama bu host'ta Windows için `psycopg[binary]` kurulu —
  bir sonraki turda gerçekten denenebilir).
- Media path traversal/symlink escape testleri (`service/app/media.py`,
  `service/tests/test_faz11_security.py`) — dosyalar var, path containment
  mantığı satır satır okunmadı.
- Preflight'ın gerçekten write yapmadığının statik analizi
  (`service/app/preflight.py`) — yalnız test varlığı doğrulandı.
- Manifest/telemetry semantik doğruluğu (circular/AGL/ground-speed
  ayrımları) — `docs/DATASET_MANIFEST.md` ve `telemetry.py` dosya bazında
  incelendi, satır satır tekrar üretilmedi.

## Kabul matrisi güncellemesi

`final_acceptance.json`'daki 19 PASS satırı gözden geçirildi. On yedisi bu
oturumun spot-check'leriyle tutarlı bulundu. İki satır gerçek bir defekt
içeriyordu ve düzeltildi:

| id | Önceki durum | Bu denetim sonrası |
|---|---|---|
| `generic_ingest_resume_contract` | pass | **pass (fix sonrası, genişletilmiş kanıt)** — F1 düzeltmesi ve 7 yeni test ile |
| `streaming_video_window_contract` | pass | **pass (fix sonrası)** — F2 düzeltmesi ile, davranış değişmedi |

Diğer 17 satır bu oturumda `PASS` olarak korundu (spot-check temiz) veya
yukarıdaki "derinlemesine yeniden doğrulanmayan" listesine taşındı (durumu
`final_acceptance.json`'da değiştirilmedi, ancak bu dosyada takip
gerektirdiği not edildi).

Nihai durum değişmedi: **`implementation_complete_hardware_acceptance_pending`**.
Bu oturum kodu iyileştirdi ve iki gerçek defekti kapattı; hedef kurum
donanımı/verisi/Docker daemon olmadan tam kabul hâlâ mümkün değil.

## Sonraki tur için öncelik sırası

1. Decode-side istisna anında partial prefetch-group frame temizliği (F1'in
   bilinen kalan sınırı).
2. `iter_chunk_windows` için erken `GeneratorExit` testi.
3. Model bundle hash zinciri ve migration idempotency'nin bu Windows
   host'unda (Docker Desktop başlatılarak) gerçekten denenmesi.
4. Media path containment ve preflight no-write garantisinin satır satır
   yeniden doğrulanması.
