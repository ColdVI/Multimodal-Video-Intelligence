# Faz 7 blokerleri

## Faz 11 current environment blockers

- **Canlı container smoke NOT RUN:** 30 Temmuz 2026 doğrulamasında Docker
  daemon'a `unix:///Users/anil/.docker/run/docker.sock` üzerinden bağlanılamadı.
  Compose parse kontrolleri geçti; hiçbir canlı DB/API sonucu üretilmedi.
  Docker Desktop/Engine başladıktan sonra kurum profili için:
  `docker compose --env-file .env up -d --build` ve ardından
  `docker compose exec -T api python -m app.ingestion.load_dataset --dataset auair`.
- **GPU acceptance NOT RUN:** bu macOS/arm64 ortamında `nvidia-smi` bulunmuyor.
  Qwen GPU smoke gerçek komutla çalıştırıldı ve
  `artifacts/faz11/gpu_smoke.json` içinde `status=not_run` yazdı; gerçek video
  throughput ve peak VRAM değeri uydurulmadı. Verified bundle ve kurum manifesti
  bulunan NVIDIA Linux makinesinde:
  `PYTHONPATH=service python scripts/gpu_smoke.py --dataset datasets/kurum.yaml --data-root /kurum/data --output artifacts/faz11/gpu_smoke.json --windows 10`.
- **CapERA data acceptance NOT RUN:** güncel çalışma ağacında
  `data/downloads/capera/CapERA_DATASET_train.json` ve
  `artifacts/embeddings/capera_2048.npy` yok. Data-dependent service testleri
  yalnız dosyalar yokken açık gerekçeyle skip edilir; dosyalar geldiğinde aynı
  testler otomatik çalışır. Bu eksiklik kurumun generic manifest yolunu bloke
  etmez.
- **Faz 11 persisted-schema acceptance NOT RUN:** migration `--plan` komutu bu
  macOS Python 3.13 test ortamında pinli `psycopg2-binary==2.9.9` için wheel
  bulunmadığı (`pg_config` yok) ve Docker daemon kapalı olduğu için canlı volume
  şemasını okuyamadı. Kod/test/dry-run sözleşmesi tamamlandı; artifact
  `artifacts/faz11/schema_migration_report.json`. Servis container'larında pinli
  Python 3.11 bağımlılıkları ve DB'ler healthy iken önce `--plan`, operator
  backup/snapshot onayından sonra ayrıca `--apply` çalıştırılmalıdır.

## Faz 8 open blockers

- A1/quality is not ready: Colab-produced real artifacts for 1391 items and
  6955 queries are absent from artifacts/embeddings. Cached CapERA ingest,
  the 1391 x 4 x 3 DB count gate, and T8 are not reported as completed.
  A0/system and T1-T7 remain independent.
- ~~Playwright is pinned in the test requirements, but a Chromium binary was
  not installed and verified. T10 skips with this explicit reason while
  other suites continue.~~ **Resolved 2026-07-30 (Faz 9):** `pip install
  playwright==1.55.0` + `playwright install --with-deps chromium` run in the
  dev venv; `service/tests/test_t10_ui.py` now drives a real Chromium session
  against the live `api`+`ui` containers and no longer skips (7/7 passing).

Kritik yol blokeri yok: üç DB, API ve UI yerel Docker'da healthy; AU-AIR yükleme ve arama çalışıyor.

- Tam L2 benchmark (20 sabit sorgu × 10 tekrar) henüz çalıştırılmadı. Denenen/olan: aynı 150-konfigürasyon matrisi `--smoke` ile hatasız tamamlandı. Neden bırakıldı: sabah sistemi ayağa kaldırma kritik yolunda uzun koşum değil. Sıradaki adım: `python -m app.bench.runner` komutunu `--smoke` olmadan çalıştırmak.
- Milvus opsiyonel compose profili uçtan uca yüklenmedi. Denenen/olan: adaptör ve profil yazıldı; zorunlu ClickHouse+Qdrant+pgvector doğrulandı. Neden bırakıldı: talimatta açıkça feda edilebilir. Sıradaki adım: `--profile milvus` ile IVF_FLAT koleksiyon ingest'i eklemek/doğrulamak.
- CapERA ve SeaDronesSee bu koşumda yüklenmedi. Denenen/olan: savunmacı loader yolları yazıldı; hazır 1.866 AU-AIR kritik hattı kullanıldı. Neden bırakıldı: yeni veri indirmesi kritik yol dışında. Sıradaki adım: resmi annotation/video dosyalarını `data/research/<dataset>/` altına koyup ilgili loader'ı çalıştırmak.
- `cached` mod bilinmeyen serbest metni model yüklemeden embed edemez. Denenen/olan: sessiz sentetik fallback yasaklandı; yalnız Colab'ın ürettiği `query_embeddings.json` kabul edilir. Sıradaki adım: ürün kullanımında gerçek text-embedding GPU servisi kullanmak veya sorgu cache'ini Colab'da genişletmek.

## Faz 9 (UI redesign) open items

Kritik yol blokeri yok — `service/tests/test_t10_ui.py` (7/7) ve repo geneli
`403 passed, 1 skipped` canlı Docker container'lara karşı doğrulandı.

- `hybrid_text` cold-start progress ekranı (§2.9) gerçek canlı veriyle ekran görüntüsü
  olarak üretilemedi: bu oturumda `EMBEDDING_MODE=synthetic` çalışıyordu, `hybrid_text`'e
  geçmek modeli indirip yüklemeyi gerektirir (kritik yol dışında). Kod yolu
  (`components.loading_state(cold_start=True)`, `run_search`'teki `/health` kontrolüyle
  tetikleniyor) yazıldı ve ölçülmüş gerçek sabitleri (28.0s model + 43.2s ilk sorgu +
  0.74s warm p50, `docs/BLOCKERS.md`'nin bu dosyadaki Faz 8 notundan) kullanıyor, ama
  ekranı bizzat `hybrid_text` modunda görmek istenirse: `EMBEDDING_MODE=hybrid_text`
  ile `docker compose -f docker-compose.faz7.yml up -d --build api` çalıştırıp ilk
  aramayı yapmak yeterli.
- Bu oturumda yeniden build edilen `video-search-faz7-api`/`video-search-faz7-ui`
  imajları yalnızca yerel Docker'da; hiçbir registry'ye push edilmedi (kapsam dışı,
  proje zaten yalnız yerel compose kullanıyor).

## Faz 10 (gerçek embedding'e geçiş) open blockers

- **Kritik yol bloke:** `readiness_check.py --profile quality` hâlâ `A1.1 FAIL`
  veriyor — `artifacts/embeddings/{capera_2048.npy, capera_ids.parquet,
  capera_queries_2048.npy, capera_query_ids.parquet, query_embeddings.json,
  embedding_manifest.json}` yerelde yok. Bu talimatın §2'sidir (kullanıcının
  Colab'da GPU runtime ile `notebooks/07_colab_embedding_production.ipynb`'yi
  çalıştırıp indirdiği ZIP'i açması) ve Codex/Claude tarafından yapılamaz.
  `data/downloads/capera/CapERA_DATASET_{train,test}.json` yerelde zaten var;
  eksik olan tek şey Colab'ın ürettiği embedding ZIP'i. Talimatın kendi kuralı
  gereği (§3.1: "A1.1 FAIL ise dur, devam etme") §3.2-§3.3, §3.5-§3.8 bu
  oturumda başlatılmadı.
- Notebook tarafında ek iş **yok**: `notebooks/07_colab_embedding_production.ipynb`
  (Faz 7/8'de yazıldı) hem 1391 video hem 6955 caption/query embedding'ini
  doğru `.npy`+`.parquet` formatında üretip zip'liyor (hücre 4-11). Önceki bir
  oturumda "notebook 02 + NDJSON dönüştürme script'i gerekir" denilmişti; bu
  artık geçersiz — notebook 02 zaten `notebooks/_archive/`'e taşındı (Faz 7
  aşama 1, bkz. `docs/PROGRESS.md`).

## Faz 10 §3.4 (vector_provenance) — durum

Kritik yol bloke yok. `datasets.vector_provenance` kolonu eklendi, ingest bunu
`settings.embedding_mode`'a göre yazıyor, `/health?dataset_id=`, `/stats`,
`/search` bunu additive alan olarak dönüyor, `mode_details()` artık önce
dataset'in kendi provenance'ına bakıyor. Doğrulanan: yalnız **auair** yarısı
(`vector_provenance='synthetic'` + danger banner, canlı Docker'a karşı test
edildi: `test_additive_fields.py::test_mixed_provenance_database_reports_auair_as_synthetic_with_danger_banner`).
**capera→'real'+success yarısı doğrulanamadı** — capera henüz ingest edilmedi
(yukarıdaki A1.1 blokerine bağlı). Colab ZIP'i geldiğinde §3.2 ingest'i
tamamlanınca bu testin capera koluyla tamamlanması gerekiyor.
