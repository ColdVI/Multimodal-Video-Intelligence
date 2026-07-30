# Faz 7 blokerleri

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
