# Faz 7 blokerleri

## Faz 8 open blockers

- A1/quality is not ready: Colab-produced real artifacts for 1391 items and
  6955 queries are absent from artifacts/embeddings. Cached CapERA ingest,
  the 1391 x 4 x 3 DB count gate, and T8 are not reported as completed.
  A0/system and T1-T7 remain independent.
- Playwright is pinned in the test requirements, but a Chromium binary was
  not installed and verified. T10 skips with this explicit reason while
  other suites continue.

Kritik yol blokeri yok: üç DB, API ve UI yerel Docker'da healthy; AU-AIR yükleme ve arama çalışıyor.

- Tam L2 benchmark (20 sabit sorgu × 10 tekrar) henüz çalıştırılmadı. Denenen/olan: aynı 150-konfigürasyon matrisi `--smoke` ile hatasız tamamlandı. Neden bırakıldı: sabah sistemi ayağa kaldırma kritik yolunda uzun koşum değil. Sıradaki adım: `python -m app.bench.runner` komutunu `--smoke` olmadan çalıştırmak.
- Milvus opsiyonel compose profili uçtan uca yüklenmedi. Denenen/olan: adaptör ve profil yazıldı; zorunlu ClickHouse+Qdrant+pgvector doğrulandı. Neden bırakıldı: talimatta açıkça feda edilebilir. Sıradaki adım: `--profile milvus` ile IVF_FLAT koleksiyon ingest'i eklemek/doğrulamak.
- CapERA ve SeaDronesSee bu koşumda yüklenmedi. Denenen/olan: savunmacı loader yolları yazıldı; hazır 1.866 AU-AIR kritik hattı kullanıldı. Neden bırakıldı: yeni veri indirmesi kritik yol dışında. Sıradaki adım: resmi annotation/video dosyalarını `data/research/<dataset>/` altına koyup ilgili loader'ı çalıştırmak.
- `cached` mod bilinmeyen serbest metni model yüklemeden embed edemez. Denenen/olan: sessiz sentetik fallback yasaklandı; yalnız Colab'ın ürettiği `query_embeddings.json` kabul edilir. Sıradaki adım: ürün kullanımında gerçek text-embedding GPU servisi kullanmak veya sorgu cache'ini Colab'da genişletmek.
