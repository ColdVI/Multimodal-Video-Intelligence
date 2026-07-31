# Kanıt kapılı uygulama planı

Bir fazın çıktısı gerçek komut ve incelenebilir artefaktla kanıtlanmadan sonraki
faz başarı sayılmaz. Ayrıntılı checklist `TASKS.md` içindedir.

## Mevcut başlangıç durumu

| Alan | Durum | Kanıt |
|---|---|---|
| Saf Python mantığı | Doğrulandı | 46/46 pytest |
| Python sözdizimi | Doğrulandı | 31 dosya py_compile |
| Model API yüzeyi | Kaynak düzeyinde incelendi | `AGENTS.md` |
| Model inference | Smoke doğrulandı | X-CLIP 512d + SigLIP2 1152d gerçek CPU |
| ClickHouse | Smoke doğrulandı | 26.7.1 schema + insert/query |
| VisDrone ingest ve GT | Kısmi | 56-sekans sözleşmesi + 5-sekans smoke |
| Retrieval kalitesi | Kısmi | 5-sekans eval teknik smoke; görsel denetim açık |

## Faz 0 — ortam ve regresyon tabanı

**Girdi:** Repo ve Python ortamı.

**İş:** Sürümleri kaydet, minimal test bağımlılıklarını kur, 46 testi ve syntax
kontrolünü çalıştır. Hata varsa regresyon testiyle düzelt.

**Kanıt:** Test çıktısı, Python/paket sürümleri, değişen dosya listesi.

**Çıkış kapısı:** Tüm saf-mantık testleri geçer. Bu, GPU hattının çalıştığını
kanıtlamaz.

## Faz 1 — ClickHouse altyapısı

**Girdi:** Docker/Compose ve yerel kaynak kapasitesi.

**İş:** Servisleri başlat, şemayı uygula, iki tabloyu ve 512/1152 boyutlu
indeksleri gerçek sunucuda doğrula. Tekrarlı schema uygulamasını test et.

**Kanıt:** Container health, ClickHouse sürümü, `SHOW TABLES`, `SHOW CREATE TABLE`.

**Çıkış kapısı:** Her iki tablo doğru boyutla var ve basit insert/select smoke
testi geçer.

## Faz 2 — resmî veri ve veri sözleşmesi

**Girdi:** Resmî Task 4 Google Drive dosyası veya önceden doğrulanmış VisDrone-MOT.

**İş:** Dizin, adlandırma, kare/anotasyon eşleşmesi, kategori kodları ve FPS'yi
örnekle. Manifest'i üret. Repo downloader'ı dışındaki kaynakları kullanma.

**Kanıt:** Sekans/kare/anotasyon sayıları, örnek parse çıktısı, manifest özeti.

**Çıkış kapısı:** En az bir smoke sekansın videosu ve pencereleri doğru süreyle
üretilir.

## Faz 3 — ilk modelle ingest ve sorgu smoke testi

**Girdi:** Faz 1 ve 2 çıktıları, model ağırlıklarına izinli erişim.

**İş:** YOLO özellikleri, X-CLIP embedding ve ClickHouse load. Önce küçük subset,
sonra kaynak uygunsa tam veri. Boyut/NaN/eksik eşleşme kontrolleri.

**Kanıt:** Her aşama satır sayısı, model/checkpoint bilgisi, tablo satır sayısı,
örnek sorgu ve dönen aralıklar.

**Çıkış kapısı:** `otobüsü göster` sorgusu gerçek model embedding'i üzerinden
sonuç döndürür. Bu yalnızca teknik çalışırlık kapısıdır.

## Faz 4 — ground truth ve kalite değerlendirmesi

**Girdi:** Gerçek anotasyon, ingest edilmiş X-CLIP tablosu.

**İş:** GT üret, yürüyüş sezgiselini görsel incele, filtre açık/kapalı eval koş.
Tekli/hareket/bileşik kategorileri ayrı raporla.

**Kanıt:** `gt.json`, `results.json`, `results_detail.json`, 5-10 görsel denetim
kaydı ve kategori sonuç tablosu.

**Çıkış kapısı:** Metrikler yeniden üretilebilir; hata örnekleri ve GT sınırlılığı
açıkça belgelenir.

## Faz 5 — ikinci model ve karar

**Girdi:** Aynı veri/GT/sorgular, SigLIP2 erişimi.

**İş:** Ayrı 1152d tabloda ingest ve aynı 2 model x 2 filtre deneyi.

**Kanıt:** Model sürümleri, eşit deney protokolü, dört hücreli sonuç tablosu.

**Çıkış kapısı:** “Hibrit faydalı mı?” ve “hangi model hangi sorgu türünde iyi?”
soruları sayı ve belirsizliklerle yanıtlanır. Sonuç olumsuz olabilir.

## Faz 6 — ölçek ve veri tabanı kararı

**Girdi:** Onaylı disk/RAM kapasitesi ve tekrar üretilebilir scale-test scripti.

**İş:** Ayrı test tablosunda kontrollü 1M, uygunsa 10M satır. Filtreli/filtresiz
warm-up ve en az 50 sorgu; p50/p95/p99.

**Kanıt:** Donanım, ClickHouse ayarları/sürümü, veri üretim yöntemi, gecikme tablosu.

**Çıkış kapısı:** ClickHouse'un gözlenen ölçeği belgelenir; 270M için karar
varsayım değil kapasite modeliyle verilir.

## Faz 7 — üretime köprü

**Girdi:** POC sonuç raporu ve kurum veri sözleşmeleri.

**İş:** Telemetri adapter'ı, metadata kataloğu, kurum altın seti, LLM parser ve
orkestrasyon için ayrı epikler oluştur.

**Kanıt:** Sahip, bağımlılık, kabul kriteri ve risk içeren backlog.

**Çıkış kapısı:** POC kodu doğrudan “production-ready” ilan edilmez; hangi
bileşenin taşındığı, değiştiği veya yeniden yazıldığı açıktır.

## Hemen durulacak durumlar

- VisDrone kayıt/kimlik bilgisi gerekiyor.
- Model lisansı veya ağ erişimi onaylı değil.
- Ölçek testi tahmini kapasiteyi aşıyor.
- Gerçek kullanıcı verisi veya hassas telemetri için yeni yetki gerekiyor.
- Veri yapısı/formatı varsayımla düzeltilemeyecek kadar belirsiz.
