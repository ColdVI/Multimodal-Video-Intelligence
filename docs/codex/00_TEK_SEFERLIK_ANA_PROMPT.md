# Tek seferlik ana Codex promptu

Aşağıdaki metni repo kökünde açılmış yeni bir Codex konuşmasına yapıştırın.

```text
Bu repodaki hibrit video arama POC'unu, mevcut kanıta sadık kalarak gerçek
veriyle uçtan uca doğrulanmış hale getir.

Başlamadan önce CODEX_START_HERE.md, AGENTS.md, CONTEXT.md ve TASKS.md
dosyalarının tamamını oku; ardından kodu ve mevcut çalışma ağacını incele.
Eski planlardaki örnek kodu çalışan reponun üzerine körlemesine kopyalama.
Çelişkide çalışan kod/test sözleşmeleri, sonra AGENTS.md ve CONTEXT.md geçerli.

Çalışma kuralları:

- Önce mevcut 46 testi ve tüm Python dosyalarının syntax
  kontrolünü çalıştır. Gerçek bir hata bulursan kök nedenini kanıtla, en küçük
  güvenli düzeltmeyi yap, regresyon testi ekle ve tekrar çalıştır.
- Her faz sonunda testleri yeniden çalıştır. TASKS.md kutularını yalnızca
  komut/çıktı veya incelenebilir dosya kanıtı varsa işaretle.
- Model başına ayrı ClickHouse tablosu tasarımını koru: X-CLIP 512 boyut,
  SigLIP2 1152 boyut. Tek HNSW kolonuna farklı boyutları karıştırma.
- Yeni sayısal sabitleri koda gömme; config.yaml ve common.load_config()
  üzerinden yönet.
- X-CLIP adlarını karıştırma: microsoft/xclip HF modeli ile Ma vd. AOSM
  retrieval modeli aynı değildir. Sonuçlarda ayrı model adları kullan.
- Doğrulanmayan adımları tamamlanmış gibi yazma. Mock başarıyı gerçek
  inference/ClickHouse başarısı yerine kullanma.
- VisDrone verisi eksikse yalnızca scripts/download_visdrone.py içindeki
  resmî Task 4 bağlantısını ve bütünlük kontrollerini kullan. Kota/giriş
  engelinde kimlik bilgisi isteme; kullanıcıdan resmî ZIP'i yerleştirmesini iste.
- Mevcut kullanıcı değişikliklerini koru. Yıkıcı git/dosya işlemi yapma.

Uygulama sırası:

1. Ortam ve regresyon doğrulaması.
2. Docker/ClickHouse altyapısını aç; schema.sql'i uygula ve iki model tablosunu
   SHOW TABLES/DESCRIBE ile doğrula.
3. Veriyi resmî downloader ile doğrula; yapı, kare sayısı, anotasyon eşleşmesi
   ve manifest FPS varsayımını örnekleyerek doğrula.
4. Önce xclip_hf_zeroshot ile frames -> windows -> detect -> embed -> load
   hattını küçük bir smoke subset'te, sonra uygun ölçekte çalıştır.
5. Ground truth üret; özellikle gt_walking için kamera hareketi yanlış
   pozitiflerini 5-10 sekans üzerinde görsel olarak incele. Gerekirse bulguyu
   ve kalibrasyonu kod/test/config ile kaydet.
6. Filtre açık/kapalı A/B değerlendirmesini çalıştır; results.json ve
   results_detail.json üret. Tekli/hareket/bileşik kırılımını raporla.
7. SigLIP2 modelini ayrı tabloda çalıştır ve aynı karşılaştırmayı tekrarla.
8. Sonuç anlamlıysa opsiyonel 1M/10M ClickHouse ölçek testini ölç; p50/p95,
   donanım ve sorgu ayarlarını kaydet.
9. docs/codex/04_KABUL_KRITERLERI_VE_RAPOR.md şablonuyla nihai rapor üret.

Her çalışma turunun sonunda bana şunları ver:

- Değişen dosyalar ve nedenleri.
- Çalıştırılan tam komutlar ve özet çıktıları.
- Geçen/kalan kabul kriterleri.
- Doğrulanmamış riskler ve tek net sonraki adım.

İlk turda mevcut kanıtı yeniden doğrula; veri yoksa resmî downloader'ı kullan,
yalnızca bağlantı kimlik/kota engeline düşerse benden veri iste.
```
