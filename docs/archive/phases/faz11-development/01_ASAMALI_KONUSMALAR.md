# Aşamalı Codex konuşmaları

Bu mesajlar uzun işi küçük kanıt kapılarına ayırır. Her yeni mesajda Codex,
önce önceki turun gerçek durumunu ve `TASKS.md` kutularını kontrol etmelidir.
Bir faz başarısızsa sonraki mesajı vermeyin; aynı konuşmada hatayı giderin.

## 1. Repo denetimi ve regresyon tabanı

```text
Repo kökünde CODEX_START_HERE.md, AGENTS.md, CONTEXT.md ve TASKS.md'yi tamamen
oku. Çalışma ağacını ve Python ortamını denetle. Mevcut testleri ve tüm Python
dosyalarının syntax kontrolünü çalıştır. İddiaları yalnızca gerçek çıktılarla
doğrula. Hata varsa kök neden + en küçük düzeltme + regresyon testi yap.
Bu turda model indirme, veri indirme veya Docker başlatma. Sonunda komutları,
46 testin sonucunu, değişen dosyaları ve sonraki tek adımı raporla.
```

## 2. Altyapı ve şema doğrulaması

```text
Faz 0'ın altyapı kısmını yap. Docker/Compose önkoşullarını kontrol et, yalnızca
yerel POC servislerini başlat, schema.sql'i gerçek ClickHouse'a uygula.
clips_xclip_hf_zeroshot ve clips_siglip2_frameavg tablolarının varlığını,
embedding boyutlarını ve HNSW indeks tanımlarını SHOW/DESCRIBE çıktılarıyla
kanıtla. Servis veya ClickHouse sürümü uyumsuzsa sebebi teşhis edip güvenli
düzeltmeyi yap; doğrulamadan TASKS.md kutusu işaretleme. Tur sonunda testleri
yeniden çalıştır ve tam komut/çıktı özetini ver.
```

## 3. Resmî veri indirme ve veri denetimi

Veri mevcut değilse önce repo içindeki doğrulamalı downloader kullanılır.

```text
Önce scripts/poc.ps1 download-data görevini çalıştır; veri zaten mevcutsa bu
görev yalnızca 56/56/24.201 sözleşmesini doğrulamalı, yeniden indirmemeli.
data/raw/VisDrone2019-MOT-train/{sequences,annotations} yapısını salt okunur
kontrollerle doğrula: sekans sayısı, örnek kare adlandırması, anotasyon alan
sayısı, sekans-anotasyon eşleşmesi ve olası FPS bilgisi. FPS=25 varsayımını
kanıtsız genelleme; veri kaynağı veya örnek metadata farklıysa manifest
tasarımını sekans bazında güvenilir hale getir. Önce küçük bir smoke sekansı
seç ve frames/windows çıktısını doğrula. Bulguları ve çalıştırılan komutları
raporla; veri yapısı yanlışsa ingest'e geçmeden dur.
```

## 4. İlk modelle uçtan uca ingest

```text
Önce xclip_hf_zeroshot ile uçtan uca ingest'i doğrula. Küçük smoke subset'te
frames -> windows -> detect -> embed -> ClickHouse load sırasını çalıştır.
Her aşamada satır sayısı, boş/NaN embedding, embedding boyutu, eksik pencere ve
tablo satır sayısını kontrol et. Model veya YOLO fallback'i kullanılırsa tam
model kimliği/sürümü kaydet. Smoke başarılı olmadan tam veriye geçme.
Gerçek bir hata bulursan test edilebilir kısmına regresyon testi ekle.
Sonunda “otobüsü göster” sorgusunu çalıştır; dönen aralıkların kalite kanıtı
değil, yalnızca hattın çalışırlık kanıtı olduğunu açıkça belirt.
```

## 5. Ground truth ve görsel kalite kapısı

```text
Gerçek VisDrone anotasyonlarından ground truth üret. Her sorgu için eşleşen
video/aralık sayılarını çıkar; boş sorguları sessizce geçme. gt_walking
sezgiselini en az 5-10 temsilî sekans üzerinde FiftyOne veya eşdeğer görsel
incelemeyle kontrol et. Ego-motion nedeniyle duran yayaların yürüyen sayılıp
sayılmadığını örnek zaman aralıklarıyla raporla. Kalibrasyon gerekirse
sabitleri config.yaml'a taşı, kodu ve sentetik testleri güncelle; görsel kanıt
olmadan problemi çözüldü sayma. Testleri tekrar çalıştır ve karar kapısını
geçti/geçmedi diye açık yaz.
```

## 6. Filtre açık/kapalı A/B değerlendirmesi

```text
xclip_hf_zeroshot için aynı sorgu ve ground truth üzerinde filtre AÇIK ve
KAPALI değerlendirmesini çalıştır. results.json ve results_detail.json
dosyalarını şema/satır sayısıyla doğrula. P@10, R@10 ve n_gt değerlerini
tekli/hareket/bileşik kategorilerinde tabloya dök. Filtrenin kazancını yalnızca
aynı embedding modelinin açık-kapalı farkıyla yorumla. Veri azlığı, detektör
yanlış negatifi ve aralık birleştirme etkisini belirt. Sonuç anlamlı değilse
bunu başarısız hipotez olarak dürüstçe raporla; metriği iyileştirmek için
eşiği sonuçlara bakarak keyfî ayarlama.
```

## 7. İkinci model ve model karşılaştırması

```text
SigLIP2 kare-ortalama baseline'ını clips_siglip2_frameavg tablosunda çalıştır.
Siglip2Model yüklemesini ve gerçek 1152 boyutlu çıktıyı doğrula; X-CLIP
tablosuna veri yazma. Aynı veri, sorgu, filtre modu ve metriklerle 2 model x
2 filtre karşılaştırmasını üret. Model adlarını ve checkpoint sürümlerini
rapora sabitle. Kazananı yalnızca toplam ortalamayla seçme; hareket ve bileşik
sorgu kırılımını özellikle göster. Testleri yeniden çalıştır.
```

## 8. Opsiyonel ölçek testi

```text
Önce ölçek testinin kapsamını ve disk/RAM tahminini yaz; güvenli kapasite
onayı olmadan 1M/10M insert başlatma. Onaylı kapasitede, embedding dağılımını
kontrollü gürültüyle çoğaltan tekrar üretilebilir bir scale-test scripti ekle.
Filtre açık/kapalı en az 50 sorgu için warm-up, p50, p95, p99, ClickHouse
sürümü ve donanımı kaydet. Test verisini ayrı tabloda tut; temizleme komutunu
raporla fakat kullanıcı açıkça istemeden silme. Sonucu 270M ölçeğe kesin hüküm
gibi ekstrapole etme.
```

## 9. Nihai rapor ve üretime köprü

```text
Tüm gerçek çıktıları docs/codex/04_KABUL_KRITERLERI_VE_RAPOR.md şablonuna
yerleştir. Tamamlandı, kısmen doğrulandı ve doğrulanmadı durumlarını ayır.
Komutları, sürümleri, veri kapsamını, sonuç tablolarını, hata örneklerini ve
karar kapılarını ekle. Üretime geçiş için telemetri formatı (MAVLink mi,
STANAG 4609/MISB KLV mi), platform metadata kataloğu ve kurum altın setini
açık bağımlılık olarak yaz. Son olarak TASKS.md'yi yalnızca kanıtlı kutularla
güncelle ve kalan işleri öncelikli kısa liste olarak ver.
```
