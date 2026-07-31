# Son kullanıcı kılavuzu — video arama arayüzü

Bu kılavuz kod veya veritabanı bilgisi gerektirmez. Sistem sizin için önceden
kurulmuş ve videolar önceden işlenmiş (ingest edilmiş) olmalıdır — bunu
kurumunuzun teknik operatörü yapar (bkz. [OPERATOR_QUICKSTART.md](OPERATOR_QUICKSTART.md)).

## UI nasıl açılır?

Kurumunuzun size verdiği adrese tarayıcıdan gidin (örnek:
`http://<kurum-sunucusu>:7860`). Sayfa "Search" (arama) ve "Compare"
(karşılaştırma) sekmeleriyle açılır; günlük kullanım için "Search"
sekmesi yeterlidir.

## Dataset nasıl seçilir?

Sayfanın üstündeki **Dataset** açılır menüsünden kurumunuzun ingest ettiği
video koleksiyonlarından birini seçin. Genelde tek bir dataset vardır ve
zaten seçili gelir.

## Doğal dil sorgusu nasıl yazılır?

**Serbest metin sorgusu** kutusuna aradığınızı düz cümleyle yazın, sonra
**Search** düğmesine basın. Sistem cümlenizi anlamına göre videolarla
eşleştirir — anahtar kelime araması değildir, yakın anlamlı ifadeler de
sonuç getirir.

Örnek sorgular:

```text
100 metrenin altında, insanların görüldüğü gece uçuşları
Araçların bulunduğu ve gimbalın aşağı baktığı görüntüler
Kıyı bölgesinde tekne görülen video aralıkları
Ani yön değişimi görülen uçuş bölümleri
```

## Filtreler nasıl kullanılır?

Sorgunuz metinle (semantik olarak) videoyu bulurken, **filtreler** kesin
sayısal/kategorik koşullar ekler — ikisi birlikte çalışır (`VE` mantığıyla):
önce filtrelere uyan pencereler daraltılır, sonra bu daraltılmış küme içinde
sorgunuza en yakın anlamlı olanlar sıralanır. Örneğin "gece uçuşları" yazıp
**İrtifa max (m)** filtresini 100 yaparsanız, hem gece hem 100m altı koşulunu
birlikte sağlayan sonuçları görürsünüz.

Kullanılabilir filtreler (kurumunuzun verisine göre bazıları gizli olabilir):
Event category, Split, Video ID, İrtifa min/max (m), Hız min/max (m/s),
Gimbal pitch min/max, Night (gece mi), ve "Diğer canonical filtreler" altında
yön/açı gibi dairesel alanlar (bu alanlarda min > max girmek "350°-10°
arası" gibi bir sarma aralığı ifade eder — kutunun üstündeki açıklama bunu
hatırlatır). **Clear filters** düğmesi hepsini sıfırlar.

## Backend/dimension seçimi normal kullanıcı için ne anlama gelir?

**Backend** ve **Dimension**, sonucu üreten arama motorunun teknik ayarlarıdır
(örn. hangi vektör veritabanı, hangi embedding boyutu). Kurumunuzun
belirlediği varsayılan değer günlük kullanım için doğru sonucu verir —
**bunları değiştirmenize gerek yoktur.**

## Hangi seçenekleri değiştirmemesi önerilir?

- **Backend / Strategy / Dimension** — teknik varsayılan zaten doğrudur.
- **Adaptive MRL / Base dimension / Adaptive top_N** — ileri düzey performans
  ayarlarıdır, günlük arama sonucunu iyileştirmez.
- **Pattern (A/B/C)** — bu seçenek arayüzde arşiv/karşılaştırma amacıyla
  durur, aramanın nasıl çalıştığını **değiştirmez**; hangisi seçili olursa
  olsun sonuç aynı arama yolunu (yukarıdaki filtre+backend ayarları) izler.
- **Tekrar (repeats)** — yalnız gecikme ölçümü içindir, sonucu değiştirmez;
  1'de bırakın.

## Sonuç skoru nasıl okunur?

Her sonuç kartındaki skor, sorgunuzla o video penceresi arasındaki anlamsal
benzerliktir — yüksek skor, daha yakın eşleşme demektir. Skorları farklı
sorgular arasında karşılaştırmayın (yalnız aynı sorgu içindeki sıralama
anlamlıdır).

## Video ve zaman aralığı nasıl açılır?

Sonuç listesinden bir satırı **Detay için sonuç seç** menüsünden seçin;
detay panelinde video ID, başlangıç/bitiş zamanı ve (kaynak dosya
mevcutsa) oynatılabilir video kesiti görünür. Kesit yalnız birkaç saniyelik
ilgili aralığı gösterir, tüm videoyu değil.

## Provenance uyarısı ne anlama gelir?

Sonuç panelinde "provenance: real" veya "provenance: synthetic" etiketi
görürsünüz:

- **real** — embedding gerçek Qwen modeliyle gerçek videodan üretildi;
  sonuçlar üretim kalitesindedir.
- **synthetic** — bu dataset (veya bir kısmı) henüz gerçek modelle işlenmedi,
  yalnız sistem/entegrasyon testi amaçlı yer tutucu vektörler kullanılıyor;
  sonuçların anlamsal kalitesine güvenmeyin, yalnız arayüzü denemek için
  kullanın.

## Synthetic ve real embedding farkı

Synthetic mod videonun içeriğini anlamaz, yalnız veritabanı/arama
altyapısının çalıştığını doğrular. Kurumunuz gerçek videolarını ingest
ettikten sonra provenance "real"e döner; o zamana kadar synthetic
sonuçlardan operasyonel karar almayın.

## Sonuç bulunamaması ne anlama gelir?

Hiç sonuç yoksa ya filtreleriniz hiçbir pencereyle eşleşmiyordur ya da
sorgunuza yeterince yakın içerik dataset'te yok demektir. Önce filtreleri
gevşetmeyi (örn. Clear filters) deneyin.

## Underfilled/candidate shortage uyarıları

İstediğiniz sonuç sayısından (top_k) daha az sonuç dönerse arayüz bunu
"underfilled" olarak işaretler ve iki farklı nedenden birini gösterir:

- **candidate shortage** — filtrelerinize uyan pencere sayısı zaten top_k'dan
  az; filtreyi gevşetmek dışında yapılacak bir şey yoktur.
- **ann filter loss** — filtrelerinize uyan yeterli pencere var ama arama
  algoritması hepsini bulamadı; bu teknik bir performans/algoritma
  durumudur, kurumunuzun teknik operatörüne bildirin.

## Hatalı veya ilgisiz sonuç nasıl raporlanır?

Sorgu metnini, seçtiğiniz filtreleri ve beklediğiniz/aldığınız sonucu
kurumunuzun teknik operatörüne iletin; provenance etiketini de belirtin
(synthetic sonuçlar üzerinden kalite şikayeti açmayın, önce real'e
geçildiğini doğrulayın).

## Kullanıcıdan beklenmeyenler

- SQL yazmak.
- Embedding üretmek veya model indirmek.
- ClickHouse/vector database bilmek.
- Her sorguda videoları elle tekrar işlemek — ingest bir kez yapılır,
  sonraki her arama saniyeler içinde önceden hazırlanmış verilerle çalışır.
