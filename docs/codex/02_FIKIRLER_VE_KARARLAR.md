# Fikirler ve karar defteri

## Ürünün tek cümlelik hedefi

Doğal dil sorgusunu önce güvenilir yapısal filtrelerle daraltıp sonra video
embedding benzerliğiyle sıralayarak, sonucu video kimliği + zaman aralığı +
skor biçiminde veren ölçülebilir bir hibrit retrieval sistemi kurmak.

## Korunacak mimari kararlar

| Karar | Gerekçe | Bozulursa risk |
|---|---|---|
| Filtre + semantik sıralama ayrı katmanlar | Hatanın parser, detektör veya embedding kaynaklı olduğu ölçülebilir | Bütün hata tek skorda kaybolur |
| Semantik modele tam sorgu gönderilir | Filtrelenen kavramların sahne bağlamı korunur | Ana bağlam embedding'den silinir |
| Model başına ClickHouse tablosu | 512d X-CLIP ile 1152d SigLIP2 aynı HNSW boyutuna sığmaz | Sessiz bozuk indeks veya sorgu hatası |
| Kural tabanlı parser önce | Dar kavram uzayında deterministik baseline verir | LLM hatası retrieval hatasıyla karışır |
| Detektör kolonları telemetri vekili | Açık drone verisinde gerçek telemetri yoktur | POC/üretim varsayımı görünmez olur |
| Otomatik VisDrone ground truth | Tekrar üretilebilir değerlendirme sağlar | Elle seçilmiş örneklerle yanıltıcı başarı |
| Faz kapıları ve kanıt kaydı | “Kod yazıldı” ile “gerçek ortamda çalıştı” ayrılır | Doğrulanmamış başarı iddiası |

## Kritik teknik notlar

- Hugging Face `microsoft/xclip` modeli ile Ma vd. retrieval-özel AOSM
  X-CLIP farklıdır. İkisini sonuçlarda tek “X-CLIP” satırında birleştirmeyin.
- SigLIP2 adapter'ı bilinen generic `AutoModel` tip sorunundan kaçınmak için
  açıkça `Siglip2Model` kullanır.
- `frames_to_intervals` kareyi nokta değil `[i/fps, (i+1)/fps)` aralığı olarak
  ele alır. `+1` düzeltmesi korunmalı; 25 kare/25 fps tam 1 saniyedir.
- `gt_walking`, görüntü düzlemindeki track hareketini kullanır ve ego-motion
  telafisi yapmaz. Gerçek veri görsel denetimi olmadan güvenilir etiket değildir.
- Filtre yanlış negatifleri geri döndürülemez aday kaybı yaratabilir. Saf
  vektör modu bu nedenle karşılaştırma baseline'ı olarak kalmalıdır.

## Önceliklendirilmiş geliştirme fikirleri

### Şimdi — POC sonucunu güvenilir yapmak

1. Küçük, sabit bir smoke sekans listesi ve beklenen ara dosya şemaları ekleyin.
2. Her embedding çıktısında model ID, checkpoint revision, boyut, normalize
   durumu ve çalışma cihazını metadata olarak saklayın.
3. Ingest aşamalarına satır sayısı, eksik pencere, NaN/Inf ve boyut doğrulaması
   ekleyin; boş çıktıyı başarı saymayın.
4. FPS'yi sekans bazında tek manifest kaynağı yapın; FiftyOne görünümündeki
   sabit 25 fps kullanımını manifest ile değiştirin.
5. Parser ve SQL üretimi için bilinmeyen kavram, Türkçe karakter varyantı,
   sayı ve çelişkili filtre testleri ekleyin.

### Sonra — retrieval kalitesini artırmak

1. Sert filtre ve yumuşak filtreyi ayrı ölçün. Güveni düşük detektör sonucu
   adayı tamamen elemek yerine skora ceza olarak eklenebilir.
2. Detektör recall'ını VisDrone anotasyonuna karşı ayrıca ölçün; hibrit kazancı
   ile filtre katmanı kalitesini aynı metrikte gizlemeyin.
3. Model skorlarını doğrudan karşılaştırmadan önce sorgu kategorisi bazında
   kalibre edin; cosine dağılımları modelden modele değişebilir.
4. Hard-negative set ekleyin: otobüs/van/truck, yürüyen/duran yaya ve benzer
   görsel fakat farklı hareket örnekleri.
5. Hareket ground truth'u için global kamera hareketini medyan track/feature
   hareketiyle telafi eden ayrı ve ölçülebilir bir baseline deneyin.

### Daha sonra — üretimleşme

1. Kural parser ile aynı `ParsedQuery` sözleşmesini kullanan JSON-schema
   kontrollü bir LLM parser ekleyin; ikisini regresyon setinde A/B çalıştırın.
2. Telemetri formatı kesinleşince detektör vekilini MAVLink veya MISB KLV
   adapter'ıyla değiştirin; retrieval katmanına dokunmayın.
3. Platform/IHA tipi gibi güvenilir metadata'yı kurumsal katalogdan join edin;
   embedding'e çözdürmeyin.
4. Kurum verisinde çift etiketleyicili 200-500 sorgu-aralık altın seti kurun.
5. 1M/10M ölçümünden sonra ClickHouse/Qdrant kararını gecikme, recall ve
   operasyon maliyeti birlikte belirlesin.

## Üretim öncesi cevaplanması gereken sorular

- Gerçek telemetri MAVLink logu mu, STANAG 4609/MISB KLV mi, başka bir format mı?
- Video, telemetri ve platform metadata'sını bağlayan değişmez kimlik nedir?
- Kurum sorgularının dil dağılımı, güvenlik sınırı ve hassas metadata kuralları nedir?
- Kabul edilen p95 sorgu gecikmesi, recall hedefi ve saklama maliyeti nedir?
- GPU, ağ erişimi ve model lisansları hangi deployment ortamında onaylıdır?
