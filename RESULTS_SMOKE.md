# Gerçek VisDrone smoke sonuçları — 24 Temmuz 2026

## Kapsam

- Veri: resmî VisDrone2019-MOT train, 56 sekans / 56 annotation / 24.201 kare.
- ZIP: 8.080.572.990 bayt; SHA-256
  `566d08fb53fff4e539f386f5a408ccf17854fd53814dc756bdede2de1dbb4014`.
- Deney subseti: 5 sekans, 691 kare, 7 kayan pencere.
- X-CLIP revision: `6746d6a5202bf24300cd9bf36f457f174483d41b`.
- SigLIP2 revision: `e8e487298228002f3d8a82e0cd5c8ea9c567f57f`.
- Ortam: Windows, Python 3.14.6, Torch 2.13.0+cpu, ClickHouse 26.7.1.1315.

Subset:

- `uav0000013_01073_v` — 58 kare
- `uav0000072_04488_v` — 85 kare
- `uav0000266_04830_v` — 116 kare
- `uav0000138_00000_v` — 213 kare
- `uav0000361_02323_v` — 219 kare

## Boru hattı kanıtı

| Aşama | Çıktı | Ölçülen süre |
|---|---:|---:|
| Resmî ZIP indirme | 8.08 GB | 56 dk 43 sn |
| ZIP çıkarma | 24.316 kayıt | 146.1 sn |
| 5 sekans frames -> MP4 | 691 kare / 5 video | 23.4 sn |
| Windowing | 7 pencere | <1 sn |
| YOLO26x özellikleri | 7/7 satır | 59.9 sn |
| X-CLIP embedding | 7×512d | 158.2 sn |
| SigLIP2 frame-average | 7×1152d | 311.6 sn |
| İki model × iki filtre eval | 24 özet / 98 detay | 42.4 sn |

Her iki modelin vektörleri sonlu ve L2-normalize bulundu. ClickHouse'ta her
modelin ayrı tablosunda 7 satır vardır; embedding boyutları karıştırılmamıştır.

## Filtre açık/kapalı smoke özeti

Aşağıdaki değerler sorgu kategorisi içinde birleştirilmiş `hits/pred/gt`
sayılarından hesaplanmıştır. İki model aynı tabloyu verdiği için tek kez
gösterilmiştir; bunun nedeni aşağıdaki sınırlılıktır.

| Filtre | Kategori | Precision | Recall | Hits / Pred / GT |
|---|---|---:|---:|---:|
| Açık | hareket | 1.000 | 1.000 | 5 / 5 / 5 |
| Açık | tekli | 1.000 | 0.889 | 8 / 8 / 9 |
| Açık | bileşik | 1.000 | 0.800 | 4 / 4 / 5 |
| Kapalı | hareket | 1.000 | 1.000 | 5 / 5 / 5 |
| Kapalı | tekli | 0.600 | 1.000 | 9 / 15 / 9 |
| Kapalı | bileşik | 0.500 | 1.000 | 5 / 10 / 5 |

Bu subsette filtre precision'ı yükseltirken YOLO'nun kaçırdığı bir kamyon
videosu nedeniyle tekli/bileşik recall'ı düşürdü. Bu beklenen hibrit sistem
trade-off'udur; tam veri ve görsel hata analizi olmadan eşik değiştirilmemelidir.

## Neden model kıyası henüz yapılamaz

`eval.top_k=10`, fakat smoke subsetinde yalnızca 5 video var. Filtre kapalıyken
her iki model de bütün videoları ilk 10'a aldığı için sıralamaları farklı olsa
bile P@10/R@10 aynı çıkıyor. Örneğin filtresiz `otobüsü göster` sıralamalarında
X-CLIP ilk sıraya `uav0000072_04488_v`, SigLIP2 ise
`uav0000266_04830_v` koydu; gerçek otobüslü iki video farklı sıralardadır.
Model kararı için daha büyük, görsel olarak denetlenmiş eval seti zorunludur.

## Gerçek çalıştırmada bulunan ve düzeltilen ek hatalar

- `config.yaml` annotation yolu resmî veri yapısıyla uyuşmuyordu.
- Odd-height 1904×1071 kareler libx264/yuv420p'yi kırıyordu; koordinatları
  değiştirmeyen 1-piksel padding eklendi.
- ClickHouse HTTP client iki `CREATE TABLE` ifadesini tek komutta reddediyordu.
- Windows Python stdout `cp1252` Türkçe GT özetinde çöküyordu.
- SigLIP2 checkpoint'i resmî config'e aykırı biçimde `Siglip2Model` sınıfına
  zorlanıyordu; resmî model kartındaki `AutoModel` kullanımına düzeltildi.
- FiftyOne temporal support video sonunu bir kare aşıyordu; manifest kare
  sayısına clamp edildi ve doğrudan script import yolu düzeltildi.

## Açık karar kapıları

- 5-10 sekanslık gerçek `gt_walking` görsel/ego-motion denetimi yapılmalı.
- Beş videodan daha büyük eval setinde iki modelin sıralama kalitesi ölçülmeli.
- Tam 56-sekans CPU ingest'i saatler süreceği için önce temsilî subset kapsamı
  kararlaştırılmalı veya CUDA destekli ortam kullanılmalı.
- 1M/10M ClickHouse ölçek testi henüz yapılmadı.
