# Kabul kriterleri ve sonuç raporu şablonu

## POC bitti tanımı

POC ancak aşağıdaki zorunlu maddeler gerçek ortam kanıtıyla tamamlandığında
“uçtan uca doğrulandı” sayılır:

- [x] 46 test ve syntax kontrolü geçiyor.
- [x] ClickHouse şeması gerçek sunucuda iki model tablosuyla uygulanıyor.
- [x] VisDrone veri yapısı ve smoke sekans FPS/manifest'i doğrulanıyor.
- [x] X-CLIP gerçek VisDrone inference, load ve örnek hibrit sorgu çalışıyor.
- [ ] Otomatik GT üretiliyor; `gt_walking` görsel denetim sonucu belgeli.
- [x] X-CLIP için filtre açık/kapalı smoke sonuçları üretiliyor; tek videolu
      sonuçların kalite kanıtı olmadığı açıkça belgeleniyor.
- [x] SigLIP2 gerçek inference ve ayrı 1152d tabloda aynı smoke eval çalışıyor.
- [x] Smoke sonuçları tekli/hareket/bileşik kırılımında raporlanıyor.
- [ ] Başarısız örnekler, doğrulanmayan riskler ve yeniden üretim komutları var.

Ölçek testi POC'un çekirdeği için opsiyoneldir; üretim veri tabanı kararı için
zorunludur.

## Sonuç raporu

### 1. Deney kimliği

- Tarih/saat:
- Commit veya paket hash'i:
- İşletim sistemi / Python:
- GPU / CPU / RAM:
- Docker ve ClickHouse sürümü:
- Model kimlikleri ve checkpoint revision'ları:
- Veri seti split'i, sekans ve pencere sayısı:

### 2. Yeniden üretim komutları

```text
# Çalıştırılan komutları sırasıyla ve aynen buraya yazın.
```

### 3. Boru hattı kanıtı

| Aşama | Girdi | Çıktı/satır sayısı | Süre | Durum | Kanıt dosyası |
|---|---:|---:|---:|---|---|
| Frames -> video | | | | | |
| Windowing | | | | | |
| Detection | | | | | |
| X-CLIP embedding | | | | | |
| X-CLIP load | | | | | |
| Ground truth | | | | | |
| SigLIP2 embedding | | | | | |
| SigLIP2 load | | | | | |

### 4. Retrieval sonuçları

| Model | Filtre | Tekli P@10 / R@10 | Hareket P@10 / R@10 | Bileşik P@10 / R@10 |
|---|---|---|---|---|
| xclip_hf_zeroshot | Kapalı | | | |
| xclip_hf_zeroshot | Açık | | | |
| siglip2_frameavg | Kapalı | | | |
| siglip2_frameavg | Açık | | | |

Her hücrede `n_gt` ve sorgu sayısını da verin. Filtre etkisini aynı modelin
açık/kapalı farkıyla, model etkisini aynı filtre modundaki farkla yorumlayın.

### 5. Ground truth görsel denetimi

| Video | Zaman | Sorgu | GT doğru mu? | Hata türü | Karar |
|---|---|---|---|---|---|
| | | | | ego-motion / track / kategori / sınır | |

En az 5-10 örnek ekleyin; yalnızca iyi örnek seçmeyin.

### 6. Hata örnekleri

- Yanlış pozitifler:
- Yanlış negatifler:
- Doğru video, yanlış zaman sınırı:
- Detektör filtresi nedeniyle kalıcı kayıp:
- Parser hatası:
- Model/altyapı hatası:

### 7. Karar kapıları

| Soru | Sonuç | Kanıt | Karar |
|---|---|---|---|
| Hibrit filtre bileşik sorguyu iyileştiriyor mu? | | | geç / kal / tekrar deney |
| Klip modeli hareket sorgusunda frame-average'i geçiyor mu? | | | |
| GT yeterince güvenilir mi? | | | |
| ClickHouse gözlenen ölçekte yeterli mi? | | | |

### 8. Doğrulanmayanlar ve riskler

Tamamlanmayan her maddeyi neden, etki ve kapanış adımıyla yazın. “Kod mevcut”
ifadesi “çalıştı” kanıtı değildir.

### 9. Nihai öneri

- Devam / pivot / durdur:
- Önerinin dayandığı üç somut bulgu:
- Üretime geçmeden önce zorunlu üç iş:
- Sahip ve sonraki karar tarihi:
