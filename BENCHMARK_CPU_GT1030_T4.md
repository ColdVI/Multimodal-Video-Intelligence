# CPU vs GT 1030 vs Tesla T4 ve model benchmark'ı

**Tarih:** 24 Temmuz 2026  
**Veri:** VisDrone2019-MOT Smoke-5 — 5 video, 691 kare, 7 pencere  
**Modeller:** X-CLIP 512d ve SigLIP2 frame-average 1152d

## Yönetici özeti

- **Sıcak sorgu latency'sinde T4 kazandı.** X-CLIP için CPU'ya göre `2,51×`,
  SigLIP2 için `11,11×` hızlanma ölçüldü.
- **X-CLIP mevcut iki model içinde efficiency lideri.** Sıcak sorguda CPU'da
  SigLIP2'den `13,06×`, T4'te `2,94×` hızlı; vektörü de `2,25×` daha küçük.
- **Smoke-5 accuracy sonucu iki modeli ayıramıyor.** Her iki model aynı P@10/R@10
  özetini verdi; yalnız 5 video varken `top_k=10` adayları doyuruyor.
- **GT 1030 için gerçek süre yazılamadı.** Kart sistem tarafından görülüyor;
  fakat aktif Torch paketi CPU-only olduğu için CUDA inference hiç çalışmadı.
- **T4 tam pipeline süresi GPU inference benchmark'ı değildir.** Colab raporundaki
  aşamalar model indirme/yükleme, disk hazırlığı ve subprocess cold-start içeriyor;
  pipeline ayrıca batch kullanmıyor. Bu nedenle T4'ün gözlenen toplam süresini
  yerel sıcak CPU süresine bölüp GPU speedup'ı çıkarmak metodolojik olarak yanlış.

## Donanım ve çalışma ortamı

| Ortam | Donanım | Bellek | Python / Torch | CUDA durumu |
|---|---|---:|---|---|
| Yerel CPU | Intel Core i5-13500, 20 logical processor | Sistem RAM'i | Python 3.14.6 / Torch 2.13.0+cpu | CPU |
| Yerel GPU | NVIDIA GeForce GT 1030, compute capability 6.1 | 4 GB | Aktif Torch 2.13.0+cpu | **Kullanılamıyor** |
| Colab GPU | Tesla T4 | 14,56 GB | Python 3.12.13 / Torch 2.11.0+cu128 | Kullanılıyor |

Yerel `nvidia-smi` kartı, 4 GB belleği ve sürücü `560.94` değerini doğruladı.
Ancak `torch.cuda.is_available()` değeri `False`, `torch.version.cuda` değeri
`None`, CUDA device count değeri `0` çıktı. Bu nedenle GT 1030 satırına tahminî
benchmark eklenmedi.

## Karşılaştırılabilir sıcak sorgu benchmark'ı

Bu tablo, model yüklemesinden sonraki altı **filtresiz** sorgunun ortalamasını
kullanır. Filtresiz seri, ilk CUDA/CPU warm-up sorgusundan sonra çalıştığı için
iki ortam arasında en temiz mevcut karşılaştırmadır. Arama backend'i exact
in-memory cosine'dır; ClickHouse latency'si değildir.

| Model | Yerel CPU ort. | Tesla T4 ort. | T4 hızlanması |
|---|---:|---:|---:|
| X-CLIP 512d | 24,600 ms | 9,815 ms | **2,51×** |
| SigLIP2 frameavg 1152d | 321,153 ms | 28,902 ms | **11,11×** |

İlk sorgu/warm-up etkisi özellikle büyüktür:

- Yerel X-CLIP ilk sorgu `457,17 ms`, sıcak seri `24,60 ms` ortalama.
- Yerel SigLIP2 ilk sorgu `4.874,01 ms`, sıcak seri `321,15 ms` ortalama.
- T4 raporundaki ayrı ilk interaktif X-CLIP sorgusu `507,73 ms`; sonraki sıcak
  X-CLIP serisi yaklaşık `9,82 ms` ortalama.
- T4 SigLIP2 ilk eval sorgusu `90,51 ms`; sıcak seri `28,90 ms` ortalama.

Bu nedenle serviste modeller açılışta warm-up edilmeli; ilk kullanıcı sorgusu
kernel/model hazırlama maliyetini ödememelidir.

## Model efficiency kıyası

| Özellik | X-CLIP | SigLIP2 frameavg | Sonuç |
|---|---:|---:|---|
| Video örnekleme | 32 kare, video encoder | 8 kare, ayrı image encoder + average | X-CLIP temporal olarak daha doğal |
| Embedding boyutu | 512 Float32 | 1152 Float32 | X-CLIP `2,25×` küçük |
| Ham vektör/satır | 2.048 byte | 4.608 byte | X-CLIP daha az RAM/disk/index |
| CPU sıcak query | 24,60 ms | 321,15 ms | X-CLIP `13,06×` hızlı |
| T4 sıcak query | 9,82 ms | 28,90 ms | X-CLIP `2,94×` hızlı |
| Filtre açık ort. P@10 | 1,000 | 1,000 | Berabere; veri küçük |
| Filtre açık ort. R@10 | 0,889 | 0,889 | Berabere; veri küçük |
| Filtre kapalı ort. P@10 | 0,633 | 0,633 | Berabere; aday doygun |
| Filtre kapalı ort. R@10 | 1,000 | 1,000 | Berabere; aday doygun |

**Mevcut kanıta göre çalışma modeli X-CLIP olmalı.** SigLIP2 daha büyük ve daha
yavaş olmasına rağmen Smoke-5'te ölçülebilir kalite kazancı göstermedi. Bu,
SigLIP2'nin genel olarak kötü olduğu anlamına gelmez; yalnız mevcut POC'ta ana
model olmasını destekleyen kanıt yoktur. Daha büyük görsel-denetimli eval setine
kadar deneysel baseline veya top-N reranker olarak tutulabilir.

## Cold end-to-end gözlemi

| Pipeline aşaması | Yerel CPU önceki smoke | Colab T4 raporu |
|---|---:|---:|
| Video hazırlama | 23,4 sn | 221,239 sn |
| Windowing | `<1 sn` | 0,079 sn |
| YOLO alanları | 59,9 sn | 93,082 sn |
| X-CLIP embedding | 158,2 sn | 506,965 sn |
| SigLIP2 embedding | 311,6 sn | 554,812 sn |
| Ground truth | raporda ayrı ölçülmedi | 0,458 sn |
| T4 manifest toplamı | karşılaştırılabilir toplam yok | 1.376,635 sn |

Bu tablonun gösterdiği şey “CPU, T4'ten hızlıdır” değildir. Yerel model cache'i
sıcaktı; Colab aşama timer'ı ise model indirme/yükleme ve yavaş sanal disk gibi
kurulum maliyetlerini kapsıyor. Ayrıca mevcut `ingest/03_embed.py` pencereleri
tek tek işler; GPU batch yoktur. Sonuç, mevcut Colab pipeline'ın T4'ü verimli
kullanmadığını gösterir.

## Neden GT 1030 benchmark'ı eksik?

Gerçek kontrol çıktısı:

```text
nvidia-smi: NVIDIA GeForce GT 1030, 4096 MiB, driver 560.94, compute 6.1
torch: 2.13.0+cpu
torch.cuda.is_available(): false
torch.version.cuda: null
torch.cuda.device_count(): 0
```

GT 1030 benchmark'ı için mevcut çalışan CPU ortamı bozulmadan ayrı bir CUDA
ortamı kurulmalı, seçilen Torch wheel'inin `sm_61` desteği doğrulanmalı ve
özellikle SigLIP2'nin 4 GB belleğe sığıp sığmadığı gerçek çalıştırmayla
görülmelidir. Bu yapılmadan üretilecek GT 1030 rakamı ölçüm değil tahmin olur.

## Teknik karar

1. Şimdilik ana model: **X-CLIP**.
2. GPU hedefi: **T4 veya daha güçlü CUDA GPU**.
3. GT 1030: ayrı uyumluluk deneyi yapılana kadar production benchmark dışında.
4. Pipeline optimizasyonu: model download ile inference timer'ını ayır.
5. `load`, `warm-up`, `video inference`, `text inference` sürelerini ayrı kaydet.
6. Pencereleri batch işle; mevcut window-by-window GPU kullanımını kaldır.
7. X-CLIP için 8/16/32 kare ve stride 4/6/8 ablation'ı çalıştır.
8. En az 30 sorgu ve `top_k < video_count` olan büyük eval seti olmadan model
   kalite zaferi ilan etme.

## Kanıt kaynakları

- Kullanıcı ZIP'i: `report_20260724_133745.zip`
- ZIP SHA-256:
  `632F77B0C7DDA75E45571484F7F924C5A9530E5070297B0C75D1A3AE19104EF7`
- T4 manifest:
  `artifacts/colab_report_20260724_133745/report_20260724_133745/run_manifest.json`
- T4 metrikleri:
  `artifacts/colab_report_20260724_133745/report_20260724_133745/metrics.json`
- Yerel önceki ingest süreleri: `STATUS.md` ve `RESULTS_SMOKE.md`
- Yerel sıcak query ölçümü: 24 Temmuz 2026'da aynı
  `notebooks.colab_dashboard.evaluate_models()` akışı, offline model cache ve
  aynı Smoke-5 embedding/ground truth dosyalarıyla yeniden çalıştırıldı.

