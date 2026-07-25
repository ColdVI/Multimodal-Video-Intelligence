# Uygulama durumu — 24 Temmuz 2026

## Tamamlanan ve gerçek çalıştırmayla doğrulananlar

- Windows/Python 3.14 için `.venv` kuruldu; `requirements.txt` eksiksiz yüklendi.
- 46 pytest testi geçti; Python dosyaları `py_compile` kontrolünden geçti.
- Üçüncü taraf cache/config yazımları repo içindeki `.runtime/` dizinine alındı.
- Sistem ffmpeg'i yokken `imageio-ffmpeg` fallback'i gerçek binary ile çalıştı.
- Docker Desktop başlatıldı; ClickHouse `26.7.1.1315` ve MinIO çalışıyor.
- `schema.sql` gerçek ClickHouse'a uygulandı; 512d ve 1152d tablolar doğrulandı.
- İki tabloda insert + filtre + `cosineDistance` smoke testi geçti ve satırlar temizlendi.
- X-CLIP checkpoint'i indirildi; CPU'da gerçek 512d video/metin embedding üretildi.
- Transformers 5.x pooled-output değişikliği düzeltildi ve regresyon testi eklendi.
- Gerçek X-CLIP vektörüyle parser → filtreli/filtresiz ClickHouse sorgusu →
  interval merge hattı uçtan uca geçti.
- `yolo26x.pt` indirildi; CPU'da tek görüntü inference ve sentetik MP4 üzerinde
  `window_features()` filtresi/optical-flow hattı geçti.
- Windows için `scripts/poc.ps1` runner eklendi; `test` ve `schema` görevleri geçti.
- Resmî Task 4 VisDrone-MOT trainset Google Drive'dan indirildi: 8.080.572.990
  bayt, SHA-256 `566d08fb53fff4e539f386f5a408ccf17854fd53814dc756bdede2de1dbb4014`.
- Veri sözleşmesi doğrulandı: 56 sekans, 56 annotation, 24.201 JPEG ve sıfır
  sekans/annotation isim uyuşmazlığı. `scripts/poc.ps1 download-data` aynı
  sözleşmeyi yeniden üretilebilir biçimde doğruluyor.
- 5 gerçek sekans/691 kare üzerinde ingest smoke'u geçti: 5 MP4 -> 7 pencere ->
  7 YOLO özellik satırı -> 7×512d X-CLIP ve 7×1152d SigLIP2 embedding ->
  model başına ayrı ClickHouse tablosunda 7 satır.
- Gerçek annotation'dan altı sorgu için GT ve iki model × iki filtre eval
  üretildi (`results.json`: 24, `results_detail.json`: 98 satır).
- Gerçek çalıştırmada ayrıca annotation yolu, ClickHouse HTTP multi-statement
  load ve Windows Python UTF-8 hataları bulunup düzeltildi.
- Exact filtre, exact brute-force vektör, HNSW ve iki hibrit strateji için yedi
  salt-okunur SQL tek katalogda toplandı; aynı katalog test ve rapor tarafından
  kullanılıyor. Gerçek ClickHouse raporu HTML ve JSON olarak üretildi.
- Mevcut 14 satırlık smoke verisinde exact brute-force ve HNSW aynı sıralamayı
  verdi; yaklaşık mesafe değerleri birebir aynı değildi (maksimum fark
  `0.0001023`). Exact ve iki hibrit sorgunun her biri dört filtre eşleşmesi döndürdü.
  `EXPLAIN indexes = 1`, HNSW indeksini similarity ve postfilter/rescore planında
  gösterdi; brute-force exact ve prefilter planında göstermedi.

## Ölçülen ortam

- GPU: NVIDIA GeForce GT 1030, 4 GB.
- Kurulu Torch: `2.13.0+cpu`; CUDA kullanılamıyor.
- X-CLIP CPU smoke: model yükleme 0.87 sn, 32 kare embedding 7.92 sn,
  metin embedding 0.32 sn.
- YOLO26x CPU smoke: ilk indirme+yükleme 207.8 sn; tek 640×640 inference 1.54 sn.
- Gerçek iki-pencere YOLO: 17.7 sn; gerçek iki-pencere X-CLIP: 51.2 sn.
- 5-sekans ölçümü: frames 23.4 sn, YOLO 59.9 sn, X-CLIP 158.2 sn,
  SigLIP2 311.6 sn, iki-model eval 42.4 sn.
- `.venv` boyutu: yaklaşık 1.83 GB (model cache'leri hariç).
- Smoke-5 sıcak query benchmark'ında T4, X-CLIP'i yerel CPU'dan `2,51×`,
  SigLIP2'yi `11,11×` hızlı çalıştırdı. GT 1030 sistemde görünmesine rağmen
  aktif Torch CPU-only olduğu için CUDA benchmark'ı çalıştırılamadı. Ayrıntı:
  `BENCHMARK_CPU_GT1030_T4.md`.

## Faz 1 benchmark altyapısı (docs/codex/05_..., 25 Temmuz 2026)

- `bench/` paketi gerçek çalıştırıldı: 19 sekans (56'dan seçilmiş temsili bench
  subset, `config.yaml: bench.subset`) → 73 pencere → 2 model × 2 filtre = 4
  run, `artifacts/benchmark_report.{html,json}`.
- GT 28 sorguya çıkarıldı (TR+EN çiftli, tekli/hareket/sayısal/bileşik/
  negatif-kontrol). Filtre AÇIK tutarlı biçimde precision ve MRR'ı artırıyor
  (ör. X-CLIP bileşik: MRR 0.92 filtreli vs 0.51 filtresiz); recall filtresiz
  biraz daha yüksek kalabiliyor (aday havuzu daralmıyor) — beklenen ödünleşim.
  Negatif-kontrol sorguları (n_gt=0) recall/precision=0 veriyor; bu P/R@K
  metriğinin bir sınırıdır (boş GT'ye karşı hiçbir tahmin "doğru" sayılamaz),
  başarısızlık değildir.
- Determinizm kontrolü GEÇTİ: aynı RunSpec iki kez koşuldu, 28 sorgunun tümünde
  metrikler birebir aynı çıktı.
- SigLIP2 sorgu gecikmesi X-CLIP'in ~3 katı (mean 0.92sn vs 0.31sn, filtreli),
  512d/1152d boyut farkına ve CPU-only ortama tutarlı.
- GT 1030 CUDA denemesi bloke oldu (Windows `MAX_PATH`/`LongPathsEnabled=0`);
  detay `docs/codex/02_FIKIRLER_VE_KARARLAR.md`. CPU-only ortam korunuyor,
  54→72 test hâlâ geçiyor.

## Açık kalanlar

- Tüm 56 sekans üzerinde toplu MP4/YOLO/model ingest henüz koşulmadı; mevcut
  kanıt 5 sekans/7 pencere smoke subsetine aittir.
- `gt_walking` gerçek annotation'da çalıştı fakat 5-10 sekanslık görsel
  ego-motion denetimi yapılmadı.
- Filtre A/B ve iki model eval teknik olarak üretildi; 5 video `top_k=10`'dan
  küçük olduğu için model sıralama kalitesi bu metrikle ayırt edilemiyor.
- FiftyOne dataset'i headless olarak 98 sample ile kuruldu; insanın GUI'de
  5-10 örnek `gt_walking` denetimi henüz yapılmadı.

## Sonraki kontrollü adım

Veri ve 5-sekans smoke hazır. Sonraki zorunlu kapı FiftyOne'da görsel GT
denetimidir. Mevcut ilk smoke'u
yeniden üretmek için:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/poc.ps1 download-data
powershell -ExecutionPolicy Bypass -File scripts/poc.ps1 frames -Sequence uav0000138_00000_v
powershell -ExecutionPolicy Bypass -File scripts/poc.ps1 windows
```

Tam 56-sekans CPU ingest'i saatler sürebileceği için ölçüm yapmadan başlatılmamalıdır.
