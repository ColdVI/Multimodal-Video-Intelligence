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

## Faz 2 kanıtlı bulgusu: seçici filtrede varsayılan strateji güvensiz

100K satırlık sentetik ölçekte (`bench_scale_512`, üretim tablosuna
dokunmadan) `vector_search_filter_strategy='auto'` (ClickHouse varsayılanı,
postfiltering) seçici filtrede (`bus_count>=1 AND person_count>=3`) **0 satır**
döndürdü; `prefilter` ve `bruteforce` aynı filtreyle doğru 10 satır döndürdü.
`vector_search_index_fetch_multiplier`'ı varsayılan 1'den 50'ye çıkarmak
düzeltiyor ama o noktada gecikme prefilter'a yakınsıyor. **Karar:** filtre
seçiciliği bilinmeyen/yüksek olabilecek sorgularda `search/query.py::search()`
`strategy='prefilter'` ile çağrılmalı; `strategy='auto'` (varsayılan, imza
korunuyor) yalnızca gevşek/filtresiz sorgular için güvenli kabul edilir. Detay
ve tam sayılar: `STATUS.md` Faz 2 bölümü, `artifacts/strategy_matrix_report.json`.

## Faz 3 kanıtlı bulgusu: küçük fine-tune'lu model her sınıfta kazanmıyor

`05_CODEX_BENCHMARK_VE_OPTIMIZASYON_PLANI.md`'nin referans aldığı
`dronefreak/visdrone-detection-model-zoo` reposu gerçekte yok (indirmeden
önce doğrulandı, 401 döndü) — gerçek alternatif `mshamrai/yolov8{n,s,m}
-visdrone` kullanıldı. 73 pencerelik gerçek bake-off'ta VisDrone-tuned
`yolov8n`/`yolov8s`, COCO-pretrained `yolo26x`'e göre person/bus eşik
doğruluğunda eşdeğer/hafif iyi ve ~2× hızlıydı — ama **truck recall'da
COCO x-large modeli (0.88) her iki küçük VisDrone-tuned varyandan
(0.50/0.71) belirgin önde**. Karar: `yolov8n_visdrone` varsayılan (hız +
downstream Recall/Precision@10 birlikte gerekçelendiriyor), ama "küçük
fine-tune'lu model büyük genel modeli her sınıfta geçer" varsayımı genel
bir kural olarak KABUL EDİLMEDİ — sınıf bazlı doğrulama olmadan yeni bir
dedektör değişikliği yapılmamalı. Detay: `TASKS.md` Faz 3,
`artifacts/detector_bakeoff.json`.

## Faz 4 aday doğrulaması: planın "kolay HF indirme" varsayımı iki adayda yanlış çıktı

Planın Faz 4 aday tablosu VideoCLIP-XL ve LanguageBind_Video'yu "HF checkpoint'i
lokal indirilebilir" diye tanımlıyor; ikisi de gerçek ve indirilebilir ama
"kolay entegrasyon" varsayımı doğrulanmadı:

- **VideoCLIP-XL** (`alibaba-pai/VideoCLIP-XL`, gerçek): lisansı
  **CC-BY-NC-SA-4.0 (ticari olmayan)** — bu projenin "kurumsal İHA şirketi
  üretimi" hedefiyle doğrudan çelişiyor, teknik performanstan bağımsız
  olarak eleniyor. Ayrıca standart `transformers` checkpoint'i değil; özel
  `modeling.py` + `utils/` kodu vendor edilmesi gerekiyor (planın
  `xclip_ma_aosm` için öngördüğü entegrasyon riskiyle aynı sınıfta).
- **LanguageBind_Video** (`LanguageBind/LanguageBind_Video`, gerçek): lisans
  MIT (temiz), ama `model_type: LanguageBindVideo` yüklü `transformers`
  (5.14.1) tarafından tanınmıyor — `AutoModel.from_pretrained()` gerçek
  çalıştırmada `ValueError` verdi. Resmi olmayan bir PyPI paketi
  (`languagebind`, PKU-YuanGroup deposunun pip-kurulabilir hâli) veya
  orijinal GitHub kodu gerekiyor. Üçüncü parti/resmî-olmayan paket güven
  riski + süre bütçesi nedeniyle bu oturumda entegre edilmedi.

**Karar:** Faz 4'ün gerçek, ölçülen yeni adayı olarak `Qwen/Qwen3-VL-Embedding-2B`
kullanıldı (Apache-2.0, gerçek doğrulandı, `transformers` 5.14.1 mimariyi
zaten native destekliyor, `sentence-transformers` üzerinden standart
`model.encode()` ile çalışıyor — vendor kod gerekmiyor). Detay ve gerçek
ölçüm sonuçları: `STATUS.md` Faz 4, `artifacts/`.

**İki ek gerçek bulgu:**
1. Qwen'in CPU'da `embed_video` maliyeti (~14.5 dk/pencere, 73 pencerede
   ölçülen) bu donanımda ingest'i pratik dışı bırakıyor — ama bu CPU'ya
   özgü, modelin kendisine dair bir hüküm değil. GPU ölçümü bu oturumda
   yapılamadı (`scripts/colab_gpu_bench.py` hazırlandı, kullanıcı Colab'de
   çalıştıracak, henüz doğrulanmadı).
2. MRL boyut taraması (2048→256d, tek gerçek koşumdan türetildi):
   kalite kaybı <0.02 recall, depolama ~7.4× azalıyor. **Üretim adayı
   olarak 256d/512d önerilir**, 2048d yalnızca üst-sınır referansıdır —
   bu ClickHouse tarafında Faz 2'nin "prefilter maliyeti boyutla
   doğrusal büyür" bulgusuyla doğrudan birleşiyor.

Ayrıca: Qwen-2048, aynı harness'ta X-CLIP'e karşı eşdeğer çıktı (hareket
kategorisinde bile fark yok) — MMEB-V2 liderliğine rağmen. Bu, mevcut
28-sorgu/19-sekans bench'in model kalitesini ayırt etme gücünün sınırlı
olabileceğine işaret ediyor; kesin "model X daha iyi" iddiası öncesi
sorgu/pencere setinin büyütülmesi düşünülmeli.

## Kritik teknik notlar

- Hugging Face `microsoft/xclip` modeli ile Ma vd. retrieval-özel AOSM
  X-CLIP farklıdır. İkisini sonuçlarda tek “X-CLIP” satırında birleştirmeyin.
- SigLIP2 checkpoint'inin resmî config'i `model_type: siglip` kullanır ve
  model kartı doğrudan `AutoModel` önerir. Adı SigLIP2 olsa da `Siglip2Model`'e
  zorlamak patch embedding şekillerini bozar (gerçek çalıştırmada yakalandı);
  adaptör bilinçli olarak `AutoModel` kullanır (`models/siglip_avg.py`).
- `frames_to_intervals` kareyi nokta değil `[i/fps, (i+1)/fps)` aralığı olarak
  ele alır. `+1` düzeltmesi korunmalı; 25 kare/25 fps tam 1 saniyedir.
- `gt_walking`, görüntü düzlemindeki track hareketini kullanır ve ego-motion
  telafisi yapmaz. Gerçek veri görsel denetimi olmadan güvenilir etiket değildir.
- Filtre yanlış negatifleri geri döndürülemez aday kaybı yaratabilir. Saf
  vektör modu bu nedenle karşılaştırma baseline'ı olarak kalmalıdır.
- Üretim ortamı hem air-gapped (ağ çağrısı yok) HEM GPU garantisiz kabul
  edilir. İki kısıt farklı şeyleri eler: (a) runtime'da API çağrısına
  bağımlı hiçbir model/adaptör kullanılamaz — bu zaten `offline_mode`
  (`local_files_only=True`) ile yapısal olarak engelli; (b) CPU, "en kötü
  durum" değil "birincil ölçüm ortamı"dır — bench raporlarında CPU satırı
  zorunlu, GPU satırları (lokal GT 1030, Colab T4 gibi bulut GPU) ek/
  karşılaştırma amaçlıdır, varsayım değildir. Bench koşum manifest'i her
  satırda `hardware_profile` (ör. `cpu`, `gt1030_cuda`, `colab_t4`) alanını
  taşır ki rapor okuyan hangi sayının hangi donanım sınıfına ait olduğunu
  karıştırmasın.
- GT 1030 CUDA denemesi bu oturumda BLOKE: `torch==2.13.0+cu126` kurulumu
  Windows `MAX_PATH` (260 karakter) sınırına takıldı (repo yolu derin +
  torch'un `dist-info/licenses/third_party/kineto/.../duktape-*` gibi çok
  derin iç içe lisans dizinleri). Kök neden `HKLM:\SYSTEM\CurrentControlSet\
  Control\FileSystem\LongPathsEnabled=0`; düzeltme admin yetkisiyle bu
  değeri 1 yapmak (Microsoft'un resmi, geri alınabilir, reboot gerektirmeyen
  önerisi) ama bu ortamdaki shell admin değil ve etkileşimsiz UAC onayı
  alınamıyor. CPU-only `torch==2.13.0+cpu` kurulumu da AYNI limite takıldı;
  wheel'i elle (zipfile ile, yalnızca 5 aşırı-derin 3.-parti LICENSE dosyası
  atlanarak - torch'un kendi kodu/lisansı etkilenmedi) çıkararak eski çalışan
  duruma dönüldü, 54/54 test tekrar geçti. GT1030 CUDA ölçümü "yapılmadı"
  olarak raporlanır; düzeltme tek satır: kullanıcı elle yükseltilmiş
  PowerShell'de `Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\
  Control\FileSystem' -Name LongPathsEnabled -Value 1` çalıştırırsa sonraki
  oturumda tekrar denenebilir.

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
