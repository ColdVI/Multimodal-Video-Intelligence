# VideoSearch — web sohbeti devir paketi

> Bu dosya, masaüstü Codex konuşmasının kelimesi kelimesine ham dışa aktarımı
> değildir. Web sohbetinin projeyi eksiksiz devralabilmesi için kullanıcı
> isteklerini, alınan kararları, doğrulanmış sonuçları, açık riskleri ve sonraki
> işleri tek bir kendi-kendine yeterli bağlamda birleştirir. Test edilmemiş hiçbir
> adım tamamlanmış gibi yazılmamıştır.

## Web sohbetine yapıştırılacak başlangıç mesajı

Aşağıdaki metni ve mümkünse bu repoyu web sohbetine ver:

```text
VideoSearch hibrit video arama POC'una devam ediyoruz. Bu mesajın altındaki
teknik bağlamı mevcut gerçek durum kabul et. Önce repo içindeki README.md,
STATUS.md, RESULTS_SMOKE.md, AGENTS.md ve WEB_CHAT_HANDOFF.md dosyalarını oku.

Amacımız videoyu sadece etiketlemek değil; doğal dil sorgusuna karşılık gelen
video ve zaman aralıklarını efficient biçimde bulmak. YOLO yapısal nesne
özelliklerini, X-CLIP/SigLIP2 anlamsal embedding'i, ClickHouse exact filtre +
vector similarity aramasını sağlıyor.

Şu anda 46/46 test geçiyor. Resmî VisDrone2019-MOT verisi doğrulandı ve gerçek
5-sekans/7-pencere smoke hattı iki modelle çalıştı. Ancak tam 56-sekans ingest,
büyük model kalite benchmark'ı ve 1M/10M ClickHouse ölçek testi yapılmadı.

Sonraki ana hedef sistemi efficient hale getirmek. Önce ölçüm altyapısı ve
Fast/Balanced/Accurate çalışma profilleri; sonra GPU batching, kare sayısı ve
window overlap ablation'ları, küçük YOLO karşılaştırması ve tek ana model seçimi
uygulanmalı. Accuracy kaybını ölçmeden performans optimizasyonu yapma.

Her değişiklikte mevcut testleri gerçekten çalıştır, ölçüm üret ve doğrulanmayan
iddiaları açıkça işaretle. Büyük veri/model dosyalarını Git'e commit etme.
```

## 1. Projenin amacı

Kullanıcının asıl hedefi, drone/video arşivinde doğal dille arama yapıp ilgili
zaman aralığını döndüren verimli bir sistem kurmak:

```text
Sorgu: "otobüsün yanında yürüyen insanlar"

Beklenen sonuç:
uav0000138 — 00:16–00:24
bus_count: 1
person_count: 6
semantic_score: 0.87
```

Bu bir video captioning veya yalnızca nesne etiketleme projesi değildir. Sistem
iki farklı sinyali birleştirir:

1. Exact/yapısal sinyal: nesne sayıları ve hareket özellikleri.
2. Semantic sinyal: metin ve video embedding'lerinin yakınlığı.

Temel akış:

```text
Video
  -> zaman pencereleri
  -> YOLO nesne özellikleri + hareket özellikleri
  -> X-CLIP veya SigLIP2 embedding
  -> model başına ayrı ClickHouse tablosu
  -> doğal dil sorgusunu parse et
  -> exact filtre + vector similarity
  -> komşu zaman pencerelerini birleştir
  -> video/zaman/skor sonucu ve rapor
```

## 2. Kullanıcının konuşma boyunca istediği şeyler

- İlk çalışan POC paketinin gerçekten uygulanması ve test edilmesi.
- VisDrone verisinin hangisinin kullanılacağının seçilmesi ve indirilmesi.
- Terminal ağırlıklı olmayan, Colab üzerinde görülebilir bir Control Room.
- Sorgu, dönen videolar, model karşılaştırması, accuracy ve rapor çıktılarının
  ekranda görülmesi.
- MinIO ve ClickHouse ekranlarının ne işe yaradığının açıklanması.
- ClickHouse exact search, similarity search ve hybrid search ayrımının
  gösterilmesi.
- SQL sorgularının test ve rapor için kalıcı tek kaynak olarak saklanması.
- SigLIP2, X-CLIP ve VideoCLIP'in rollerinin açıklanması.
- Sistemi compute, sorgu, depolama ve doğruluk/maliyet açısından efficient hale
  getirecek işlerin belirlenmesi.
- Bütün bağlamın web sohbetine aktarılabilecek Markdown'a dönüştürülmesi ve
  projenin GitHub'a yayınlanması.

## 3. Veri seti kararı

Kullanıcı VisDrone'un DET, VID, SOT, MOT ve Crowd Counting indirmelerini sundu.
Bu projenin video içi nesne, zaman ve track bilgisine ihtiyacı olduğu için
**Task 4: VisDrone-MOT trainset** seçildi.

Doğrulanan resmî veri:

- ZIP boyutu: `8.080.572.990` bayt.
- SHA-256:
  `566d08fb53fff4e539f386f5a408ccf17854fd53814dc756bdede2de1dbb4014`
- 56 sekans.
- 56 annotation dosyası.
- 24.201 JPEG kare.
- Sekans/annotation isim uyuşmazlığı: 0.

Google Drive kota hatası nedeniyle Colab'ın resmî dosyayı doğrudan indirmesi
her zaman güvenilir değildir. Bu nedenle ayrıca gerçek veriden küçük bir smoke
ZIP'i oluşturuldu:

- `VisDrone2019-MOT-smoke-5.zip`
- 5 sekans, 5 annotation, 691 kare.
- Boyut: `288.021.737` bayt.
- SHA-256:
  `50069F3D6C6D7278BDFC40BF05DF52616D08E79E99EE78047E2437AC9D0C369D`

Tam veri veya smoke ZIP GitHub'a commit edilmemelidir.

## 4. Model rolleri

### YOLO

YOLO gerçek nesne algılayıcıdır. Karelerde insan, otobüs, otomobil gibi
nesneleri kutular ve sınıflar. Pipeline bu sonuçlardan `person_count`,
`bus_count` gibi yapısal kolonlar üretir.

SQL'in `bus_count >= 1` koşulu saklanan kolona göre exact'tir; YOLO'nun gerçek
dünyada her otobüsü kusursuz yakaladığı anlamına gelmez.

### X-CLIP

Repodaki model:

```text
microsoft/xclip-base-patch16-zero-shot
model adı: xclip_hf_zeroshot
embedding boyutu: 512
```

Bir video penceresinden 32 kare örnekler, videoyu tek vektöre; metni de aynı
uzaya bir vektöre çevirir. Doğrudan `otobüs` etiketi döndürmez. Video ve metin
vektörlerinin semantic yakınlığıyla sıralama yapılır.

Bu adapter, Microsoft'un Hugging Face zero-shot X-CLIP modelidir; dokümanlarda
geçen retrieval-özel başka X-CLIP varyantlarıyla karıştırılmamalıdır.

### SigLIP2 frame-average

Repodaki model:

```text
google/siglip2-so400m-patch14-384
model adı: siglip2_frameavg
embedding boyutu: 1152
```

SigLIP2 esasen image-text modelidir. Pipeline sekiz kareyi ayrı ayrı embed edip
ortalamasını alır. Bu nedenle baselinedır; kare sırası ve hareketi doğal bir
video modeli kadar iyi temsil etmeyebilir.

### VideoCLIP / VideoCLIP-XL

Gönderilen haftalık sunum ve `docs/codex/XclipVSVideoXClip.md` notunda
VideoCLIP-XL karşılaştırması vardır. Ancak bu repoda VideoCLIP adapter'ı,
checkpoint'i veya ClickHouse tablosu yoktur.

Model registry'de yalnızca:

```text
xclip_hf_zeroshot
siglip2_frameavg
```

bulunur. Sunumu değerlendirmek, modelin bu POC'a uygulanmış olduğu anlamına
gelmez.

## 5. Neden model başına ayrı ClickHouse tablosu var?

X-CLIP 512 boyut, SigLIP2 1152 boyut embedding üretir. İki farklı boyutu aynı
vektör kolonuna koymak HNSW indeks sözleşmesini bozar. Bu yüzden:

```text
clips_xclip_hf_zeroshot
clips_siglip2_frameavg
```

tabloları ayrıdır.

## 6. Exact, similarity ve hybrid search

### Exact search

Vektör kullanmaz; kolon değerlerini deterministik filtreler:

```sql
WHERE bus_count >= 1
  AND person_count >= 1
```

### Exact brute-force vector similarity

Bütün adayların cosine distance değerini hesaplar. HNSW optimizasyonu kapalıdır:

```sql
SETTINGS query_plan_try_use_vector_search = 0
```

### HNSW approximate similarity

Yaklaşık en yakın komşu indeksinin kullanılmasına izin verir:

```sql
SETTINGS query_plan_try_use_vector_search = 1
```

### Hybrid search

Exact nesne/motion filtresi ile semantic vektör sıralamasını aynı sorguda
birleştirir. Repoda hem prefilter hem postfilter + rescore örneği vardır.

SQL'in tek doğru kaynağı `sql/search_lab/` dizinidir. İnsan ClickHouse `/play`
ekranına aynı dosyayı yapıştırır; test ve rapor kodu da aynı dosyayı okur.

Katalog:

1. `01_table_inventory.sql`
2. `02_index_inventory.sql`
3. `03_exact_filter.sql`
4. `04_similarity_exact_bruteforce.sql`
5. `05_similarity_hnsw.sql`
6. `06_hybrid_prefilter.sql`
7. `07_hybrid_postfilter_rescore.sql`

Rapor komutu:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/poc.ps1 search-report
```

Çıktılar:

```text
artifacts/clickhouse_search_report.html
artifacts/clickhouse_search_report.json
```

Gerçek 14 satırlık smoke tablolarında:

- Exact filtre 4 sonuç döndürdü.
- HNSW ve exact brute-force aynı sıralamayı verdi.
- Distance değerleri birebir aynı değildi; maksimum fark `0.0001023`.
- HNSW sorgusu ve postfilter/rescore planı vector indeksini kullandı.
- Exact brute-force ve prefilter planı vector indeksini kullanmadı.

Bu kadar küçük veri, HNSW hız/recall avantajını kanıtlamaz.

## 7. Colab Control Room

Colab için terminal gerektirmeyen Gradio tabanlı dashboard hazırlandı:

```text
notebooks/VideoSearch_Colab_Dashboard.ipynb
notebooks/colab_dashboard.py
```

Sekmeler:

1. Veri/GPU.
2. Pipeline.
3. Query.
4. Accuracy.
5. Rapor.

Kullanıcı ekranda sorguyu, dönen sonuçları, model/filtre metriklerini ve rapor
paketini görebilir. Colab dashboard bellek-içi exact cosine retrieval kullanır;
ClickHouse latency benchmark'ı değildir. Yerel Docker hattı ClickHouse mimari
kanıtını sağlar.

Colab çalışma sırası:

```text
1. GPU runtime seç.
2. video-search-poc-colab.zip paketini yükle ve notebook kurulum hücrelerini çalıştır.
3. Control Room'u aç.
4. VisDrone2019-MOT-smoke-5.zip yükle veya Drive yolu ver.
5. Veri/GPU -> Pipeline -> Query -> Accuracy -> Rapor sırasını izle.
```

## 8. MinIO ve ClickHouse'un rolü

- ClickHouse: metadata, embedding, exact filtre, cosine similarity ve HNSW
  sorgularını çalıştırır.
- MinIO: üretim mimarisinde video/segment/frame gibi binary objeler için object
  storage katmanıdır. Mevcut küçük smoke retrieval'ın kritik darboğazı değildir.
- ClickHouse satırında video kimliği ve zaman aralığı tutulur; büyük video
  dosyasını veritabanına gömmek yerine object storage referansı kullanılabilir.

## 9. Gerçek smoke çalışmasının kanıtı

Kapsam:

- 5 gerçek VisDrone sekansı.
- 691 kare.
- 7 kayan pencere.
- 7 YOLO özellik satırı.
- 7 adet 512d X-CLIP embedding.
- 7 adet 1152d SigLIP2 embedding.
- Her modelin ClickHouse tablosunda 7 satır.
- 6 ground-truth sorgusu.
- İki model x iki filtre için 24 özet ve 98 detay eval satırı.

Ölçülen CPU süreleri:

| Aşama | Süre |
|---|---:|
| Frames -> MP4 | 23,4 sn |
| YOLO | 59,9 sn |
| X-CLIP | 158,2 sn |
| SigLIP2 | 311,6 sn |
| İki-model eval | 42,4 sn |

Filtre smoke özeti:

| Filtre | Kategori | Precision | Recall |
|---|---|---:|---:|
| Açık | hareket | 1,000 | 1,000 |
| Açık | tekli | 1,000 | 0,889 |
| Açık | bileşik | 1,000 | 0,800 |
| Kapalı | hareket | 1,000 | 1,000 |
| Kapalı | tekli | 0,600 | 1,000 |
| Kapalı | bileşik | 0,500 | 1,000 |

Bu sonuç, filtrelerin precision/recall trade-off'unu gösterir. Beş video
`top_k=10`'dan küçük olduğu için X-CLIP ile SigLIP2 kalite şampiyonunu seçmek
için yeterli değildir.

## 10. Gerçek çalıştırmada bulunan hatalar

- Ground-truth interval dönüşümünde off-by-one hatası bulundu ve düzeltildi.
- 25 karelik bir tespit `(N-1)/fps` nedeniyle 0,96 saniye hesaplanıp 1 saniye
  eşiğinde sessizce silinebiliyordu.
- Odd-height 1904x1071 karelerde libx264/yuv420p hatası düzeltildi.
- ClickHouse HTTP multi-statement schema yükleme problemi düzeltildi.
- Windows `cp1252` Türkçe çıktı problemi düzeltildi.
- Transformers 5.x pooled output tipi değişikliği iki adapterda düzeltildi.
- SigLIP2 checkpoint'ini yanlış sınıfa zorlama problemi `AutoModel` ile düzeltildi.
- FiftyOne temporal support video sonunu aşınca manifest frame count'a clamp edildi.
- Farklı embedding boyutlarının aynı HNSW kolonunda tutulması tasarım hatası,
  model başına ayrı tabloyla düzeltildi.

## 11. Test durumu

24 Temmuz 2026 tarihinde:

- `46/46` pytest geçti.
- 40 Python dosyası `py_compile` kontrolünden geçti.
- ClickHouse `26.7.1.1315` gerçek sunucuda kullanıldı.
- X-CLIP ve SigLIP2 gerçek checkpoint inference/load/eval smoke'u geçti.
- YOLO gerçek inference smoke'u geçti.

## 12. Bilinen sınırlar

- Tam 56-sekans model ingest yapılmadı.
- Mevcut kalite kanıtı 5 sekans/7 penceredir.
- `gt_walking` için 5–10 sekanslık insan gözüyle ego-motion denetimi yapılmadı.
- Büyük model karşılaştırması yapılmadı.
- 1M/10M ClickHouse ölçek testi yapılmadı.
- Yerel Torch CPU-only çalışıyor; tam turlar GPU'lu ortamda yapılmalı.
- Mevcut SQL Search Lab self-probe vektörü kullanır; doğal dil kalite benchmark'ı
  değildir.

## 13. Sistemi efficient hale getirme planı

Efficiency tek başına hız değildir. Karar metrikleri:

```text
Recall@K / Precision@K
saniye / video dakikası
GPU belleği
embedding MB / video saati
query p50 / p95 latency
```

Önerilen uygulama sırası:

### A. Ölçüm altyapısı

- Decode, YOLO, embedding, load ve query aşamalarını ayrı zamanla.
- GPU peak memory, batch throughput ve disk çıktısını kaydet.
- Aynı ground truth üzerinde Fast/Balanced/Accurate profilleri üret.

### B. GPU batching ve mixed precision

- X-CLIP video pencerelerini batch işle.
- SigLIP2 karelerini batch işle.
- YOLO karelerini batch işle.
- T4 belleğine göre otomatik batch boyutu belirle.
- Doğrulayarak FP16/BF16 kullan.

### C. Daha az tekrar

Mevcut pencere:

```yaml
size_s: 8
stride_s: 4
```

Karşılaştırılacak profiller:

| Profil | Window / stride |
|---|---|
| Fast | 8 / 8 |
| Balanced | 8 / 6 |
| Accurate | 8 / 4 |

### D. Kare ablation'ı

- X-CLIP: 8 / 16 / 32 kare.
- SigLIP2: 4 / 8 kare.
- Accuracy ciddi düşmeden en ucuz seçeneği seç.

### E. Model maliyetini azalt

- X-CLIP ve SigLIP2'yi üretimde sürekli beraber çalıştırma.
- Büyük eval sonrası tek ana model seç.
- Gerekirse ikinci modeli yalnızca top-N reranker yap.
- YOLO26x yerine nano/small varyantlarını nesne-count doğruluğuyla karşılaştır.

### F. Decode/cache/idempotency

- Videoyu bir kez decode edip aynı kareleri YOLO ve embedding modelleriyle paylaş.
- Video hash'i ve run manifest sakla.
- Değişmeyen videoyu yeniden ingest etme.
- Kesilen işin kaldığı pencereden devam et.
- Statik/çok benzer pencereleri perceptual hash veya scene-change ile birleştir.

### G. Retrieval cascade

```text
Exact metadata filtresi
  -> HNSW top-100
  -> güçlü modelle top-20 rerank
  -> top-10 + interval merge
```

### H. Query efficiency

- Normalize edilmiş tekrarlı metin embedding'lerini cache'le.
- Küçük top-k aday setini rerank et.
- ClickHouse optimizasyonunu ancak anlamlı satır ölçeğinde yap.

### I. Ölçek testi

- 100K, 1M ve 10M satırda exact brute-force/HNSW/hybrid pre/postfilter ölç.
- Küçük 7-satır smoke sonucunu üretim latency kanıtı sayma.

## 14. Önerilen sonraki sprint

1. `benchmark_pipeline.py` benzeri tekrar üretilebilir benchmark runner.
2. `fast`, `balanced`, `accurate` config profilleri.
3. Colab T4 üzerinde batch inference.
4. X-CLIP kare sayısı ve stride deney matrisi.
5. Sonuçları tek HTML/JSON efficiency raporunda birleştirme.
6. Accuracy/maliyet Pareto tablosuna göre ana model/profil seçimi.

## 15. Repo dosya haritası

- `README.md`: proje ve hızlı başlangıç.
- `STATUS.md`: güncel doğrulanmış durum.
- `RESULTS_SMOKE.md`: gerçek VisDrone smoke sonuçları.
- `AGENTS.md`: coding agent kuralları ve doğrulanan/doğrulanmayan işler.
- `CONTEXT.md`: mimari karar gerekçeleri.
- `TASKS.md`: faz bazlı görevler.
- `COLAB_README.md`: görsel Colab Control Room kullanımı.
- `notebooks/VideoSearch_Colab_Dashboard.ipynb`: Colab giriş noktası.
- `models/`: X-CLIP ve SigLIP2 adapterları.
- `ingest/`: decode/window/detect/embed/load hattı.
- `search/`: parser, ClickHouse query, merge ve SQL kataloğu.
- `sql/search_lab/`: exact/similarity/hybrid SQL tek kaynağı.
- `reports/`: ClickHouse HTML/JSON raporlayıcı.
- `eval/`: ground truth, metrikler ve model/filtre eval.
- `tests/`: 46 test.
- `docs/codex/`: hazır konuşmalar, plan ve kabul kriterleri.

## 16. GitHub'a konmaması gerekenler

`.gitignore` tarafından dışarıda tutulması gerekenler:

```text
data/raw/
data/downloads/
data/embeddings_*.json
data/windows.json
data/features.json
data/groundtruth/
results.json
results_detail.json
.venv/
.runtime/
.testdeps/
weights/
runs/
datasets/
*.pt
__pycache__/
*.pyc
```

Özellikle şunlar GitHub'a yüklenmemeli:

- 8,08 GB resmî VisDrone ZIP.
- 274 MB smoke ZIP.
- `yolo26x.pt`.
- Hugging Face model cache'leri.
- `.venv` ve çalışma cache'leri.

## 17. Yeniden üretim komutları

Test:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/poc.ps1 test
```

Altyapı:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/poc.ps1 infra-up
powershell -ExecutionPolicy Bypass -File scripts/poc.ps1 schema
```

Veri doğrulama:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/poc.ps1 download-data
```

ClickHouse Search Lab raporu:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/poc.ps1 search-report
```

## 18. Web sohbetinden beklenen çalışma biçimi

- Kod ve test sözleşmesini dokümandan önce doğru kaynak kabul et.
- Test edilmemiş bir adımı tamamlanmış yazma.
- Efficiency değişikliklerini accuracy ile beraber ölç.
- Tam CPU ingest gibi saatler sürecek işleri ölçüm/onay olmadan başlatma.
- Veri/model ağırlığı veya secret commit etme.
- Değişiklikten sonra testleri gerçekten çalıştır ve sonucu raporla.
- Model kalite kararı için beş videoluk smoke metriğini kullanma.

