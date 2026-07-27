# VideoSearch Colab Control Room

Bu teslim, terminal komutu yazmadan GPU pipeline, sorgu inceleme, accuracy ve
rapor disari aktarma icin hazirlandi.

## Acilis

1. `notebooks/VideoSearch_Colab_Dashboard.ipynb` dosyasini Google Colab'de ac.
2. Calisma zamani turunu **GPU** yap.
3. `Calisma zamani > Tumunu calistir` sec.
4. Ilk dosya secicisinde `video-search-poc-colab.zip` paketini yukle.
5. Kurulum bitince acilan bes sekmeyi sirayla kullan.

## Sekmeler

- **Veri/GPU:** GPU'yu gosterir; yaklasik 277 MB'lik raporluk
  `VisDrone2019-MOT-smoke-5.zip`, tam VisDrone ZIP'i veya resmi Google Drive
  indirmesini dogrular.
- **Pipeline:** 5/10/20/56 video ve model secimiyle frames-to-video, windowing,
  YOLO, embedding ve ground truth adimlarini calistirir.
- **Query:** Turkce sorguyu, cikan filtreleri, aday sayisini, skoru, zaman
  araligini ve uc karelik gorsel onizlemeyi birlikte gosterir.
- **Accuracy:** Model x filtre icin sorgu-bazli ve toplu precision@k/recall@k
  tablosu ile grafik uretir.
- **Rapor:** HTML ozet, CSV/JSON kanitlari, kosu manifesti ve metodolojik
  uyarilari tek ZIP'e koyar; isterse Google Drive'a kopyalar.

## Dürüstlük sınırı

Notebook aramayi `exact_in_memory_cosine` backend'iyle yapar. Model ve filtre
kalitesini goruntulemek icindir; ClickHouse gecikme benchmark'i degildir. HTML
rapor bu ayrimi otomatik yazar. ClickHouse mimari/latency testi yerel Docker
hattinda veya ayri bir ClickHouse sunucusunda yapilmalidir.

Varsayilan **Raporluk 5** smoke kapsami entegrasyon kanitidir. `top_k=10`, bes
videoyu doyurabildigi icin model zaferi iddiasi icin kullanilamaz. Model
karsilastirma raporu icin video ve sorgu sayisini artir.

## Faz 4 GPU hiz olcumu (yeni, 27 Temmuz 2026)

Yerel gelistirme makinesi CPU-only; Qwen3-VL-Embedding-2B gibi buyuk
modellerin gercek GPU hizi bu depoda henuz olculmedi (agent Colab'i canli
suremiyor - tarayici/API erisimi yok). Bunu siz Colab'de kendiniz
calistirabilirsiniz:

1. Yerel makinede: `python scripts/package_colab_gpu_bundle.py` — bench
   subset videolarini, pencereleri, YOLO ozelliklerini ve dedektor
   checkpoint'lerini `artifacts/colab_gpu_bundle.zip`'e toplar (model
   agirliklari DAHIL DEGIL - Colab'de internet var, HF'den kendisi iner).
2. Colab'de GPU runtime ile bu repoyu acin, `colab_gpu_bundle.zip`'i
   yukleyip repo kokune cikarin.
3. `!pip install -q sentence-transformers qwen-vl-utils`
4. `!python scripts/colab_gpu_bench.py`
5. Cikan `artifacts/colab_gpu_bench.json`'i indirip depoya geri getirin;
   sonuclari `STATUS.md`/`TASKS.md` Faz 4'e elle ekleyin.

**Not:** `scripts/colab_gpu_bench.py` sadece embedding + YOLO dedeksiyon
HIZINI olcer (kalite zaten cihazdan bagimsizdir); ClickHouse gerektirmez -
Colab'de gecici bir ClickHouse kurmak kirilgan olurdu. Bu script yazildi
ama gercek bir GPU'da test edilmedi - kodun kendisi zaten CPU'da
dogrulanmis fonksiyonlari (`bench/timing.py`, `models.get_embedder`)
kullanir, ama uctan uca kosum kanitlanmadi.
