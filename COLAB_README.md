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
