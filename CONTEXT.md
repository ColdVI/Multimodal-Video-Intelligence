# Baglam ve tasarim kararlari

Bu POC, bir vendor/danisman raporunun onerdigi hibrit video arama
mimarisini (telemetri/filtre kolonlari + klip-embedding + LLM ayristirici),
gercek IHA verisi olmadan, acik drone veri setleriyle sinamak icin var.

## Rapor incelemesinden dogrulanan iddialar

- X-CLIP isim cakismasi gercek: HuggingFace'teki `microsoft/xclip` (Ni ve
  ark., Kinetics siniflandirma) ile Ma ve ark.'in (2022, ACM MM, AOSM
  modulu, retrieval-ozel) X-CLIP'i FARKLI modeller.
- "Bagimsiz test" olarak referans verilen NDCG@10 sayilari (X-CLIP 0.47,
  InternVideo2 0.30, kapali modeller 0.75+) gercek ve dogrulanabilir -
  kaynagi Mixpeek'in Mart 2026 video-embedding-benchmark'i. AMA: 20 stok
  video (spor/yemek/doga/kentsel/teknoloji, <=10sn), IHA goruntusu degil.
  Sayi transfer etmeyebilir - bu POC'un varlik sebeplerinden biri.
- ClickHouse'un HNSW indeksi buyuk vektor sayisinda "gercek ANN" degil
  "filtrelenmis kaba kuvvet" gibi davranabilir (canli bir GitHub issue'da
  raporlanmis). POC'un olcek testi (TASKS.md Faz 4) bunu kucuk olcekte
  erken sinar.

## Acik sorular (POC bunlari KAPATMAZ, uretime tasinmadan once cevaplanmali)

1. **Telemetri formati**: rapor pymavlink/MAVLink varsayiyor. Askeri ISR
   platformlari genelde STANAG 4609/MISB KLV kullanir (telemetri videonun
   icine gomulu, ayri log dosyasi degil). Bu, uretimdeki telemetri-isleme
   adiminin hangi kutuphaneyle yazilacagini belirliyor — POC'ta bu adim
   YOK, cunku VisDrone'da telemetri kavrami yok, detektor kolonlariyla
   vekillendirdik.
2. **Platform/IHA tipi filtresi**: sorgudaki "TB2" gibi unsurlar muhtemelen
   zaten kurumun kendi metadata katalogunda var, embedding'e hic
   sorulmamali. POC'ta `platform` kolonu sabit `'visdrone'` — gercek
   coklu-platform senaryosu burada yok, Faz 5'te ayri bir katalog join'i
   olarak eklenecek.
3. **150-sorgu istatistiksel guc esigi (28 Temmuz 2026)**: VisDrone'da
   `eval/make_groundtruth.py::build_queries()` 28 sorgu uretiyor - bu
   esigin ALTINDA. Adaptive MRL harness'i (bkz. STATUS.md "Unified Search
   Harness") bu 28 sorguyla GERCEKTEN calistirildi ve gercek sayilar
   uretti, ama bu PILOT bir olcum, baglayici bir "X boyutu Y'den iyi"
   uretim karari DEGIL - kucuk n'de gurultu buyuk boyut farklarini
   maskeleyebilir/taklit edebilir. Baglayici karar icin ya VisDrone GT'si
   150+ sorguya genisletilmeli ya da MSR-VTT'nin 1000 sorguluk GPU
   kosumu (Faz 6, henuz calistirilmadi - bu makinede GPU yok) tamamlanmali.
   Bu esik gecilmeden planner (Faz 7) esikleri veya dashboard'a yeni
   sekmeler EKLENMEMELI - tahminle deger yazmak yerine bilincli olarak
   ertelendi.

## POC'ta bilincli basitlestirmeler

- **Tek tablo yerine model-basina-tablo** (`clips_<model_adi>`): rapor tek
  tabloda "cift kolon" oneriyor (embedding_xclip, embedding_siglip gibi
  ayri kolonlar ayni satirda). POC'ta bake-off'ta 2+ farkli boyutlu model
  (X-CLIP 512d, SigLIP2 1152d) ayni anda test edildigi icin, tek kolonlu
  bir sema HNSW indeksinin boyutunu tek bir degere kilitler ve ikinci
  modelin vektorleri bozuk indekslenir. Model-basina-tablo bunu basitce
  onluyor; uretimde satir-basina tek model olacagi icin rapordaki "cift
  kolon" deseni orada dogru kalir.
- **Kural tabanli ayristirici, LLM degil**: sorgu uzayi dar (~10-15 kavram),
  kural yeterli ve hata ayiklamasi trivial. Arayuz LLM'e geciste hazir.
- **PostgreSQL yok**: yalnizca `platform` sabit deger, gercek bir katalog
  join'i yok (docker-compose.yml'de bilerek yok — kullanilmayan servis
  eklemek yeni bir "olu referans" olurdu).
  **YENIDEN DOGRULANDI (28 Temmuz 2026)**: bir dis plan (Codex "Unified
  Search Harness") Postgres'i "authoritative segment_features + ClickHouse
  materyalize kopya + parity/checksum" olarak onerdi. Sorunun kendisini
  (iliskisel butunluk gercekten gerekli mi) inceledik, varsayimla
  gecistirmedik:
  - Model -> tablo eslemesi zaten depolama gerektirmeyen bir isimlendirme
    kurali: `search/query.py:76`, `table = f"clips_{model_name}"`.
  - Segment kimligi zaten `(video_id, t_start)` demeti ile calisiyor ve
    tablolar-arasi karsilastirma icin YETERLI - `reports/
    strategy_matrix_report.py::hnsw_recall_at_k()` bunu FK/UUID olmadan,
    duz Python kumesi kesisimiyle yapiyor.
  - `brightness`/`camera_motion` gibi "segment_features" zaten ClickHouse
    satirinin kendisinde (`schema.sql`) - ayri bir authoritative kaynak
    yok, dolayisiyla senkronize edilecek/parity'si bozulacak iki kopya
    da yok.
  - MSR-VTT (bu oturumda eklenen ikinci veri seti) BILEREK ClickHouse'a
    hic yazmiyor (`scripts/validate_msrvtt.py` docstring'i) - coklu-dataset
    genislemesi bile yeni bir katalog-join ihtiyaci DOGURMADI.
  Sonuc: mevcut ihtiyaç tek-ClickHouse-tablosuyla (duz, join'siz) cozuluyor;
  Postgres eklenmedi (schema/ephemeral-test dahil hicbir kod yazilmadi).
  Karar hala gecerli - "henuz ele alinmamis" degil, "yeniden incelenip
  teyit edilmis".
- **Kafka/Temporal yok**: orkestrasyon dayanikliligi uretim problemi,
  POC'un sorusu "retrieval calisiyor mu". `make ingest` yeterli.

## Genelden ozele sorgu ilerleyisi

`eval/make_groundtruth.py::build_queries()` su sirayla zorlasiyor: tekli
statik nesne ("otobusu goster") -> hareket kavrami ("yuruyen adami goster",
saf embedding'in zayif oldugu yer) -> bilesik ("otobus ve yuruyen adam",
filtre+embedding'in birlikte calismasi gereken yer). Bu, raporun sorgu
siniflandirmasinin kucuk olcekli karsiligi.

## Bu POC'ta bulunup duzeltilen bir hata (ornek)

`frames_to_intervals`'in ilk halinde N ardisik True kare (N-1)/fps sureye
duşuyordu (frame indekslerini nokta gibi degil, aralik gibi ele almamaktan
kaynaklanan klasik off-by-one) — 25 kare, 25fps'te tam 1.0sn olmasi
gerekirken 0.96sn cikiyor ve varsayilan 1.0sn min_dur esigini
yanlislikla kirpiyordu. Gercek pytest calistirmasinda
(`test_gt_object_finds_bus_frames`, `test_gt_walking_detects_displacement`)
yakalanip duzeltildi. Bunu not dusuyoruz cunku "test yazdim" ile "testleri
calistirip en az bir gercek hata buldum" arasindaki fark
onemli.
