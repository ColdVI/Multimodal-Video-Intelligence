# Hibrit video arama — POC

> Codex ile devam edecekseniz önce `CODEX_START_HERE.md` dosyasını açın.
> Tek seferlik ana prompt ve fazlara ayrılmış hazır konuşmalar
> `docs/codex/` altındadır.

Dogal dil sorgusundan (`"otobus ve yuruyen adam"`) video zaman araligina
(`uav0000086 0:00:12-0:00:41 (skor 0.91)`) giden hibrit retrieval hattinin
kucuk olcekli, acik-veri dogrulamasi.

- Arka plan ve tasarim kararlarinin gerekcesi: `CONTEXT.md`
- Coding agent talimatlari (ne test edildi ve ne edilmedi): `AGENTS.md`
- Faz bazli gorev listesi: `TASKS.md`
- Guncel uygulanmis durum ve olcumler: `STATUS.md`
- Terminal gerektirmeyen Colab dashboard: `COLAB_README.md`
- Web sohbetine tek mesajlik eksiksiz devir baglami: `WEB_CHAT_HANDOFF.md`
- CPU / GT 1030 / Tesla T4 ve model benchmark'i: `BENCHMARK_CPU_GT1030_T4.md`

## Colab GPU + gorsel Control Room

`notebooks/VideoSearch_Colab_Dashboard.ipynb` dosyasi Colab'de butonlarla su
akisi verir: VisDrone dogrulama/indirme -> GPU inference pipeline -> gorsel
sorgu sonuclari -> model x filtre accuracy -> HTML/CSV/JSON rapor paketi.
Terminal komutu yazmak gerekmez. Portable yukleme paketi
`video-search-poc-colab.zip` olarak teslim edilir; kullanim adimlari
`COLAB_README.md` dosyasindadir.

Colab dashboard exact bellek-ici cosine arama kullanir ve rapora bunu acikca
yazar. ClickHouse gecikme benchmark'i degildir; ClickHouse mimari testi yerel
Docker hattinda kalir.

## Hemen dogrulanabilir kisim (veri/GPU/ClickHouse gerekmez)

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File scripts/poc.ps1 test
```

Linux/macOS:

```bash
pip install -r requirements.txt
make test
```

46 test, saf mantik ve veri/SQL sözleşmeleri (interval birlestirme, temporal IoU, sorgu ayristirma,
ground-truth turetme). Bu depoyu teslim etmeden once calistirdim, hepsi
gecti - detay AGENTS.md'de.

## Tam kurulum

```bash
pip install -r requirements.txt
make infra-up      # ClickHouse + MinIO
make schema

make download-data # resmî Task 4 MOT trainseti + boyut/SHA/veri sözleşmesi

make ingest MODEL=xclip_hf_zeroshot
make ingest MODEL=siglip2_frameavg
make groundtruth
make eval
make fiftyone       # sonuclari gozle incele
```

## ClickHouse Search Lab

Exact kolon filtresi, exact brute-force vector search, HNSW, hybrid prefilter
ve postfilter/rescore SQL'lerinin tek kaynagi `sql/search_lab/` dizinidir.
ClickHouse `/play` ekraninda bu dosyalar kopyalanabilir; test ve rapor kodu da
ayni SQL'i okur.

```bash
make search-report
```

Windows'ta `scripts/poc.ps1 search-report` ayni isi yapar. Ciktilar
`artifacts/clickhouse_search_report.html` ve `.json` dosyalaridir. Rapor her
sorgunun sonucunu, sunucu/HTTP suresini, aktif vector ayarlarini ve
`EXPLAIN indexes = 1` planini saklar.

Windows'ta ayni gorevler:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/poc.ps1 infra-up
powershell -ExecutionPolicy Bypass -File scripts/poc.ps1 schema
powershell -ExecutionPolicy Bypass -File scripts/poc.ps1 download-data
# Önce tek gerçek sekans smoke'u:
powershell -ExecutionPolicy Bypass -File scripts/poc.ps1 ingest -Model xclip_hf_zeroshot -Sequence uav0000138_00000_v
```

## Neden bu yapi

- `models/` — her embedding modeli ayni arayuzu (`VideoTextEmbedder`) uygular;
  `models/__init__.py`'deki registry'e bir satir ekleyerek yeni model eklenir.
- `search/parser.py` — kural tabanli; arayuzu (`ParsedQuery`) sabit tutarak
  LLM tabanli ayristiriciya geciste yalnizca bu dosyanin ici degisecek.
- `eval/make_groundtruth.py` — VisDrone'un kutu+track anotasyonundan sorgu
  bazli ground truth otomatik turetiyor, elle etiketleme yok.
- Her sayisal sabit (pencere boyutu, gap tolerance, IoU esigi) `config.yaml`'da.
- Model basina ayri ClickHouse tablosu (`clips_<model>`) — nedeni CONTEXT.md.

## Bilinen sinirlar

- ClickHouse, gercek X-CLIP inference ve YOLO gerçek VisDrone smoke testleri gecti;
  tam kanit ve sureler `STATUS.md`'de.
- VisDrone verisi hazır; 5 gerçek sekans/7 pencere iki-model ingest/GT/eval geçti. Tam 56-sekans
  ingest ve FiftyOne görsel denetimi çalıştırılmadı.
- SigLIP2 gerçek 1152d inference/load/eval geçti; 5-videolu smoke model kalite
  ayrımı için yetersizdir (`RESULTS_SMOKE.md`).
- Bu makinedeki Torch CPU-only; tam veri turlari icin GPU'lu ortam onerilir.
- Detektor kolonlari (`person_count` vb.) uretimdeki telemetrinin POC
  vekilidir; gercek IHA verisine geciste degisecek tek katman budur
  (bkz. CONTEXT.md, Faz 5).
