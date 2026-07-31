# Hibrit video arama — POC

## Faz 7: çalışan sistem

GPU gerektirmeyen varsayılan `synthetic` modla Postgres/pgvector,
ClickHouse, Qdrant, FastAPI ve Gradio birlikte ayağa kalkar:

```bash
cp .env.faz7.example .env.faz7
docker compose -f docker-compose.faz7.yml up -d --build
docker compose -f docker-compose.faz7.yml exec -T api python -m app.ingestion.load_dataset --dataset auair
```

- UI: <http://localhost:7860>
- API/OpenAPI: <http://localhost:8000/docs>
- Sağlık: <http://localhost:8000/health>

UI'daki kırmızı banner bilinçlidir: `synthetic` sonuç sıralamaları semantik
kalite iddiası taşımaz; yalnız sistem ve gecikme doğrulamasıdır. Gerçek Qwen
embedding üretimi `notebooks/07_colab_embedding_production.ipynb`, cached moda
geçiş ve tam işletim adımları `docs/getting-started/COLAB_RUNBOOK.md`
dosyasındadır.

## Dokümantasyon

- [Hızlı başlangıç](docs/getting-started/OPERATOR_QUICKSTART.md)
- [Kendi datasetini ekleme](docs/datasets/DATASET_ONBOARDING_GUIDE.md)
- [Colab kullanımı](docs/getting-started/COLAB_RUNBOOK.md)
- [Güncel mimari](docs/architecture/CURRENT_SYSTEM.md)
- [FAZ11 final raporu](docs/reports/faz11/FINAL_REPORT.md)
- [Tüm dokümanlar](docs/README.md)

> Codex ile devam edecekseniz önce [docs/agents/START_HERE.md](docs/agents/START_HERE.md) dosyasını açın.
> Prompter ve handoff notları `docs/agents/prompts/` ve `docs/agents/`
> altındadır.

Dogal dil sorgusundan (`"otobus ve yuruyen adam"`) video zaman araligina
(`uav0000086 0:00:12-0:00:41 (skor 0.91)`) giden hibrit retrieval hattinin
kucuk olcekli, acik-veri dogrulamasi.

 Arka plan ve tasarim kararlarinin gerekcesi: `docs/architecture/CURRENT_SYSTEM.md`
 Coding agent talimatlari (ne test edildi ve ne edilmedi): `docs/agents/AGENT_INSTRUCTIONS.md`
 Faz bazli gorev listesi: `docs/agents/TASKS.md`
 Guncel uygulanmis durum ve olcumler: `docs/operations/STATUS.md`
 Terminal gerektirmeyen Colab dashboard: `docs/getting-started/COLAB_README.md`
 Web sohbetine tek mesajlik eksiksiz devir baglami: `docs/agents/WEB_CHAT_HANDOFF.md`
 CPU / GT 1030 / Tesla T4 ve model benchmark'i: `docs/operations/benchmarks/BENCHMARK_CPU_GT1030_T4.md`

## Colab GPU + gorsel Control Room
 Portable yukleme paketi
 `video-search-poc-colab.zip` olarak teslim edilir; kullanim adimlari
 `docs/getting-started/COLAB_README.md` dosyasindadir.
 Model basina ayri ClickHouse tablosu (`clips_<model>`) — nedeni
  `docs/architecture/CURRENT_SYSTEM.md`.
 ClickHouse, gercek X-CLIP inference ve YOLO gerçek VisDrone smoke testleri gecti;
  tam kanit ve sureler `docs/operations/STATUS.md`'de.
 SigLIP2 gerçek 1152d inference/load/eval geçti; 5-videolu smoke model kalite
  ayrımı için yetersizdir (`docs/operations/benchmarks/RESULTS_SMOKE.md`).
 Detektor kolonlari (`person_count` vb.) uretimdeki telemetrinin POC
  vekilidir; gercek IHA verisine geciste degisecek tek katman budur
  (bkz. `docs/architecture/CURRENT_SYSTEM.md`, Faz 5).

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
- Model basina ayri ClickHouse tablosu (`clips_<model>`) — nedeni
  `docs/architecture/CURRENT_SYSTEM.md`.

## Bilinen sinirlar

- ClickHouse, gercek X-CLIP inference ve YOLO gerçek VisDrone smoke testleri gecti;
  tam kanit ve sureler `docs/operations/STATUS.md`'de.
- VisDrone verisi hazır; 5 gerçek sekans/7 pencere iki-model ingest/GT/eval geçti. Tam 56-sekans
  ingest ve FiftyOne görsel denetimi çalıştırılmadı.
- SigLIP2 gerçek 1152d inference/load/eval geçti; 5-videolu smoke model kalite
  ayrımı için yetersizdir (`RESULTS_SMOKE.md`).
- Bu makinedeki Torch CPU-only; tam veri turlari icin GPU'lu ortam onerilir.
- Detektor kolonlari (`person_count` vb.) uretimdeki telemetrinin POC
  vekilidir; gercek IHA verisine geciste degisecek tek katman budur
  (bkz. `docs/architecture/CURRENT_SYSTEM.md`, Faz 5).
