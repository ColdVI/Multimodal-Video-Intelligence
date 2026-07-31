> **GÜNCELLEME (27 Temmuz 2026):** Bu dosyadaki Faz 4 devam adımları
> tamamlandı (Qwen embedding, MRL kırpma, karşılaştırma, commit'lendi).
> Faz 5 de tamamlandı. Güncel durum ve açık kalanlar için:
> [docs/reports/faz11/FINAL_REPORT.md](../reports/faz11/FINAL_REPORT.md) (nihai rapor) ve
> [docs/operations/STATUS.md](../operations/STATUS.md)/[docs/agents/TASKS.md](TASKS.md)
> (faz bazlı kanıt). Bu dosya artık yalnızca tarihsel/referans amaçlıdır.

# Devir notu — internetsiz devam için (25 Temmuz 2026, oturum sonu)

8 saatlik internet penceresi kapanıyor. Bu dosya, sonraki ~1.5 gün
internetsiz çalışacak oturum için tam durum + kesin sonraki adımlardır.
Plan:
[docs/archive/phases/faz11-development/05_CODEX_BENCHMARK_VE_OPTIMIZASYON_PLANI.md](../archive/phases/faz11-development/05_CODEX_BENCHMARK_VE_OPTIMIZASYON_PLANI.md). İlerleme
kanıtı: `TASKS.md`, `STATUS.md`.

## Tamamlanan ve commit'lenen (Faz 0-3)

Hepsi `git log` içinde ayrı commit olarak var, gerçek veriyle doğrulandı:
- Faz 0: offline_mode desteği, weights_manifest.json.
- Faz 1: `bench/` paketi, 28 sorgu (TR+EN), 19-sekans bench subset gerçek
  ingest edildi, determinizm kontrolü geçti.
- Faz 2: ClickHouse strateji matrisi — **kritik bulgu:** varsayılan
  `vector_search_filter_strategy='auto'` seçici filtrede 100K ölçekte
  0 satır döndürüyor; `prefilter` güvenli. `search/query.py::search()`'e
  `strategy` parametresi eklendi.
- Faz 3: YOLO dedektör bake-off — `yolov8n_visdrone` yeni varsayılan
  (config.yaml: detector.default_variant), COCO'dan ~2× hızlı + downstream
  eşdeğer/daha iyi.

## Şu an devam eden — Faz 4 (COMMIT EDİLMEDİ, YARIM)

**Arka planda çalışıyordu (internet KAPANMADAN önce başladı, model zaten
lokal cache'te — devamı için internet GEREKMİYOR, saf CPU hesaplama):**

```
python ingest/03_embed.py --model qwen3vl_emb_2048
```

Bu, `Qwen/Qwen3-VL-Embedding-2B` ile 19-sekans/73-pencerelik bench subset
için gerçek video embedding üretiyor (frame-average, n_sample=6,
~52sn/pencere ölçüldü → tahmini toplam ~65 dakika). **Bu iş bittiğinde
(veya tekrar başlatılınca) şu adımlar sırayla çalıştırılmalı:**

```powershell
# 1. Embed bitmediyse/kesildiyse tekrar baştan calistir (idempotent, sadece data/embeddings_qwen3vl_emb_2048.json'a yazar):
.venv\Scripts\python.exe ingest\03_embed.py --model qwen3vl_emb_2048

# 2. ClickHouse'a yukle (schema zaten uygulandi: clips_qwen3vl_emb_2048/1024/512/256 tablolari mevcut):
.venv\Scripts\python.exe ingest\05_load_clickhouse.py --model qwen3vl_emb_2048

# 3. MRL kirpma (2048d'den 1024/512/256 turetir - internet gerektirmez, modeli tekrar kosturmaz):
.venv\Scripts\python.exe scripts\mrl_truncate_embeddings.py
.venv\Scripts\python.exe ingest\05_load_clickhouse.py --model qwen3vl_emb_1024
.venv\Scripts\python.exe ingest\05_load_clickhouse.py --model qwen3vl_emb_512
.venv\Scripts\python.exe ingest\05_load_clickhouse.py --model qwen3vl_emb_256

# 4. HF_HUB_OFFLINE=1 ile offline dogrulama (model zaten cache'te, ag cagrisi olmamali):
$env:HF_HUB_OFFLINE=1; .venv\Scripts\python.exe -c "from models import get_embedder; m = get_embedder('qwen3vl_emb_2048'); print(m.embed_text('test').shape)"

# 5. Gercek bench karsilastirmasi (X-CLIP/SigLIP2/Qwen 4 boyut, filtreli, 28 sorgu):
.venv\Scripts\python.exe -c "
from common import load_config
from bench.runner import compute_gt, run_one
from bench.spec import RunSpec
from eval.make_groundtruth import build_query_metadata
from bench.report import aggregate_by_category
import json
cfg = load_config()
gt, seq_ids = compute_gt(cfg, cfg['bench']['subset'])
meta = build_query_metadata()
for model in ['qwen3vl_emb_2048','qwen3vl_emb_1024','qwen3vl_emb_512','qwen3vl_emb_256']:
    spec = RunSpec(model_name=model, use_filters=True)
    result = run_one(spec, cfg, gt, meta)
    print(model, json.dumps(aggregate_by_category(result['rows']), indent=2, ensure_ascii=False))
"

# 6. ClickHouse index boyutlarini oku (depolama ekseni icin):
curl "http://localhost:8123/?query=SELECT%20table,formatReadableSize(data_compressed_bytes)%20FROM%20system.data_skipping_indices%20WHERE%20table%20LIKE%20'clips_qwen%25'"
```

**Sonra:** `TASKS.md`'ye Faz 4 bölümü ekle (Faz 0-3 desenini izle: gerçek
sayılar, ne yapılmadı açıkça işaretli), `STATUS.md`'ye özet, commit.

## İnternetsiz dönemde YAPILMAMASI gerekenler

- Yeni `pip install` / `huggingface_hub` indirmesi denemeyin (network
  hatası alırsınız, hata mesajını "başarısız" diye yanlış yorumlamayın —
  sadece internet yok).
- `torch`/`transformers` sürümüne DOKUNMAYIN. Bu oturumda CUDA torch kurmaya
  çalışırken Windows `MAX_PATH` sınırına takılıp venv'i bozduk (kurtarıldı,
  detay:
  [docs/archive/phases/faz11-development/02_FIKIRLER_VE_KARARLAR.md](../archive/phases/faz11-development/02_FIKIRLER_VE_KARARLAR.md)). `LongPathsEnabled=0`
  registry ayarı hâlâ düzeltilmedi (admin yetkisi + kullanıcı onayı
  gerekiyor) — GT1030 CUDA denemesi tekrar yapılmasın.
- VideoCLIP-XL / LanguageBind_Video entegrasyonu tekrar denenmesin: ikisi
  de doğrulandı ama gerçek engelleri var (VideoCLIP-XL: CC-BY-NC-SA-4.0
  ticari olmayan lisans + özel kod; LanguageBind_Video: `transformers`
  5.14.1 mimariyi tanımıyor, resmi olmayan paket gerekiyor). Detay:
  [docs/archive/phases/faz11-development/02_FIKIRLER_VE_KARARLAR.md](../archive/phases/faz11-development/02_FIKIRLER_VE_KARARLAR.md) Faz 4 bölümü.

## İnternetsiz dönemde YAPILABİLECEKLER (hepsi lokal/cache'ten)

- Yukarıdaki Faz 4 adımlarının tamamı (indirme gerektirmiyor, model zaten
  `.cache/huggingface` içinde).
- Faz 5: `config.yaml`'a `profiles: fast/balanced/accurate` bölümü, gerçek
  Faz 3-4 sonuçlarına dayalı; nihai Pareto raporu
  ([docs/archive/phases/faz11-development/04_KABUL_KRITERLERI_VE_RAPOR.md](../archive/phases/faz11-development/04_KABUL_KRITERLERI_VE_RAPOR.md) şablonu).
- `TASKS.md`'nin kabul kriterleri özetini kanıtlı kutucuklarla güncelle.

## Commit disiplini

Kullanıcı onayı: her faz sonunda commit at (bu oturum boyunca uygulandı).
Bu kural devam etmeli. Şu an working tree'de commit'lenmemiş Faz 4
başlangıç dosyaları var: `models/qwen3vl_emb.py`, `schema.sql` (4 yeni
tablo), `models/__init__.py`, `eval/run_eval.py`, `requirements.txt`,
[docs/archive/phases/faz11-development/02_FIKIRLER_VE_KARARLAR.md](../archive/phases/faz11-development/02_FIKIRLER_VE_KARARLAR.md), `scripts/mrl_truncate_embeddings.py`,
`tests/test_mrl_truncate.py`. Bunlar 87/87 testi bozmuyordu (son doğrulama
oturum sonunda CPU yoğunluğu nedeniyle tamamlanamadı — önce tekrar
`pytest tests/ -v` ile doğrulayın) — bu haliyle commit edilebilir, sonra
Faz 4'ün geri kalanı üstüne eklenir.
