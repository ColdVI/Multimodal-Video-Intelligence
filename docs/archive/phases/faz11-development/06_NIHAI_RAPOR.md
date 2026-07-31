# Nihai rapor — Faz 0-5 benchmark ve optimizasyon

> `docs/codex/04_KABUL_KRITERLERI_VE_RAPOR.md` şablonunun yapısını izler;
> içerik `docs/codex/05_CODEX_BENCHMARK_VE_OPTIMIZASYON_PLANI.md` Faz 0-5
> çalışmasının gerçek sonuçlarıdır. Tarih: 24-27 Temmuz 2026.

## 1. Deney kimliği

- Tarih: 24-27 Temmuz 2026 (tek oturum, ~8 saatlik aktif internet penceresi
  + arka planda süren uzun CPU işleri).
- Commit'ler: `first commit` → `Faz 4: Qwen3-VL-Embedding sonuçları` (bkz.
  `git log --oneline`, sekiz commit, her faz ayrı).
- OS/Python: Windows, Python 3.14.6.
- CPU/GPU: CPU-only bu oturumda. NVIDIA GeForce GT 1030 (4GB) fiziksel
  olarak mevcut ama CUDA torch kurulumu Windows `MAX_PATH` sınırına takıldı
  (`LongPathsEnabled=0`, admin yetkisi gerekiyor, onay alınmadı). Colab/GPU
  ölçümü bu oturumda yapılamadı — `scripts/colab_gpu_bench.py` hazır ama
  test edilmedi.
- Docker/ClickHouse: `clickhouse/clickhouse-server` 26.7.1.1315, MinIO.
- Model kimlikleri: `microsoft/xclip-base-patch16-zero-shot`,
  `google/siglip2-so400m-patch14-384`, `Qwen/Qwen3-VL-Embedding-2B`,
  `yolo26x.pt` (COCO), `mshamrai/yolov8n-visdrone`, `mshamrai/yolov8s-visdrone`.
- Veri: VisDrone2019-MOT-train, 56 sekans mevcut; bench subset 19 sekans
  (`config.yaml: bench.subset`), 73 pencere, 28 sorgu (TR+EN, 5 kategori).

## 2. Yeniden üretim komutları

```powershell
powershell -ExecutionPolicy Bypass -File scripts\poc.ps1 test
powershell -ExecutionPolicy Bypass -File scripts\poc.ps1 bench
python scripts\run_strategy_matrix_report.py
python scripts\run_detector_bakeoff.py
python scripts\build_scale_table.py --model xclip_hf_zeroshot --n 100000
python scripts\run_scale_evidence.py --table bench_scale_512
python ingest\03_embed.py --model qwen3vl_emb_2048
python scripts\mrl_truncate_embeddings.py
```

Tam komut geçmişi ve zamanlamalar `STATUS.md`/`TASKS.md` faz bölümlerinde.

## 3. Faz özeti tablosu

| Faz | Kanıt | Durum |
|---|---|---|
| 0 | offline_mode + weights_manifest.json (6 checkpoint, SHA-256) | Tamamlandı |
| 1 | `bench/` paketi, 28 sorgu, 19 sekans/73 pencere, determinizm GEÇTİ | Tamamlandı |
| 2 | Strateji matrisi + 100K ölçek: R1 doğrulandı (0 satır!), R2 düzeltildi | Tamamlandı (1M/10M yapılmadı) |
| 3 | 3 dedektör varyantı, yolov8n_visdrone kazandı | Tamamlandı (ablation'lar yapılmadı) |
| 4 | Qwen3-VL-Embedding-2B + MRL taraması, CPU-impratik bulgusu | Tamamlandı (GPU eksik) |
| 5 | Profiller + bu rapor | Bu belge |

## 4. En önemli 5 bulgu

1. **ClickHouse varsayılan strateji seçici filtrede güvensiz.**
   `vector_search_filter_strategy='auto'` (=postfiltering), 100K satırda
   seçici filtre (`bus_count>=1 AND person_count>=3`) altında **0 satır**
   döndürdü — LIMIT 10 istenmesine rağmen. `prefilter` ve `bruteforce`
   doğru 10 döndürdü. `vector_search_index_fetch_multiplier`'ı 50'ye
   çıkarmak düzeltiyor ama o noktada gecikme zaten prefilter'a yakınsıyor.
   **Öneri: seçici filtreli sorgularda `strategy='prefilter'`.**
2. **`max_limit_for_vector_search_queries` planın varsaydığından 10× büyük**
   (1000, 100 değil) — `top_k=200` zaten güvenli, araştırma öncesi
   varsayımın düzeltilmesi gerekti.
3. **Küçük fine-tune'lu dedektör her sınıfta kazanmıyor.** `yolov8n_visdrone`
   person/bus'ta COCO x-large'a eşdeğer/iyi ve ~2× hızlı, ama truck
   recall'da (0.50 vs 0.88) belirgin geride — yine de downstream
   Retrieval@10'a yansımadı (truck-özel sorgu payı düşük), bu yüzden
   varsayılan olarak seçildi.
4. **MRL boyut kırpması (2048d→256d) kalite kaybı olmadan ~7.4× depolama
   kazandırıyor** — tek gerçek embed koşumundan türetildi, modeli 4 kez
   koşturmaya gerek kalmadı. Üretim adayı: 256d/512d.
5. **CPU'da Qwen3-VL-Embedding-2B ingest için pratik değil** (~14.5
   dk/pencere, X-CLIP'in ~27 katı) — ilk smoke testi bunu ~17× hafife
   aldırmıştı (sentetik görüntüyle gerçek video karesi karıştırılmış).
   Bu CPU'ya özgü; GPU sonucu yok. Ayrıca Qwen-2048, X-CLIP'e karşı bu
   benchmarkta eşdeğer çıktı (MMEB-V2 liderliğine rağmen) — mevcut
   28-sorgu/73-pencere setinin ayırt etme gücü sınırlı olabilir.

## 5. Pareto tablosu (gerçek ölçülen eksenler)

| Profil | Dedektör | Embedding | Strateji | CPU embed hızı (video) | Depolama/1M satır | Notlar |
|---|---|---|---|---:|---:|---|
| fast | yolov8n_visdrone | xclip_hf_zeroshot (512d) | auto | ~32sn/pencere | ~1.19 GB | En hızlı uçtan uca |
| balanced | yolov8n_visdrone | siglip2_frameavg (1152d) | auto | ~62sn/pencere | ~2.67 GB | X-CLIP'e yakın kalite, ~2× yavaş |
| accurate | yolov8n_visdrone | siglip2_frameavg (1152d) | prefilter | ~62sn/pencere | ~2.67 GB | Seçici filtrede doğru sonuç garantisi |
| (referans, GPU gerekli) | yolov8n_visdrone | qwen3vl_emb_512 | prefilter | CPU'da ~14.5 dk (GPU'da doğrulanmadı) | ~1.06 GB | MRL 512d, 2048d kalitesiyle eşdeğer |

`query p95` sütunu kasıtlı olarak yok — bu oturumda ölçülen sorgu p50/p95
73 satırlık smoke veride anlamsız küçük (`bench/report.py`/`artifacts/
benchmark_report.json`'da mevcut ama N küçük olduğu için Pareto kararına
girmedi).

## 6. ClickHouse strateji önerisi

- Gevşek/filtresiz sorgu → `auto` (varsayılan, HNSW hızından tam
  faydalanır).
- Seçici filtre (ör. `bus_count>=1 AND person_count>=N`) → **`prefilter`**
  zorunlu; `auto`/`postfilter_rescore` yüksek `fetch_multiplier` (~50)
  olmadan sessizce eksik/boş sonuç döndürebilir.
- `search/query.py::search(..., strategy=...)` bu seçimi parametre olarak
  destekliyor; imza değişmedi.

## 7. Dedektör kararı

`yolov8n_visdrone` (`config.yaml: detector.default_variant`) — hız (~2×)
+ downstream Recall/Precision@10 (çoğu kategori eşdeğer/daha iyi) birlikte
gerekçelendiriyor. Truck recall zayıflığı bilinçli bir ödünleşim olarak
kaydedildi, gizlenmedi.

## 8. Model kararı

Birincil: **X-CLIP (hız) / SigLIP2 (kalite-hız dengesi)** — ikisi de CPU'da
pratik ve bu benchmarkta ayrışmıyor. Qwen3-VL-Embedding-2B (256d/512d MRL)
GPU'da doğrulanırsa "accurate" profiline terfi edebilecek bir aday olarak
kaydedildi ama bu oturumda CPU-pratik değil. VideoCLIP-XL (lisans) ve
LanguageBind_Video (mimari desteği yok) elendi.

## 9. Offline paket içeriği

`weights/weights_manifest.json`: X-CLIP (783.7 MB), SigLIP2 (4578.6 MB),
Qwen3-VL-Embedding-2B (4271.1 MB), yolo26x.pt (118.7 MB), yolov8n-visdrone
(6.2 MB), yolov8s-visdrone (22.5 MB) — hepsi SHA-256 ile. `weights/`
gitignore'da, git'e commit edilmedi.

## 10. Üretime açık bağımlılıklar (doğrulanmadı)

- **GPU ölçümü yok.** Tüm hız sayıları CPU-only. Colab/gerçek GPU
  ölçümü olmadan "accurate" profilin gerçek gecikmesi bilinmiyor.
- 1M/10M gerçek ölçek testi yapılmadı (yalnızca 100K, ekstrapolasyon var).
- `gt_walking` FiftyOne görsel denetimi hâlâ açık (ego-motion yanlış
  pozitif riski, önceki oturumdan miras).
- Gerçek telemetri şeması entegrasyonu (YOLO kolonları şu an vekil).
- Kurum altın sorgu seti — mevcut 28 sorgu sentetik/kurallı üretildi.
- Batch inference, imgsz/FP16/n_sample ablation'ları yapılmadı.
- CODEC(NONE), binary vector bind, materialize_skip_indexes_on_insert
  throughput denemeleri yapılmadı.

## 11. Sonraki karar

Devam: evet, mimari kararlar (prefilter, yolov8n_visdrone, X-CLIP/SigLIP2
ikilisi) gerçek veriyle doğrulandı. Üretime geçmeden önce zorunlu üç iş:
(1) GPU ölçümü (Colab bundle hazır, çalıştırılmayı bekliyor), (2) `gt_walking`
görsel denetimi, (3) 1M+ ölçek testi. Sahip: bu depoyu devralan sonraki
oturum/kişi; `NEXT_SESSION_HANDOFF.md` ve bu rapor başlangıç noktasıdır.

## 12. Faz 6 düzeltmeleri — decode bug'u ve iki iddianın yeniden ölçümü

> Bu bölüm §4-5'teki orijinal Faz 0-5 bulgularını SİLMİYOR, üzerine
> düzeltme ekliyor — geçmiş kayıt korunuyor, güncel doğru sayı burada.
> Tetikleyici: kullanıcı gerçek T4/L4 GPU koşumlarını ayrı bir sohbette
> analiz ettirdi; iki iddia çıktı, ikisini de bağımsız doğruladım (biri
> doğru, biri yanlış çıktı — detay `TASKS.md` Faz 6).

**Bulunan gerçek bug:** `ingest/03_embed.py`, `ingest/04_detect.py` ve
`scripts/colab_gpu_bench.py`, pencere başına n_sample kez AYRI
`cv2.CAP_PROP_POS_FRAMES` seek yapıyordu. H.264'te her seek decoder'ı en
yakın keyframe'e geri gönderip yeniden decode ettiriyor. `ingest/
frame_io.py` (yeni) bunu TEK seek + sequential okumaya çevirdi: gerçek
video üzerinde **12.6× hızlanma, kareler eski yöntemle bit-bit aynı**
(doğrulama: `tests/test_frame_io.py`, ayrıca 73 pencerelik gerçek
production verisiyle uçtan uca `max|diff|=0.0`).

**Düzeltilen iddia 1 — "küçük fine-tune'lu YOLO ~2× hızlı" (§4 bulgu 3):**
Decode düzeltmesinden sonra temiz 3'lü CPU karşılaştırması (aynı 73
pencere, bu makine): yolov8n_visdrone 1.12s/pencere, yolov8s_visdrone
1.45s/pencere, yolo26x_coco 4.25s/pencere. **Gerçek fark ~3.79×** —
orijinal "~2×" iddiasından KÜÇÜK değil, BÜYÜK. Yani decode gürültüsü
eski ölçümde farkı ABARTMAMIŞ, HAFİFLETMİŞ.

**Doğrulanan ama YANLIŞ çıkan iddia 2:** Dış analiz, "CPU'daki 2× sayısı
da muhtemelen decode-dominant" diye spekülasyon yürütmüştü. Ölçüldüğünde
bu YANLIŞ çıktı (yukarıdaki 3.79× gerçek, decode-bağımsız bir fark).
Buna karşılık L4 GPU'da ölçülen ~1.03× (üç YOLO varyantı arasında
neredeyse ayırt edilemez) bulgusu hâlâ geçerli VE decode-dominant —
GPU'da model compute süresi o kadar küçülüyor ki sabit seek maliyeti
payı büyüyor. CPU'da tam tersi: model compute zaten büyük, seek payı
görece küçük. **İki ortam farklı rejimde, ikisi de doğru, çelişmiyor.**
GPU tarafında temiz yeniden ölçüm hâlâ kullanıcının kendi Colab
oturumunu gerektiriyor (`scripts/colab_gpu_bench.py` artık dedektör
için de paylaşılan decode kullanıyor, hazır).

**Düzeltilen iddia 3 — "6 embedding modeli ölçüldü" (§8, Pareto tablosu):**
Gerçekte 3 model ölçüldü (X-CLIP, SigLIP2, Qwen3-VL-Embedding-2B);
qwen3vl_emb_1024/512/256 aynı transformer forward-pass'ini paylaşıyor
(MRL `truncate_dim` yalnız çıktı vektörünü kırpıyor, `models/
qwen3vl_emb.py`). `scripts/colab_gpu_bench.py` şema v2 artık bunu tek
`qwen3vl_emb` + `truncate_dims: [2048,1024,512,256]` satırıyla yazıyor.

**Artık ÖLÇÜLDÜ (önceden çıkarımdı):** Kullanıcı `scripts/
dtype_arch_probe.py`'yi gerçek T4'te çalıştırdı
(`artifacts/dtype_arch_probe_Tesla_T4.json`). Sonuç bf16/Turing
hipotezini net doğruluyor:

| Alan | Değer |
|---|---|
| compute_capability | (7, 5) — Turing |
| `bf16_native_by_capability` | false |
| model gerçek dtype | `torch.bfloat16` |
| `attn_implementation` | `sdpa` (varsayılan hipotezdeki gibi `eager` DEĞİL — düzeltme aşağıda) |
| native (bf16) tek-görüntü encode, medyan | 24.75s |
| `.half()` (fp16) tek-görüntü encode, medyan | 0.36s |
| **fp16 hızlanma** | **68.81×** |

Bu, mekanizmayı izole ediyor: `attn_implementation` iki ölçüm arasında
SABİT (`sdpa`, hiç değişmiyor) — yani asıl orijinal hipotezdeki
"FlashAttention-2 yokluğu eager attention'a düşürüyor" çerçevesi
YANLIŞ (model zaten `eager` değil `sdpa` kullanıyor, FA2 kurulu değil
ama bu SDPA'yı devre dışı bırakmıyor). Fark SADECE dtype'tan geliyor
(bf16→fp16) — Turing'in fp16 tensor core'ları var ama bf16 yok, bu
yüzden bf16 yavaş (muhtemelen tensor-core-dışı) bir yola düşüyor, fp16
aynı SDPA ile ~69× hızlanıyor. **Üretim önerisi:** Turing sınıfı GPU'da
(T4, RTX 20xx) Qwen3-VL-Embedding-2B kullanılacaksa modeli `.half()`
ile fp16'ya zorlamak, Ampere/Ada GPU'ya geçmeye neredeyse eşdeğer bir
kazanç veriyor.

**Düzeltilen iddia 4 — dtype_arch_probe.py'nin kendi bug'u:** Gelen
JSON'daki `torch_compile_timing` (0.366s) GÜVENİLMEZ ÇIKTI — script'in
o anki sürümünde `torch.compile()` testi `model.half()`'ten SONRA
çalışıyordu; `nn.Module.half()` YERİNDE mutasyon yapıp `self` döndürür
(doğrulandı: `m.half() is m` → `True`), yani compile testi aslında
"native bf16 + compile" değil "fp16 + compile" ölçüyordu (0.366s'nin
fp16_timing'e = 0.360s bu kadar yakın olması bunu zaten ele veriyordu).
Script'te sıralama düzeltildi (compile artık mutasyondan ÖNCE
çalışıyor) ve regresyon testi eklendi
(`tests/test_dtype_arch_probe_ordering.py` — eski sıralamaya karşı
çalıştırıldığında gerçekten FAIL verdiği doğrulandı). Gerçek "native
bf16 + compile" sayısı için probe'un düzeltilmiş sürümle yeniden
koşulması gerekiyor — bu henüz yapılmadı.

**Açık kalan (kullanıcı kararı gerektiriyor, bu oturumda başlatılmadı):**
ClickHouse üçlü strateji bake-off + `dataset_id`, değerlendirme
istatistiksel gücü (150+ sorgu), dataset adapter katmanı, batch
inference, dtype/mimari guard. Bunlar kaynak analiz belgesinin kendisinde
"insan karar vermeli, Claude Code karar veremez" diye işaretli (hedef
GPU, ana model, korpus kapsamı, caption dataset). Detay ve hazır
prompt'lar: `TASKS.md` Faz 6.
