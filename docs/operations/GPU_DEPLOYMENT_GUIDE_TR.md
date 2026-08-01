# Kurumsal GPU Ortamı Kurulum ve İşletim Kılavuzu

## 1. Amaç ve kapsam

Bu doküman, Multimodal Video Intelligence uygulamasının NVIDIA GPU bulunan yeni bir bilgisayara kurulmasını, kurum tarafından sağlanan videoların gerçek Qwen modeliyle işlenmesini, metadata bilgilerinin PostgreSQL'e ve vektörlerin ClickHouse'a aktarılmasını ve daha önce tanımlanmamış metin sorgularının istek anında embed edilerek aranmasını açıklar.

Üretim akışında video embedding'leri ingest sırasında bir kez oluşturulur. Arama sırasında videolar yeniden işlenmez. Her yeni sorgu metni `EMBEDDING_MODE=real` altında Qwen modeliyle istek anında embed edilir; production sorgu cache'i kullanılmaz.

Önerilen üretim profili aşağıdaki gibidir:

- Metadata deposu: PostgreSQL
- Vektör deposu: ClickHouse
- Embedding boyutu: 512
- Filtre yürütme biçimi: native pushdown
- Varsayılan arama stratejisi: prefilter
- Embedding modu: real

## 2. Ön koşullar

Hedef bilgisayarda aşağıdaki bileşenler bulunmalıdır:

1. Uyumlu NVIDIA ekran kartı ve güncel NVIDIA sürücüsü
2. Docker Engine veya Docker Desktop
3. NVIDIA Container Toolkit ve Docker GPU erişimi
4. Git
5. Python 3.11 veya uyumlu bir Python ortamı
6. Kurum verisine ve Qwen model bundle'ına yeterli disk alanı

GPU erişimi aşağıdaki komutla doğrulanmalıdır:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi
```

Bu kontrol başarılı olmadan gerçek video ingest işlemi başlatılmamalıdır.

## 3. Kaynak kodun alınması

```bash
git clone https://github.com/ColdVI/Multimodal-Video-Intelligence.git
cd Multimodal-Video-Intelligence
```

Container imajları için kök dizindeki tek kanonik `Dockerfile`, Python bağımlılıkları için kök dizindeki `requirements.txt` kullanılmaktadır. Compose profilleri aynı Dockerfile'ın `api`, `ui`, `hybrid` ve `gpu` build target'larını kullanır.

## 4. Dizin yapısı

Önerilen kurum veri dizini:

```text
DATA_ROOT/
├── videos/
│   ├── flight_001.mp4
│   └── flight_002.mp4
├── telemetry/
│   ├── flight_001.csv
│   └── flight_002.csv
└── pairing/
    └── institution.csv
```

Önerilen model bundle dizini:

```text
MODEL_BUNDLE_ROOT/
├── source/
├── model/
└── manifest.json
```

Repository içinde aşağıdaki dosya hazırlanır:

```text
datasets/kurum.yaml
```

Docker, `DATA_ROOT` dizinini container içinde `/workspace/data`, `datasets/` dizinini ise `/workspace/datasets` olarak salt okunur bağlar.

## 5. Ortam değişkenleri

Örnek dosya kopyalanır:

```bash
cp .env.example .env
```

`.env` dosyası hedef bilgisayardaki gerçek yollar ve güvenli parolalarla düzenlenmelidir:

```dotenv
POSTGRES_DB=uav_search
POSTGRES_USER=mvi
POSTGRES_PASSWORD=<GUCLU_POSTGRES_PAROLASI>

CLICKHOUSE_DB=uav_search
CLICKHOUSE_USER=mvi
CLICKHOUSE_PASSWORD=<GUCLU_CLICKHOUSE_PAROLASI>

DATA_ROOT=/absolute/path/to/institution-data
MODEL_BUNDLE_ROOT=/absolute/path/to/mvi-model-bundle
ARTIFACTS_ROOT=./artifacts

EMBEDDING_MODE=real
ENABLED_VECTOR_BACKENDS=clickhouse
DEFAULT_VECTOR_BACKEND=clickhouse
ENABLED_DIMENSIONS=512
FILTER_EXECUTION_MODE=pushdown

CUDA_IMAGE_TAG=12.1.1-runtime-ubuntu22.04
BIND_HOST=127.0.0.1
API_TOKEN=
```

Windows yolları Docker tarafından erişilebilir biçimde, örneğin `D:/institution-data` ve `D:/mvi-model-bundle` olarak yazılmalıdır.

Servis farklı bilgisayarlardan erişilecek şekilde yayınlanacaksa `BIND_HOST` loopback dışına alınmalı ve güçlü bir `API_TOKEN` tanımlanmalıdır. `.env` dosyası gizli bilgi içerdiğinden Git'e eklenmemelidir.

## 6. Dataset manifestinin hazırlanması

Örnek kurum manifesti kopyalanır:

```bash
cp datasets/example_institution.yaml datasets/kurum.yaml
```

`datasets/kurum.yaml` içinde en az aşağıdaki alanlar kurum verisine göre düzenlenmelidir:

- `dataset_id`
- video dosyalarını bulan `source.videos_glob`
- video/telemetry eşleştirme yöntemi
- pencere süresi, stride ve frame sayısı
- telemetry zaman kolonu
- irtifa, hız, yön ve diğer metadata kolon eşlemeleri
- zaman referansı ve birimler

Kurum CSV kolonlarının fiziksel olarak yeniden adlandırılması gerekmez. Manifest, kaynak kolonlarını canonical PostgreSQL alanlarına veya `extra` JSON alanına eşler. Telemetry bulunmayan veri setlerinde video ve segment metadata'sı yine PostgreSQL'e yazılır; yalnız telemetry alanları boş kalır.

## 7. Model bundle'ın hazırlanması

Aynı model kimliği, revision ve kaynak commit'i hem video hem metin embedding'lerinde kullanılmalıdır:

```bash
python scripts/prepare_model_bundle.py \
  --model-id Qwen/Qwen3-VL-Embedding-2B \
  --model-revision 9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda \
  --source-repo https://github.com/QwenLM/Qwen3-VL-Embedding.git \
  --source-commit 393e2978d27852b0d0230d6994f37f9c15bed73c \
  --bundle-root /absolute/path/to/mvi-model-bundle
```

Model bundle build sırasında container imajına indirilmez. Bundle, GPU compose profili tarafından salt okunur bağlanır. Manifest ve hash doğrulaması geçmeden ingest başlatılmamalıdır.

## 8. Read-only preflight

Aşağıdaki komut host, secret, dizin, dataset ve model bundle kontrollerini gerçekleştirir:

```bash
python scripts/preflight.py \
  --dataset datasets/kurum.yaml \
  --env-file .env \
  --json-out artifacts/faz11/preflight.json
```

`status=pass` ve exit code `0` görülmeden servis başlatma veya ingest kabul edilmiş sayılmaz.

## 9. Servislerin GPU profiliyle başlatılması

```bash
docker compose \
  --env-file .env \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  up -d --build
```

Durum kontrolü:

```bash
docker compose \
  --env-file .env \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  ps

curl -fsS "http://127.0.0.1:8000/health?dataset_id=<DATASET_ID>"
```

Erişim adresleri:

- UI: `http://127.0.0.1:7860`
- API/OpenAPI: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

## 10. Küçük GPU smoke işlemi

20 GB kurum verisinin tamamından önce 1-2 gerçek MP4 veya sınırlı sayıda pencereyle GPU smoke yapılmalıdır:

```bash
python scripts/gpu_smoke.py \
  --dataset datasets/kurum.yaml \
  --data-root /absolute/path/to/institution-data \
  --output artifacts/faz11/gpu_smoke.json \
  --windows 10
```

Smoke çıktısında sentetik fallback bulunmamalı; model, GPU, dtype, embedding shape ve sonlu değer kontrolleri başarılı olmalıdır.

## 11. Gerçek ingest

Preflight ve GPU smoke başarılı olduktan sonra ingest başlatılır:

```bash
docker compose \
  --env-file .env \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  exec -T api python -m app.ingestion.ingest \
  --dataset /workspace/datasets/kurum.yaml \
  --resume
```

İşlem sırası aşağıdaki gibidir:

```text
MP4 dosyaları
→ video chunk ve retrieval window üretimi
→ frame örnekleme
→ Qwen video embedding, GPU
→ PostgreSQL video/segment/telemetry metadata yazımı
→ ClickHouse 512d vektör yazımı
→ metadata ve vektör satır sayısı doğrulaması
→ yeni run'ın aktif edilmesi
```

`--resume`, daha önce tamamlanmış chunk'ları yeniden decode veya embed etmez. Başarısız ya da tamamlanmamış yeni run doğrulanmadan mevcut active run değiştirilmez.

Run raporları aşağıdaki dizine yazılır:

```text
artifacts/faz11/ingest_runs/<run_id>/report.json
artifacts/faz11/ingest_runs/<run_id>/errors.jsonl
```

## 12. PostgreSQL metadata entegrasyonu

PostgreSQL entegrasyonu generic ingest hattının zorunlu parçasıdır. Başlıca tablolar şunlardır:

- `ingest_runs`: model, revision, manifest ve run durumu
- `ingest_chunks`: resume ve chunk commit durumu
- `run_videos`: video kimliği ve kaynak URI
- `run_segments`: `segment_id`, `video_id`, `t_start`, `t_end`
- `run_segment_metadata`: kişi, araç ve diğer türetilmiş metadata
- `run_segment_telemetry`: irtifa, hız, konum, yön ve `extra` alanları
- `dataset_active_runs`: sorgularda kullanılacak doğrulanmış aktif run

Metadata önce run kapsamına yazılır. ClickHouse vektör sayıları ile PostgreSQL segment sayıları doğrulandıktan sonra active run işaretçisi değiştirilir. Arama sonucundaki `video_id`, `segment_id`, `t_start`, `t_end`, dosya yolu ve telemetry alanları PostgreSQL'den hydrate edilir.

## 13. Canlı metin sorgusu

`EMBEDDING_MODE=real` altında her yeni metin sorgusu istek anında GPU'da embed edilir. Video embedding'leri yeniden üretilmez ve production query cache'i kullanılmaz.

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "otopark girişinde yürüyen iki kişi",
    "dataset_id": "<DATASET_ID>",
    "backend": "clickhouse",
    "strategy": "prefilter",
    "dimension": 512,
    "top_k": 10,
    "repeats": 1
  }'
```

Yanıtta en az `video_id`, `segment_id`, `t_start`, `t_end`, `score`, model revision ve aşama süreleri kontrol edilmelidir.

## 14. Aktif arama stratejileri

Desteklenen stratejiler:

| Backend | Stratejiler | Varsayılan |
|---|---|---|
| ClickHouse | `exact`, `ann`, `prefilter`, `postfilter` | `prefilter` |
| Qdrant | `exact`, `ann`, `ann_high_ef`, `prefilter` | `ann` |
| pgvector | `exact`, `ann`, `iterative_scan`, `iterative_strict` | `iterative_scan` |
| Milvus | `ann` | Profil bağımlı |
| NumPy referans | `exact` | `exact` |

Kurum üretim profili için ClickHouse `prefilter`, 512 boyut ve native filter pushdown önerilir. Qdrant ve pgvector profilleri karşılaştırma, yedekleme veya ayrı kabul çalışmaları için kullanılabilir.

Adaptive MRL isteğe bağlıdır. Exact rerank yalnız desteklendiği backend üzerinde etkinleştirilmelidir; desteklenmeyen backend/strateji kombinasyonları API tarafından açık hata ile reddedilir.

## 15. Kabul kriterleri

Sistem ancak aşağıdaki kanıtlar mevcutsa hedef ortamda kabul edilmiş sayılmalıdır:

1. Docker GPU erişimi ve `nvidia-smi` başarılıdır.
2. Preflight sonucu `pass` durumundadır.
3. GPU smoke gerçek MP4 üzerinde tamamlanmıştır.
4. Ingest raporu başarılıdır ve sentetik fallback kullanılmamıştır.
5. PostgreSQL metadata ve ClickHouse vektör sayıları tutarlıdır.
6. Yeni run aktif edilmiştir.
7. Daha önce kullanılmamış bir metin sorgusu canlı GPU embedding ile HTTP 200 döndürmüştür.
8. Sonuçta video kimliği ve geçerli zaman aralığı bulunmaktadır.
9. Model ID, revision, source commit, embedding boyutu ve normalizasyon video ve sorgu tarafında aynıdır.

Yerel CPU veya cached/hybrid sonuçları, kurum GPU production kabulünün yerine geçmez.
