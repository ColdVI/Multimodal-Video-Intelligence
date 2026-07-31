# Colab portable embedding runbook

`notebooks/08_colab_portable_runner.ipynb` is a **portable Qwen embedding
production path**, not an alternate deployment of the institution system.
Read this before running it.

## Colab'ın rolü ve olmayan rolü

Colab burada yalnız üç şey içindir:

```text
GPU embedding production
small-scale evaluation
artifact export
```

Kalıcı sistemin rolü ayrıdır ve Colab bunu **değiştirmez**:

```text
NVIDIA Linux host
Docker Compose
PostgreSQL
ClickHouse
API
UI
persistent DATA_ROOT
persistent model bundle
```

Somut olarak:

- Colab'da tam kalıcı PostgreSQL + ClickHouse + API + UI deployment'ı
  **desteklenmez**. Docker daemon'ı Colab runtime'ında güvenilir bir
  production yolu **değildir** — bu notebook Docker kullanmaz.
- Colab runtime'ı kapanınca (timeout, disconnect, yeniden başlatma) yerel
  (`/content/...`) her şey kaybolur. Yalnız **Drive'a yazılan** dosyalar
  hayatta kalır — bu yüzden notebook tüm gerçek çıktıyı doğrudan
  `DRIVE_ROOT/artifacts/colab_embeddings/` altına yazar, `/content` altına
  değil.
- Büyük videoları doğrudan mount edilmiş Drive'dan decode etmek yavaş
  olabilir; notebook bu yüzden her videoyu işlemeden önce geçici olarak
  yerel Colab SSD'sine (`/content/stage`) kopyalar, işleyip hemen siler —
  kalıcı state hiçbir zaman yerelde tutulmaz.
- Bu notebook'un ürettiği embedding'ler kurum sisteminin **kaynağı
  değildir** — yalnız bir girdi hazırlama adımıdır. Aramayı gerçekten
  çalıştıran sistem her zaman kalıcı NVIDIA Linux host'taki Docker Compose
  deployment'ıdır (bkz. [USER_GUIDE.md](USER_GUIDE.md)).

## Ne zaman kullanılır

- Kurum henüz kendi kalıcı GPU host'unu hazırlamadıysa, küçük/orta ölçekli
  bir video koleksiyonu için embedding üretip kurum stack'ine aktarmak.
- Model/pipeline değişikliklerini kalıcı altyapı kurmadan hızlı denemek.
- Küçük ölçekli recall/latency sağlık kontrolü (notebook §7).

## Çalıştırma sırası

1. Notebook'u Colab'a yükleyin, **Runtime → Change runtime type → GPU**
   seçin.
2. §1 (environment checks) hücresini çalıştırın — GPU yoksa notebook açık
   bir mesajla durur, sessizce CPU'ya düşmez.
3. §2'de Drive mount edilir; `DRIVE_ROOT/data/videos/**/*.mp4` altına
   institution videolarınızı (veya bunlara sembolik/gerçek erişimi) önceden
   koyun.
4. §3-4, pinlenmiş Qwen kaynak commit'i (`393e2978d27852b0d0230d6994f37f9c15bed73c`)
   ve model revision'ını (`9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`) — bu
   repodaki `docs/MODEL_BUNDLE.md`/`.env.example` ile aynı pinler — indirir
   ve shard planını çıkarır.
5. §5, her shard'ı işler; kesintiden sonra notebook'u tekrar çalıştırmak
   yalnız tamamlanmamış shard'ları işler (`shard_*.json`'daki
   `status=completed` kontrolüyle) — tamamlanmış shard'lar tekrar
   işlenmez.
6. §6, her shard'ın hash'ini, tekrarsız ID'lerini, sonlu (non-NaN/Inf)
   olduğunu, doğru boyutta olduğunu ve L2-normalize olduğunu doğrular;
   sonunda tek bir `embedding_manifest.json` (kendi hash'i dahil) üretir.
7. §7 opsiyoneldir — küçük ölçekli bir öz-değerlendirme.
8. §8, `DRIVE_ROOT/artifacts/colab_embeddings_export.zip`'i üretir.

## Kurum sistemine aktarım

```bash
# Zip'i Drive'dan indirip kurum host'una kopyalayın, sonra:
unzip colab_embeddings_export.zip -d /kurum/staging/colab_embeddings
# embedding_manifest.json'daki model_id/model_revision/source_commit'in
# .env'deki QWEN_MODEL_REVISION/QWEN_SOURCE_COMMIT ile eşleştiğini doğrulayın.
```

Bu notebook **gerçek ingest'in yerine geçmez** — kurum stack'i kendi
`python -m app.ingestion.ingest --dataset ...` akışıyla, kendi pinlenmiş
model bundle'ıyla (`scripts/prepare_model_bundle.py`) çalışmaya devam eder.
Colab çıktısı, ileride bu akışa opsiyonel bir "önceden hesaplanmış
embedding" girdisi olarak bağlanabilir; bu entegrasyon FAZ 11 kapsamının
dışındadır ve bu oturumda kod olarak eklenmedi — yalnız notebook'un export
ettiği format/hash sözleşmesi burada belgelidir.

## Bu oturumda doğrulanan ve doğrulanmayan

Bu notebook'un **yapısal geçerliliği** (gerçek `nbformat` ile parse/validate)
bu oturumda doğrulandı: `notebooks/08_colab_portable_runner.ipynb` geçerli
nbformat 4.5 JSON'dur (bkz. `tests/test_faz11_docs_and_notebook.py`).
Gerçek Colab GPU runtime'ında **çalıştırılmadı** — bu ortamda GPU/Colab
erişimi yok. Aşağıdaki iddialar bu yüzden `not_run` kabul edilmelidir, gerçek
Colab oturumunda doğrulanana kadar:

- Gerçek model indirme ve gerçek embedding üretimi.
- Gerçek shard resume davranışı (kesinti simülasyonu).
- Gerçek küçük-ölçek recall sonuçları.

Notebook hücrelerinin mantığı (hash zinciri, resume kontrolü, finite/norm/
boyut doğrulaması, atomic write deseni) bu depodaki eşdeğer production
kodla (`service/app/embedding/bundle.py`, `service/app/ingestion/ingest.py`)
aynı desenleri kullanacak şekilde yazıldı, ancak notebook kodu ayrı bir
dosyadır ve production tarafından import edilmez — iki taraf da bağımsız
olarak bakım gerektirir.
