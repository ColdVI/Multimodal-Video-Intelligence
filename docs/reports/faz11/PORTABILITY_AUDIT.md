# Faz 11 taşınabilirlik denetimi

Bu tablo başlangıç commit'i `23eb2a894c9b24f998b05a93c6a33262a860796d`
üzerinde, 30 Temmuz 2026 tarihinde kaynak kod ve gerçek komut çıktılarıyla
hazırlandı. Başlangıç çalışma ağacı temizdi. Ayrıntılı makine-okunur kanıt
`artifacts/faz11/baseline.json` dosyasındadır.

| Alan | Başlangıç durumu | Kaynak kanıt | Faz 11 gereksinimi |
|---|---|---|---|
| CapERA bağımlılığı | Taşınabilir değil | `service/app/config.py` import sırasında `config.yaml:datasets.capera` okuyor; iki service testi yerel CapERA JSON'u yokken fail oldu. | CapERA config ve verisi yalnız ilgili yol çağrıldığında çözülecek. |
| Backend profili | Taşınabilir değil | API lifespan, health, stats ve ingest PostgreSQL, ClickHouse ve Qdrant'ı koşulsuz çağırıyor. | Kurum varsayılanı yalnız ClickHouse vector backend; benchmark profili üç backend'i koruyacak. |
| Dimension profili | Taşınabilir değil | Global `DIMENSIONS=(2048,1024,512,256)` schema, ingest, stats ve UI'ya sabit. | Etkin boyutlar env/registry üzerinden yönetilecek; kurum varsayılanı 512. |
| Gerçek video embedding ingest | Çalışmıyor | `_build_vectors()` `embed_item(..., media=...)` geçmiyor; real mod açık hata verir. | Streaming pencere frame'leri Qwen batch API'sine taşınacak. |
| Uzun video belleği | Uygun değil | `DatasetBundle` ve `_build_vectors()` tüm corpus ile dört boyutu listelerde materyalize ediyor; backend sınırından önce `.tolist()` çağrılıyor. | Chunk/resume, sınırlı prefetch, ayrı embed ve DB batch ayarları. |
| Dataset yolu | Kuruma taşınabilir değil | Legacy loader'lar repo/artefact düzenine ve belirli dataset kimliklerine bağlı. | Hosttan bağımsız YAML manifest, `DATA_ROOT` altında relative glob ve tek adapter sınırı. |
| Telemetri hizalama | Eksik | Genel absolute/relative clock ve circular interpolation sözleşmesi yok. | Açık clock/anchor/offset formülleri ve canonical alan doğrulaması. |
| Run-versioning | Eksik | PostgreSQL/ClickHouse/Qdrant anahtarları dataset veya segment bazlı; active/staging run ayrımı yok. | Run-scoped storage, chunk ledger, doğrulanmış activation ve GC. |
| Filter execution | Ölçeklenmez | `search/engine.py` tüm candidate ID'leri PostgreSQL'den Python listesi ve set'ine alıyor. | Varsayılan backend-native pushdown; limitli legacy benchmark yolu. |
| ClickHouse filter payload | Eksik | Yalnız üç telemetri alanı ve sınırlı count payload'ı var; nullable değerler NaN'e çevriliyor. | Tüm P0 canonical alanların nullable run-scoped projeksiyonu. |
| UI görünürlüğü | Statik | Backend/dimension ve üç telemetri slider'ı sabit; aktif run/model/filter mode görünmüyor. | `/strategies` ve filter schema kontrollü görünürlük, run/provenance diagnostics. |
| Medya | Placeholder | UI player component'i mevcut fakat gerçek güvenli clip endpoint'i yok. | `DATA_ROOT` containment, süre limiti ve atomik ffmpeg cache ile media endpoint. |
| Model reproducibility | Uygun değil | GPU Dockerfile build sırasında branch'ten clone ve revision'sız download yapıyor; ağırlık image'a gömülüyor. | Exact source commit/model revision manifesti ve read-only bundle mount. |
| Compose güvenliği | Uygun değil | DB portları hosta açık; varsayılan gerçek parola benzeri değerler var; API/UI tüm arayüzlere publish ediliyor. | Internal DB ports, explicit secret validation, configurable loopback bind ve optional API token. |
| Offline kapsamı | Kısmi | Yerel model cache yolları var; tam deployment image/wheel/runtime bağımlılıkları paketlenmiyor. | Model/source bundle taşınabilirliği ayrı, tam air-gap önkoşulları dürüstçe belgeli. |
| Preflight | Eksik | Kurum manifesti, GPU/container, model hash, disk ve clock hizalamasını tek kapıda denetleyen yol yok. | Yazma yapmayan host ve container preflight, belgeli exit codes ve JSON artifact. |
| Migration | Eksik | Persisted Faz 7 schema run-scoped değil; plan/dry-run migration katmanı yok. | Veri silmeden deterministic legacy run planı ve satır-count raporu. |

## Baseline sonuçları

- `pytest tests/ -v`: collection error; `clickhouse_connect` mevcut pytest
  ortamında kurulu değil.
- `PYTHONPATH=service pytest service/tests/ -v -p no:cacheprovider`:
  41 passed, 2 failed, 15 skipped. İki failure yalnız yerel CapERA verisinin
  eager okunmasından kaynaklandı.
- 192 Python dosyasının syntax kontrolü geçti.
- `docker compose config` ve Faz 7 compose config parse kontrolleri geçti.

Bu sonuçlar baseline'dır; başarısız testler sonraki aşamalarda düzeltilmeden
nihai kabul PASS sayılmayacaktır.
