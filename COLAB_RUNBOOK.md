# Faz 6 Colab Runbook - MRL & Vector Backend Arastirmasi

Bu paket, `07_MRL_VE_VECTOR_BACKEND_ARASTIRMA_SPEC.md`'nin GPU/vector-backend
gerektiren kismini **Google Colab**'da calistirmak icindir. Yerel makinede
(GT 1030, CPU-only Torch) Qwen embedding uretimi veya backend benchmarki
**calistirilmaz** - bu paket onun yerine gecer.

## Sizin yapmanız gereken TEK 5 sey

1. `Multimodal-Video-Intelligence-phase6.zip` dosyasini Google Drive'iniza yukleyin.
2. Google Drive'da ZIP'e sag tiklayip **"Extract" / "Ayikla"** secin - tam
   olarak asagidaki klasore cikmali (Drive'in "Extract" ozelligi zip adiyla
   ayni isimde bir klasor olusturur; gerekirse klasoru surukleyip bu tam
   yola tasiyin):
   ```
   /content/drive/MyDrive/VidEmbedd/phase6_repo/
   ```
   (Bu, `src/research/colab_paths.py::REPO_EXTRACT_PATH` ile ayni sabit yol -
   degistirmeyin, her notebook'un ilk hucresi bunu arar.)
3. O klasordeki `notebooks/00_research_scope_and_dataset_audit.ipynb`
   dosyasina Drive'da cift tiklayip **"Open with Google Colaboratory"** secin.
4. **Runtime > Change runtime type** ile GPU (T4 yeterli) secin - sadece
   Faz 1 (asagida) icin. Faz 2 icin GPU GEREKMEZ (CPU/high-RAM yeterli,
   maliyet icin GPU'suz secim onerilir).
5. **Runtime > Run all** (veya hucreleri sirayla calistirin). Ilk hucre
   Drive izni ister - onaylayin.

**Bunlarin DISINDA hicbir sey yapmaniza gerek yok**: dosya yolu duzenleme,
pip install, database kurulumu, dataset elle tasima - hepsi kod ile
otomatik yapilir (asagida acikliyoruz).

---

## Neden iki asama?

| Asama | Runtime | Notebook'lar | Neden |
|---|---|---|---|
| **Faz 1 (GPU)** | GPU (T4) | 00, 01, 02 | Qwen3-VL-Embedding-2B GPU gerektirir (CPU'da ~58 gun surer, bkz. notebook 02) |
| **Faz 2 (CPU/high-RAM)** | CPU, yuksek RAM | 03, 04, 05, 06 | ClickHouse/Qdrant/pgvector kurulumu + benchmark GPU gerektirmez; GPU runtime'i bosuna tutmamak icin ayri oturum |

Iki asama **ayri Colab runtime'lari** olabilir (ör. bugun Faz 1, yarin
Faz 2) - butun buyuk/kalici veri (dataset, embedding, sonuc dosyalari)
Drive'da durdugu icin oturumlar arasi veri KAYBOLMAZ.

---

## Faz 1 - GPU asamasi (notebook 00 -> 01 -> 02)

Ayni GPU runtime'inda, sirayla:

1. **`00_research_scope_and_dataset_audit.ipynb`** - dataset masa basi
   denetimi (erisilebilirlik kontrolu, ~1 dakika).
2. **`01_auair_download_and_validation.ipynb`** - AU-AIR'i Drive'a indirir
   (gdown, resume destekli - kesilirse hucreyi tekrar calistirin, kaldigi
   yerden devam eder), sema/video/hard-stop/pencereleme/telemetri/secicilik
   dogrulamasini yapar.
   - **CapERA video:** Bu depoda YOK. Eger CapERA'da GERCEK embedding
     uretmek istiyorsaniz, ham CapERA video dosyalarini ONCEDEN
     `/content/drive/MyDrive/VidEmbedd/phase6_mrl_vector_backend/datasets/capera/videos/`
     klasorune siz yerlestirmelisiniz (bkz. `drive_manifest.json`) - bu,
     depodaki mevcut CapERA verisinin (caption JSON + agregatif sonuclar)
     KAPSAMI DISINDA, ayri bir veri kaynagidir.
   - `scripts/verify_drive_inputs.py`'yi calistirarak (bir hucrede
     `!python scripts/verify_drive_inputs.py`) hangi dataset'lerin hazir
     oldugunu kontrol edebilirsiniz.
3. **`02_qwen2b_embedding_and_mrl.ipynb`** - GPU kapisi GERCEKTEN kontrol
   eder (GPU yoksa acikca durur, sahte sonuc URETMEZ). GPU varsa:
   AU-AIR + CapERA (varsa) + MSR-VTT icin 2048d Qwen embedding uretir,
   **her 100 item'da Drive'a checkpoint yazar** (kesilirse hucreyi tekrar
   calistirmak KALDIGI YERDEN devam eder - checkpoint'ler
   `.../checkpoints/*.ndjson`), sonra 1024/512/256 boyutlarini turetir.

Faz 1 sonunda Drive'da: `checkpoints/*.ndjson`, `embeddings/*_qwen*.json`,
`02_qwen2b_embedding_and_mrl_manifest.json` (icinde `embedding_ready`
bayraklari dataset basina).

---

## Faz 2 - CPU/high-RAM asamasi (notebook 03 -> 04 -> 05 -> 06)

Runtime'i **GPU'suz** degistirin (Runtime > Change runtime type > None,
yuksek RAM secilebiliyorsa secin). Ayni sekilde Drive'da
`phase6_repo/notebooks/03_...ipynb`'yi acip calistirin, sirayla:

1. **`03_postgres_metadata_telemetry.ipynb`** - AU-AIR telemetri/metadata
   icin basit bir Postgres semasi kurar (yerel/ephemeral - Drive'a DEGIL),
   notebook 01'in ciktisini yukler, secicilik esiklerini canli sorguyla
   dogrular.
2. **`04_vector_backend_loading.ipynb`** - ilk hucre
   `scripts/colab_preflight.py`'yi calistirip
   `environment_capability_report.json` uretir. Sonra ClickHouse, Qdrant,
   pgvector'i **SIRAYLA** (asla ayni anda ikisi/ucu birden) kurar,
   baslatir, saglik kontrolu yapar, notebook 02'nin embedding'lerini yukler,
   durdurur, yerel veri dizinini temizler. Bir backend kurulamazsa
   `environment_unavailable` yazar - sessiz fallback YOK, sahte sonuc
   URETILMEZ.
3. **`05_hybrid_query_benchmark.ipynb`** - notebook 04'te "healthy" olan
   backend'ler + notebook 02'nin embedding'leri varsa GERCEK sorgu
   benchmarki kosar (latency, recall-vs-exact, top-k agreement). Ikisi de
   yoksa acikca "KANIT YOK" yazar, `vector_database_results.csv`'yi BOS
   birakir.
4. **`06_results_and_decision_report.ipynb`** - yukaridaki TUM
   notebook'larin GERCEK ciktilarini toplar (rakam URETMEZ), SS14'un 8
   sorusunu ve mentor ozetini yazar.

Faz 2 sonunda Drive'da: `ingest_report.csv`, `vector_database_results.csv`,
`vector_database_report.md`, `mentor_summary.md`, `decision_report_answers.json`.

---

## Sonuclari nerede bulacaksiniz

Hepsi `/content/drive/MyDrive/VidEmbedd/phase6_mrl_vector_backend/` altinda
(bkz. `drive_manifest.json` tam liste icin). Her kosumdan sonra (opsiyonel,
ekstra guvence icin) bir hucrede:
```
!python scripts/copy_results_to_drive.py
```
calistirabilirsiniz - bu, oturuma OZGU (Drive'a otomatik yazilmayan) iki
seyi de Drive'a kopyalar: `environment_capability_report.json` ve
calistirilmis `.ipynb` dosyalarinin kendisi (Colab oturumu kapaninca
`/content` kaybolur, Drive kaybolmaz).

## Sorun giderme

- **"REPO_ROOT yok" hatasi (bootstrap hucresi):** ZIP'i tam olarak
  `/content/drive/MyDrive/VidEmbedd/phase6_repo/` klasorune cikardiginizdan
  emin olun (adim 2).
- **Backend kurulamiyor (`environment_unavailable`):**
  `environment_capability_report.json`'a bakin - apt-get/docker/port/RAM
  hangisi eksik gorun bilir. `backend_versions.json`'daki surumler artik
  indirilemiyor olabilir (paket yayinlarindan kaldirilmis olabilir) -
  o zaman surumu guncelleyip ilgili `scripts/install_*_colab.sh`'daki
  URL'yi de guncelleyin.
- **AU-AIR indirme cok yavas/kesiliyor:** gdown Drive'in kendi hiz
  sinirlamasina tabi - hucreyi tekrar calistirmak resume ile devam eder.
- **CapERA/MSR-VTT embedding_ready=False:** ham video dosyalari Drive'da
  yok demektir - `dataset_download_manifest.json`'daki `source` alanina
  bakip elle yerlestirin (spec'in kendi kurali: bu depoda CapERA ham
  videosu hic olmadi, MSR-VTT videolarini ayrica Drive'a tasimaniz
  gerekebilir).

## Onemli: bu paketin NE OLMADIGI

- Production sistem degil - notebook tabanli, tek seferlik arastirma.
- `docker-compose.yml` (ana POC) DEGISTIRILMEDI - Faz 2'nin backend'leri
  TAMAMEN ayri, gecici altyapidir.
- `backend_versions.json`'daki surumler KASITLI sabit (latest degil) -
  degistirmeyin, degismesi gerekiyorsa BILEREK guncelleyin.
