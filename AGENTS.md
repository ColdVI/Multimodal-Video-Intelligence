# Coding agent talimatlari

## Gercekten calistirip dogruladim

- `tests/` altindaki 46 testin tamami (`scripts/poc.ps1 test` / `pytest`) -
  saf Python mantigi, GPU/ag/veri gerektirmez. Bu depoyu teslim etmeden
  once kostum. Ilk kosuda 2 test kirildi (`test_gt_object_finds_bus_frames`,
  `test_gt_walking_detects_displacement`); kok neden
  `eval/make_groundtruth.py::frames_to_intervals`'deki gercek bir
  off-by-one hatasiydi (kare-indeksi sureye cevirirken +1 eksikti).
  Duzeltip tekrar kostum. Transformers 5.x pooled-output regresyon testi de
  sonradan eklendi; guncel sonuc 46/46. Detay: CONTEXT.md ve STATUS.md.
- Tum `.py` dosyalari `python -m py_compile` ile sozdizimi kontrolunden
  gecti.
- X-CLIP API cagrilarini transformers'in kendi kaynak koduna karsi
  dogruladim (`processor(videos=..., ...)`, `model.get_video_features(...)`)
  ve gercek checkpoint ile CPU inference kostum: video/metin 512d, sonlu ve
  L2-normalize. Transformers 5.x'in Tensor yerine `BaseModelOutputWithPooling`
  dondurmesi gercek calistirmada yakalanip iki adapter icin duzeltildi.
- ClickHouse 26.7.1'de schema, 512d/1152d insert, filtre, cosineDistance ve
  gercek X-CLIP vektoruyle filtreli/filtresiz `search.query` smoke testi gecti.
- `yolo26x.pt` indirildi; CPU tek-goruntu inference ve sentetik MP4 uzerinde
  `window_features()` kolon uretimi calisti.
- Resmî VisDrone-MOT trainseti indirildi ve 56/56/24.201 veri sözleşmesi
  doğrulandı. Tek gerçek otobüslü sekans frames/windows/YOLO/X-CLIP/load/query/
  GT/eval hattından geçti; ayrıntılar `STATUS.md` dosyasında.
- SigLIP2 checkpoint'i `model_type: siglip` kullanır; resmî model kartındaki
  gibi `AutoModel` ile yüklenmelidir. `Siglip2Model`'e zorlamak gerçek
  çalıştırmada patch embedding şekil uyuşmazlığı verdi ve düzeltildi.
- Offline mod (`common.offline_mode_enabled()`, `HF_HUB_OFFLINE=1` veya
  `config.yaml: offline_mode: true`) her iki adaptörde `local_files_only=True`
  tetikler. `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` ile gerçek çalıştırma:
  X-CLIP yükleme 0.85 sn + embed_text 0.45 sn (512d), SigLIP2 yükleme 2.65 sn +
  embed_text 4.89 sn (1152d) — ikisi de zaten lokal `.runtime/huggingface`
  cache'inden yüklendi, ağ çağrısı yapılmadı.
- `scripts/package_weights.py` gerçek çalıştırıldı: X-CLIP (783.7 MB),
  SigLIP2 (4578.6 MB) ve `yolo26x.pt` (118.7 MB) `weights/weights_manifest.json`
  içine model ID/revision/SHA-256/boyut ile paketlendi (`weights/` gitignore'da).

## Dogrulamadim (gercek ortam gerekiyor)

- `models/siglip_avg.py` — gerçek SigLIP2 sonucu için `STATUS.md` geçerlidir.
- Tam 56-sekans YOLO/model load çalıştırılmadı; gerçek kanıt 5-sekans/7-pencere
  subsetidir. Her iki model de gerçek inference/load/eval'den geçti.
- `eval/make_groundtruth.py::gt_walking` gerçek annotation'da koştu fakat
  kamera-hareketi (ego-motion) yanlış pozitif riski için 5-10 sekanslık
  FiftyOne görsel denetimi yapılmadı.

## Veri indirme sözleşmesi

Resmî VisDrone GitHub deposundaki açık Google Drive Task 4 train bağlantısı
`scripts/download_visdrone.py` ile indirilebilir. Script sabit dosya boyutu,
SHA-256, ZIP yol güvenliği ve 56/56/24.201 sözleşmesini doğrular. Bağlantı
kota/giriş engeline düşerse kimlik bilgisi isteme; kullanıcıdan resmî dosyayı
elle `data/downloads/` altına koymasını iste.

## Konvansiyonlar

- Tum sayisal sabitler `config.yaml`'da, `common.load_config()` ile okunur.
  Kodun icine yeni bir hardcoded sabit ekleme.
- Yeni embedding modeli eklerken sirayla: `models/<isim>.py` yaz
  (`VideoTextEmbedder` arayuzunu uygula) -> `models/__init__.py`'deki
  `_REGISTRY`'ye ekle -> `schema.sql`'e o modelin `dim` degerine gore yeni
  bir `clips_<isim>` tablosu ekle -> `eval/run_eval.py::MODELS` listesine
  ekle.
- Sozlesmeler (fonksiyon imzalari) icin dogru kaynak dogrudan kod:
  `models/base.py::VideoTextEmbedder`, `search/parser.py::ParsedQuery`,
  `eval/metrics.py::evaluate()` donus sozlugu.
- `search/query.py::search()` imzasi: `search(q, model_name, top_k=200,
  use_filters=True)`. Bunu cagiran her yerde parametre adlarini
  degistirme — onceki surumde bu imza ile cagiran kod arasinda uyusmazlik
  vardi, bilerek tek noktadan sozlesme haline getirildi.

## Sirada ne var

`TASKS.md`'deki checklist'i sirayla isle. Her fazin sonunda Windows'ta
`powershell -ExecutionPolicy Bypass -File scripts/poc.ps1 test`, Linux/macOS'ta
`make test` calistir — pure-logic testler hicbir fazda bozulmamali.
