# Hibrit Video Arama Sistemi — POC İmplementasyon Planı

**Amaç:** Rapordaki hibrit mimarinin (filtre kolonları + klip embedding + LLM ayrıştırıcı) çekirdeğini, gerçek İHA verisi olmadan, açık drone veri setleriyle uçtan uca doğrulamak. Çıktı: "otobüsü göster" → `video7 00:00:12–00:00:41 (skor 0.91)` formatında sonuç veren, ölçülebilir bir sistem.

**Temel tasarım kararı:** POC verisinde telemetri yok. Üretim sistemindeki telemetri kolonlarının (hız, güneş açısı, deniz/kara) yerini POC'ta **detektörden türetilen kolonlar** alır (person_count, bus_count, is_night, camera_motion). Bu bilinçli bir eşleme: sorgu mimarisi (LLM ayrıştırma → SQL filtre → vektör artık → aralık birleştirme) birebir aynı kalır, yalnızca filtre kolonlarının *kaynağı* değişir. Gerçek veriye geçişte tek değişen Aktivite 2'dir.

**Kritik avantaj:** VisDrone-MOT anotasyonları (kare bazlı bounding box + track ID + 10 kategori) ground truth'un **otomatik** üretilmesini sağlar. "Otobüs görünüyor mu" → anotasyondan; "yaya yürüyor mu" → track yer değiştirmesinden; "otobüs VE yürüyen yaya" → ikisinin zaman kesişiminden. Manuel etiketleme faz 1'de sıfır.

---

## Faz 0 — Ortam ve Veri (2-3 gün)

### 0.1 Repo yapısı

```
video-search-poc/
├── docker-compose.yml
├── config.yaml
├── data/
│   ├── raw/                # VisDrone videoları (kare dizisi → mp4)
│   ├── annotations/        # VisDrone MOT anotasyonları
│   └── groundtruth/        # otomatik üretilen sorgu→aralık eşleşmeleri
├── ingest/
│   ├── 01_frames_to_video.py
│   ├── 02_windowing.py
│   ├── 03_embed.py         # model-agnostik embedding (adapter deseni)
│   ├── 04_detect.py        # YOLO → filtre kolonları
│   └── 05_load_clickhouse.py
├── search/
│   ├── parser.py           # sorgu → {filters, semantic_residual}
│   ├── query.py            # ClickHouse hibrit sorgu
│   └── merge.py            # pencere → aralık birleştirme
├── eval/
│   ├── make_groundtruth.py # anotasyon → sorgu bazlı GT aralıkları
│   ├── metrics.py          # tIoU, P@K, R@K, kategori kırılımı
│   └── run_eval.py
├── models/
│   ├── base.py             # VideoTextEmbedder arayüzü
│   ├── xclip_hf.py         # microsoft/xclip (hızlı başlangıç)
│   ├── siglip_avg.py       # kare-ortalama baseline (= mevcut sistem proxy'si)
│   └── languagebind.py     # 2. aday
└── notebooks/
    └── inspect_fiftyone.py
```

### 0.2 Altyapı (docker-compose)

```yaml
# docker-compose.yml
services:
  clickhouse:
    image: clickhouse/clickhouse-server:latest
    ports: ["8123:8123", "9000:9000"]
    volumes:
      - ch_data:/var/lib/clickhouse
    ulimits: { nofile: { soft: 262144, hard: 262144 } }
    environment:
      CLICKHOUSE_SKIP_USER_SETUP: 1

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    ports: ["9002:9000", "9001:9001"]
    environment:
      MINIO_ROOT_USER: poc
      MINIO_ROOT_PASSWORD: pocsecret
    volumes:
      - minio_data:/data

  postgres:
    image: postgres:16
    ports: ["5432:5432"]
    environment:
      POSTGRES_PASSWORD: poc
    volumes:
      - pg_data:/var/lib/postgresql/data

volumes: { ch_data: {}, minio_data: {}, pg_data: {} }
```

POC'ta Kafka/Temporal **yok** — kasıtlı. Orkestrasyon dayanıklılığı üretim problemi; POC'un sorusu "retrieval çalışıyor mu". Basit bir `make ingest` yeterli. (Üretim geçişinde raporun Temporal önerisi geçerli kalır.)

### 0.3 Veri seti

**Birincil: VisDrone-MOT** (train: 56, val: 7, test-dev: 7-17 sekans; aiskyeye.com'dan kayıtla indirilir). Kare dizileri + `sequences/*.txt` anotasyonları:

```
<frame_id>,<track_id>,<bbox_left>,<bbox_top>,<bbox_w>,<bbox_h>,<score>,<category>,<trunc>,<occl>
# category: 1=pedestrian, 2=people, 4=car, 5=van, 6=truck, 9=bus, 10=motor ...
```

**İkincil (faz 2'de): Okutama-Action** — yürüme/koşma/oturma gibi eylem etiketleri; "yürüyen adam" GT'sini track-yer-değiştirme sezgiselinden bağımsız ikinci bir kaynakla doğrulamak için.

**Ek çeşitlilik (opsiyonel):** Pexels/Pixabay'den 20-30 CC0 drone videosu — anotasyonsuz, yalnızca "sistem hiç görmediği içerikte saçmalıyor mu" duman testi için.

```python
# ingest/01_frames_to_video.py — VisDrone kare dizilerini mp4'e çevir
import subprocess, pathlib, json

RAW = pathlib.Path("data/raw/VisDrone2019-MOT-train/sequences")
OUT = pathlib.Path("data/raw/videos"); OUT.mkdir(parents=True, exist_ok=True)
FPS = 25  # VisDrone tipik; sekans başına doğrulanabilir

manifest = {}
for seq in sorted(RAW.iterdir()):
    out = OUT / f"{seq.name}.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", str(seq / "%07d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)
    ], check=True)
    n_frames = len(list(seq.glob("*.jpg")))
    manifest[seq.name] = {"fps": FPS, "n_frames": n_frames,
                          "duration_s": n_frames / FPS}
json.dump(manifest, open("data/raw/manifest.json", "w"), indent=2)
```

> **Not:** Kare→mp4 dönüşümünde FPS varsayımı GT zaman hesabını doğrudan etkiler. Manifest'e yazıp her yerde tek kaynaktan okuyun; sekans bazında farklıysa düzeltin.

---

## Faz 1 — Ingest Hattı (1 hafta)

### 1.1 Pencereleme (8s / 4s kaydırma — raporla aynı)

```python
# ingest/02_windowing.py
import json, pathlib

WIN, STRIDE = 8.0, 4.0
manifest = json.load(open("data/raw/manifest.json"))
windows = []
for vid, m in manifest.items():
    t = 0.0
    while t < m["duration_s"]:
        t_end = min(t + WIN, m["duration_s"])
        if t_end - t >= 2.0:  # 2sn'den kısa artık pencereyi atla
            windows.append({"video_id": vid, "t_start": round(t, 2),
                            "t_end": round(t_end, 2)})
        t += STRIDE
json.dump(windows, open("data/windows.json", "w"))
print(len(windows), "pencere")
```

### 1.2 Model-agnostik embedding (Adım 0'ın kalbi)

Rapordaki en önemli disiplin "modeli kendi verinle doğrula" idi. Bunu koda gömüyoruz: tüm modeller tek arayüzü uygular, kıyaslama harness'ı model adını parametre alır.

```python
# models/base.py
from abc import ABC, abstractmethod
import numpy as np

class VideoTextEmbedder(ABC):
    name: str
    dim: int

    @abstractmethod
    def embed_video(self, frames: list[np.ndarray]) -> np.ndarray:
        """8sn pencereden örneklenmiş kareler -> tek L2-normalize vektör"""

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        ...
```

```python
# models/xclip_hf.py — hızlı başlangıç adayı
# DİKKAT: Bu, HuggingFace'teki microsoft/xclip (Ni et al., Kinetics).
# Raporun önerdiği retrieval-özel X-CLIP (Ma et al., AOSM) DEĞİL.
# POC'ta ilk çalışan uçtan-uca hat için pragmatik başlangıç; kıyaslamada
# ikisi de ayrı adapter olarak koşulmalı ve sonuç tabloya öyle yazılmalı.
import torch, numpy as np
from transformers import AutoProcessor, AutoModel
from .base import VideoTextEmbedder

class XClipHF(VideoTextEmbedder):
    name, dim = "xclip_hf_zeroshot", 512
    def __init__(self, device="cuda"):
        mid = "microsoft/xclip-base-patch16-zero-shot"
        self.proc = AutoProcessor.from_pretrained(mid)
        self.model = AutoModel.from_pretrained(mid).to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed_video(self, frames):
        # model 32 kare bekler; pencereden uniform örnekle
        idx = np.linspace(0, len(frames) - 1, 32).astype(int)
        inp = self.proc(videos=[[frames[i] for i in idx]],
                        return_tensors="pt").to(self.device)
        v = self.model.get_video_features(**inp)
        return torch.nn.functional.normalize(v, dim=-1)[0].cpu().numpy()

    @torch.no_grad()
    def embed_text(self, text):
        inp = self.proc(text=[text], return_tensors="pt",
                        padding=True).to(self.device)
        t = self.model.get_text_features(**inp)
        return torch.nn.functional.normalize(t, dim=-1)[0].cpu().numpy()
```

```python
# models/siglip_avg.py — "mevcut sistem" proxy'si (kare-ortalama baseline)
# Bu baseline şart: hibrit sistemin kazancını NEYE GÖRE ölçtüğünüzü tanımlar.
import torch, numpy as np
from transformers import AutoProcessor, AutoModel
from .base import VideoTextEmbedder

class SiglipAvg(VideoTextEmbedder):
    name, dim = "siglip2_frameavg", 1152
    def __init__(self, device="cuda"):
        mid = "google/siglip2-so400m-patch14-384"
        self.proc = AutoProcessor.from_pretrained(mid)
        self.model = AutoModel.from_pretrained(mid).to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed_video(self, frames):
        idx = np.linspace(0, len(frames) - 1, 8).astype(int)
        inp = self.proc(images=[frames[i] for i in idx],
                        return_tensors="pt").to(self.device)
        f = self.model.get_image_features(**inp)
        f = torch.nn.functional.normalize(f, dim=-1).mean(0)
        return torch.nn.functional.normalize(f, dim=0).cpu().numpy()

    @torch.no_grad()
    def embed_text(self, text):
        inp = self.proc(text=[text], return_tensors="pt",
                        padding="max_length").to(self.device)
        t = self.model.get_text_features(**inp)
        return torch.nn.functional.normalize(t, dim=-1)[0].cpu().numpy()
```

Aday listesi (her biri bir adapter dosyası):

| Adapter | Model | Neden listede |
|---|---|---|
| `siglip_avg` | SigLIP2 kare-ortalama | Mevcut sistem proxy'si — baseline |
| `xclip_hf` | microsoft/xclip-zero-shot | 5 dk'da çalışır, boru hattını açar |
| `xclip_ma` | Ma et al. X-CLIP (AOSM) | Raporun asıl önerisi |
| `languagebind` | LanguageBind-Video | Güçlü açık alternatif |
| `internvideo2_s1` | InternVideo2 Stage-1 | Raporun "mümkünse" dediği kontrol |

```python
# ingest/03_embed.py
import json, cv2, numpy as np, argparse
from models import get_embedder  # registry: isim -> sınıf

def read_window(video_path, t0, t1, n=32):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    for t in np.linspace(t0, t1, n, endpoint=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    emb = get_embedder(args.model)
    windows = json.load(open("data/windows.json"))
    out = []
    for w in windows:
        frames = read_window(f"data/raw/videos/{w['video_id']}.mp4",
                             w["t_start"], w["t_end"])
        vec = emb.embed_video(frames)
        out.append({**w, "embedding": vec.tolist()})
    json.dump(out, open(f"data/embeddings_{emb.name}.json", "w"))
```

### 1.3 Detektör → filtre kolonları (telemetri simülasyonu)

```python
# ingest/04_detect.py
# YOLO26 hedef; ortamda yoksa yolo11x aynı API ile çalışır (ultralytics).
import json, cv2, numpy as np
from ultralytics import YOLO

model = YOLO("yolo26x.pt")  # fallback: "yolo11x.pt"
COCO_MAP = {0: "person", 2: "car", 5: "bus", 7: "truck"}

def window_features(video_path, t0, t1, n_sample=6):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    counts = {v: [] for v in COCO_MAP.values()}
    brightness, prev_gray, motions = [], None, []
    for t in np.linspace(t0, t1, n_sample, endpoint=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if not ok:
            continue
        res = model(fr, verbose=False)[0]
        cls = res.boxes.cls.cpu().numpy().astype(int)
        for cid, cname in COCO_MAP.items():
            counts[cname].append(int((cls == cid).sum()))
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        brightness.append(float(gray.mean()))
        small = cv2.resize(gray, (160, 90))
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, small, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            motions.append(float(np.linalg.norm(flow, axis=2).mean()))
        prev_gray = small
    cap.release()
    return {
        "person_count": int(np.median(counts["person"] or [0])),
        "car_count":    int(np.median(counts["car"] or [0])),
        "bus_count":    int(np.median(counts["bus"] or [0])),
        "truck_count":  int(np.median(counts["truck"] or [0])),
        "brightness":   float(np.mean(brightness or [0])),
        "is_night":     bool(np.mean(brightness or [255]) < 60),
        "camera_motion": float(np.mean(motions or [0])),  # üretimdeki 'hız' analoğu
    }
```

> **Önemli sınır:** COCO-önceden-eğitilmiş YOLO havadan küçük nesnelerde zayıftır — VisDrone bu yüzden bir benchmark. POC'ta bu **kabul edilebilir**, çünkü GT anotasyondan geliyor; detektör kolonlarının hatası ayrı bir ölçüm olarak raporlanır ("filtre katmanı kirliyse hibrit ne kadar bozulur" — üretim için değerli bilgi). İkinci turda VisDrone-train ile fine-tune edilmiş bir YOLO bu tabloyu iyileştirir; ayrıca aynı ölçüm "yanlış filtre = kalıcı kaçırma" riskinin (raporun 6.2'deki uyarısı) büyüklüğünü sayısallaştırır.

### 1.4 ClickHouse şema ve yükleme

```sql
-- schema.sql
CREATE TABLE clips (
    video_id      LowCardinality(String),
    t_start       Float32,
    t_end         Float32,
    -- filtre kolonları (üretimde telemetri, POC'ta detektör):
    person_count  UInt16,
    car_count     UInt16,
    bus_count     UInt16,
    truck_count   UInt16,
    is_night      Bool,
    camera_motion Float32,
    brightness    Float32,
    -- platform kolonu (POC'ta sabit; üretimde Postgres join'den — 2. mesajdaki eksik):
    platform      LowCardinality(String) DEFAULT 'visdrone',
    embedding     Array(Float32),
    INDEX idx_vec embedding TYPE vector_similarity('hnsw', 'cosineDistance', 512),
    INDEX idx_bus bus_count TYPE minmax GRANULARITY 4,
    INDEX idx_person person_count TYPE minmax GRANULARITY 4
) ENGINE = MergeTree
ORDER BY (video_id, t_start);
```

```python
# ingest/05_load_clickhouse.py
import json, clickhouse_connect

ch = clickhouse_connect.get_client(host="localhost")
ch.command(open("schema.sql").read())
emb = {(e["video_id"], e["t_start"]): e["embedding"]
       for e in json.load(open("data/embeddings_xclip_hf_zeroshot.json"))}
feats = json.load(open("data/features.json"))
rows = []
for f in feats:
    key = (f["video_id"], f["t_start"])
    if key in emb:
        rows.append([f["video_id"], f["t_start"], f["t_end"],
                     f["person_count"], f["car_count"], f["bus_count"],
                     f["truck_count"], f["is_night"], f["camera_motion"],
                     f["brightness"], "visdrone", emb[key]])
ch.insert("clips", rows, column_names=[
    "video_id","t_start","t_end","person_count","car_count","bus_count",
    "truck_count","is_night","camera_motion","brightness","platform","embedding"])
```

---

## Faz 2 — Sorgu Hattı (1 hafta)

### 2.1 Ayrıştırıcı — önce kural tabanlı, sonra LLM

İlk hafta LLM'siz başlayın: sorgu uzayınız dar ("X'i göster", "X ve Y"), regex + sözlük yeter ve hata ayıklaması trivial. LLM'i (Qwen + JSON şema zorlama) ayrıştırıcı **arayüzü sabitken** takas edin — böylece "LLM ayrıştırma hatası" ile "retrieval hatası" birbirine karışmaz. Bu ayrım, üretimdeki "yanlış filtre çevirisi kalıcı kaçırma üretir" riskini izole ölçmenizi sağlar.

```python
# search/parser.py
import re
from dataclasses import dataclass, field

CONCEPT_MAP = {  # üretimdeki "alan kataloğu"nun POC karşılığı
    r"otobüs":              ("bus_count", ">=", 1),
    r"kamyon":              ("truck_count", ">=", 1),
    r"araba|araç|otomobil": ("car_count", ">=", 1),
    r"insan|adam|yaya|kişi":("person_count", ">=", 1),
    r"gece":                ("is_night", "=", 1),
    r"gündüz":              ("is_night", "=", 0),
    r"kalabalık":           ("person_count", ">=", 10),
}
# Kolonlaşamayan kavramlar semantik artığa düşer (rapor Adım 1 ile aynı ilke):
SEMANTIC_ONLY = ["yürüyen", "koşan", "dönen", "bekleyen", "hızlı", "kavşak",
                 "otopark", "yol kenarı"]

@dataclass
class ParsedQuery:
    filters: list = field(default_factory=list)   # (col, op, val)
    semantic: str = ""                            # embedding'e gidecek metin

def parse(q: str) -> ParsedQuery:
    p = ParsedQuery(semantic=q)  # semantik her zaman TAM sorguyu alır
    for pat, cond in CONCEPT_MAP.items():
        if re.search(pat, q, re.IGNORECASE):
            p.filters.append(cond)
    return p
```

> **Tasarım notu:** Semantik artığa filtrelenen kelimeleri *çıkarılmış* metin değil **tam sorgu** veriyoruz. "Otobüs" filtreye düşse bile embedding'in de görmesi zarar vermez, tersine sahne bağlamını korur; raporun "filtreler optimizasyon katmanıdır, zorunluluk değil" ilkesinin sorgu-tarafı karşılığı budur.

### 2.2 Hibrit ClickHouse sorgusu

```python
# search/query.py
import clickhouse_connect
from models import get_embedder
from .parser import parse

ch = clickhouse_connect.get_client(host="localhost")

def search(q: str, model_name: str, top_k: int = 200):
    p = parse(q)
    emb = get_embedder(model_name)
    qvec = emb.embed_text(p.semantic).tolist()
    where = " AND ".join(f"{c} {op} {v}" for c, op, v in p.filters) or "1"
    sql = f"""
        SELECT video_id, t_start, t_end,
               cosineDistance(embedding, {qvec}) AS dist
        FROM clips
        WHERE {where}
        ORDER BY dist ASC
        LIMIT {top_k}
    """
    return ch.query(sql).result_rows
```

İki mod ölçülür: (a) filtre AÇIK — yukarıdaki sorgu; (b) filtre KAPALI — `WHERE 1`, saf vektör. Bu A/B, hibritin kazancını **aynı embedding modeliyle** izole eder; model kıyaslaması ise aynı sorguyu farklı embedding tablolarında koşturur. İki eksen (model × filtre) çapraz tablo verir — raporun 4.3 tablosunun POC versiyonu.

### 2.3 Aralık birleştirme (istediğiniz çıktı formatı)

```python
# search/merge.py
def merge_intervals(rows, gap_tol=10.0, min_score=None):
    """rows: (video_id, t_start, t_end, dist) — pencere sonuçları ->
    videolara ayrıştırıp <=gap_tol boşlukla birleşik aralıklara indirger."""
    by_vid = {}
    for vid, t0, t1, dist in rows:
        score = 1 - dist
        if min_score and score < min_score:
            continue
        by_vid.setdefault(vid, []).append((t0, t1, score))
    results = []
    for vid, wins in by_vid.items():
        wins.sort()
        cur = list(wins[0])
        for t0, t1, s in wins[1:]:
            if t0 - cur[1] <= gap_tol:
                cur[1] = max(cur[1], t1)
                cur[2] = max(cur[2], s)  # aralık skoru = en iyi pencere
            else:
                results.append((vid, cur[0], cur[1], cur[2]))
                cur = [t0, t1, s]
        results.append((vid, cur[0], cur[1], cur[2]))
    results.sort(key=lambda r: -r[3])
    return results

def fmt(t):
    h, r = divmod(int(t), 3600); m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}"

def pretty(results, n=10):
    return "\n".join(f"{vid}  {fmt(a)}–{fmt(b)}  (skor {s:.2f})"
                     for vid, a, b, s in results[:n])
```

---

## Faz 3 — Otomatik Ground Truth + Değerlendirme (1 hafta, faz 1-2 ile paralel başlar)

### 3.1 Anotasyondan GT aralıkları

Sistemin en değerli parçası. VisDrone-MOT anotasyonu kare bazlı olduğu için her sorgu kavramı bir "kare yüklemi"ne, o da zaman aralıklarına çevrilir:

```python
# eval/make_groundtruth.py
import json, pathlib
from collections import defaultdict

CAT = {"pedestrian": 1, "people": 2, "car": 4, "van": 5,
       "truck": 6, "bus": 9}
FPS = 25
WALK_PX_PER_S = 15.0   # track merkez yer değiştirme eşiği (kalibre edilecek)

def load_ann(path):
    """frame -> [(track_id, cat, cx, cy)]"""
    frames = defaultdict(list)
    for line in open(path):
        f, tid, x, y, w, h, sc, cat, tr, oc = map(float, line.split(","))
        if sc == 0:  # ignore bölgesi
            continue
        frames[int(f)].append((int(tid), int(cat), x + w/2, y + h/2))
    return frames

def frames_to_intervals(frame_flags, fps=FPS, min_dur=1.0, gap_tol_s=2.0):
    """[bool per frame] -> [(t0, t1)] birleşik aralıklar"""
    iv, start = [], None
    gap_frames = int(gap_tol_s * fps)
    last_true = -10**9
    for i, flag in enumerate(frame_flags):
        if flag:
            if start is None:
                start = i
            last_true = i
        elif start is not None and i - last_true > gap_frames:
            iv.append((start / fps, last_true / fps)); start = None
    if start is not None:
        iv.append((start / fps, last_true / fps))
    return [(a, b) for a, b in iv if b - a >= min_dur]

def gt_object(frames, n_frames, cat_name):
    cid = CAT[cat_name]
    flags = [any(c == cid for _, c, _, _ in frames.get(i, []))
             for i in range(1, n_frames + 1)]
    return frames_to_intervals(flags)

def gt_walking(frames, n_frames):
    """pedestrian track'in ~1sn penceredeki merkez yer değiştirmesi eşik üstü mü"""
    flags = [False] * n_frames
    tracks = defaultdict(dict)  # tid -> {frame: (cx, cy)}
    for f, objs in frames.items():
        for tid, cat, cx, cy in objs:
            if cat == CAT["pedestrian"]:
                tracks[tid][f] = (cx, cy)
    for tid, pos in tracks.items():
        fs = sorted(pos)
        for f in fs:
            f2 = f + FPS  # 1 sn sonrası
            if f2 in pos:
                dx = pos[f2][0] - pos[f][0]
                dy = pos[f2][1] - pos[f][1]
                if (dx*dx + dy*dy) ** 0.5 >= WALK_PX_PER_S:
                    for k in range(f, min(f2, n_frames)):
                        flags[k] = True
    return frames_to_intervals(flags)

def intersect(iv_a, iv_b):
    out = []
    for a0, a1 in iv_a:
        for b0, b1 in iv_b:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi - lo >= 1.0:
                out.append((lo, hi))
    return out

QUERIES = {
    # kademeli zorluk — sizin "genelden özele" progresyonunuz:
    "otobüsü göster":              lambda F, N: gt_object(F, N, "bus"),
    "kamyonu göster":              lambda F, N: gt_object(F, N, "truck"),
    "arabaları göster":            lambda F, N: gt_object(F, N, "car"),
    "yürüyen adamı göster":        lambda F, N: gt_walking(F, N),
    "otobüs ve yürüyen adam":      lambda F, N: intersect(
                                       gt_object(F, N, "bus"), gt_walking(F, N)),
    "kamyon ve yaya birlikte":     lambda F, N: intersect(
                                       gt_object(F, N, "truck"),
                                       gt_object(F, N, "pedestrian")),
}

if __name__ == "__main__":
    manifest = json.load(open("data/raw/manifest.json"))
    ann_dir = pathlib.Path("data/annotations")
    gt = defaultdict(dict)
    for vid, m in manifest.items():
        frames = load_ann(ann_dir / f"{vid}.txt")
        for q, fn in QUERIES.items():
            iv = fn(frames, m["n_frames"])
            if iv:
                gt[q][vid] = iv
    json.dump(gt, open("data/groundtruth/gt.json", "w"), indent=1)
```

> **Kamera hareketi tuzağı:** Drone kendisi hareket ederken sabit duran bir yaya da piksel uzayında yer değiştirir — `gt_walking` yanlış pozitif üretir. İlk kalibrasyonda 5-10 sekansı gözle kontrol edin (FiftyOne, aşağıda); gerekirse yer değiştirmeyi aynı karedeki araç/statik nesne track'lerinin medyan hareketinden arındırın (ego-motion kompanzasyonu ~15 satır ek kod). Okutama'nın açık "walking" etiketi faz 2'de bu sezgiselin bağımsız doğrulamasıdır.

### 3.2 Metrikler (bir önceki mesajdaki çerçevenin kodu)

```python
# eval/metrics.py
def t_iou(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0

def evaluate(pred, gt, k=10, iou_thr=0.5):
    """pred: [(vid, t0, t1, skor)] sıralı; gt: {vid: [(t0, t1)]}"""
    gt_flat = [(v, iv) for v, ivs in gt.items() for iv in ivs]
    hits, matched = 0, set()
    for vid, t0, t1, _ in pred[:k]:
        ok = False
        for gi, (gvid, giv) in enumerate(gt_flat):
            if gi in matched or gvid != vid:
                continue
            if t_iou((t0, t1), giv) >= iou_thr:
                matched.add(gi); ok = True; break
        hits += ok
    n_pred = min(k, len(pred))
    return {
        "precision@k": hits / n_pred if n_pred else 0.0,
        "recall@k": len(matched) / len(gt_flat) if gt_flat else 0.0,
        "n_gt": len(gt_flat), "n_pred": n_pred,
    }
```

```python
# eval/run_eval.py
import json, itertools
from search.query import search
from search.merge import merge_intervals
from eval.metrics import evaluate

gt_all = json.load(open("data/groundtruth/gt.json"))
MODELS = ["siglip2_frameavg", "xclip_hf_zeroshot"]  # adapter eklendikçe büyür

rows = []
for model, use_filter in itertools.product(MODELS, [True, False]):
    for q, gt in gt_all.items():
        raw = search(q, model, use_filters=use_filter)
        pred = merge_intervals(raw)
        m = evaluate(pred, gt)
        rows.append({"model": model, "filter": use_filter, "query": q, **m})

# kategori kırılımı: tekli / hareket / bileşik
def cat(q):
    if " ve " in q or "birlikte" in q: return "bileşik"
    if "yürüyen" in q or "koşan" in q: return "hareket"
    return "tekli"
for r in rows:
    r["category"] = cat(r["query"])
json.dump(rows, open("results.json", "w"), indent=1)
```

Beklenen tablo şekli (raporun %3-13 → %55-75 iddiasının POC sınaması):

| model | filtre | tekli P@10 | hareket P@10 | bileşik P@10 |
|---|---|---|---|---|
| siglip_avg (mevcut proxy) | kapalı | ? | ? | ? |
| siglip_avg | açık | ? | ? | ? |
| xclip | kapalı | ? | ? | ? |
| xclip | açık | ? | ? | ? |

Dört hipotez sınanır: (1) filtre AÇIK her modelde bileşik sorguyu iyileştirir mi, (2) klip-embedding hareket sorgusunda kare-ortalamayı geçer mi, (3) iki etki toplanır mı, (4) filtre kirliliği (detektör hatası) hangi noktada kazancı yer.

### 3.3 Görsel inceleme (FiftyOne)

```python
# notebooks/inspect_fiftyone.py
import fiftyone as fo, json

ds = fo.Dataset("poc-results", overwrite=True)
results = json.load(open("results_detail.json"))  # sorgu bazlı pred+gt aralıkları
for r in results:
    s = fo.Sample(filepath=f"data/raw/videos/{r['video_id']}.mp4")
    s["query"] = r["query"]
    s["pred"] = fo.TemporalDetections(detections=[
        fo.TemporalDetection(label="pred", support=[int(a*25), int(b*25)])
        for a, b, _ in r["pred"]])
    s["gt"] = fo.TemporalDetections(detections=[
        fo.TemporalDetection(label="gt", support=[int(a*25), int(b*25)])
        for a, b in r["gt"]])
    ds.add_sample(s)
fo.launch_app(ds)
```

Bu, sizin "video indirip elle bakma" pratiğinizin araçlaşmış hali: pred/gt aralıkları timeline'da yan yana, yanlışın *neden* yanlış olduğu (yanlış video mu, doğru video yanlış aralık mı, aralık sınırı mı kaymış) 10 saniyede görülür.

---

## Faz 4 — Model Bake-off + Ölçek Testi (1 hafta)

**4.1 Bake-off:** `xclip_ma` (orijinal AOSM X-CLIP), `languagebind`, `internvideo2_s1` adapter'ları eklenir; `run_eval.py` değişmeden tümü koşulur. Kazanan model raporun Adım 0 kapısını POC verisiyle geçmiş olur. (Not: orijinal X-CLIP repo'su araştırma kodu — CLIP4Clip altyapısına dayanır, checkpoint indirme + önişleme uyarlaması 1-2 gün sürebilir; `xclip_hf` bu yüzden ilk hafta koşulan pragmatik vekildir ve sonuç tablosunda ikisi ayrı satır olarak raporlanır.)

**4.2 ClickHouse ölçek testi** (2. mesajdaki HNSW şüphesinin sınaması):

```python
# scale_test.py — vektörleri sentetik çoğaltıp gecikme eğrisi çıkar
# gerçek dağılımı bozmamak için mevcut vektörlere küçük gauss gürültüsü ekle
import numpy as np, clickhouse_connect, time

ch = clickhouse_connect.get_client(host="localhost")
base = ch.query("SELECT embedding FROM clips").result_rows
target_sizes = [100_000, 1_000_000, 10_000_000]
# ... her hedefte: kopyala+gürültüle+insert, OPTIMIZE TABLE FINAL,
#     ardından filtreli ve filtresiz 50 sorgu koş, p50/p95 kaydet.
```

Çıktı: satır sayısı × (filtreli / filtresiz) gecikme eğrisi. 10M'de filtreli sorgu p95 > ~1sn ise, üretimdeki 270M için raporun "Qdrant B planı" spekülasyon olmaktan çıkar — bu bulgu tek başına POC'un parasını öder.

**4.3 Quantization mini-testi (opsiyonel):** Aynı harness'ta embedding'leri int8'e indirip P@10 farkını ölç — raporun 5.1 merdiveninin ilk basamağının kanıtı, ~yarım gün.

---

## Faz 5 — Üretime Köprü (POC bitiminde, planlama)

POC'tan üretime taşınan değişmezler: pencere sınırları (8s/4s), ClickHouse satır iskeleti, sorgu hattı (parser arayüzü → SQL → merge), değerlendirme harness'ı ve GT metodolojisi. Değişenler:

1. **Aktivite 2 takası:** detektör-kolonları → telemetri-kolonları. Ön koşul: telemetri formatı doğrulaması (MAVLink mi, STANAG 4609/KLV mi — 2. mesajdaki bloke edici madde). Parser'daki `CONCEPT_MAP` alan kataloğunun v0'ı olur; `platform` kolonu Postgres join ile dolar.
2. **Kural-tabanlı parser → LLM parser:** arayüz (`ParsedQuery`) aynı kaldığı için takas izole; LLM'in ürettiği filtreler kural-tabanlının ürettiğiyle regresyon setinde kıyaslanır ("yanlış filtre çevirisi" metriği doğar).
3. **GT metodolojisi → altın set:** VisDrone'un otomatik GT'si yerini kurum verisinde elle etiketlenen 200-500 çifte bırakır; ama tIoU/kategori-kırılımı/çift-etiketleyici çerçevesi ve FiftyOne akışı aynen taşınır. POC'un `QUERIES` sözlüğündeki kademeli zorluk yapısı ("tekli → hareket → bileşik") altın setin kategori şablonudur; kurum setine "görsel özdeş ayrım" (günbatımı/gündoğumu) ve zor-negatif çiftler eklenir.
4. **Orkestrasyon:** `make ingest` → Temporal workflow (raporun Aktivite 1-5'i).

---

## Zaman Çizelgesi ve Karar Kapıları

| Hafta | İş | Kapı (geç/kal kararı) |
|---|---|---|
| 1 | Faz 0 + pencereleme + ilk embedding (`xclip_hf`) + şema | Uçtan uca tek sorgu çalışıyor mu ("otobüsü göster" → aralık listesi) |
| 2 | Detektör kolonları + parser + merge + GT üretici | Otomatik GT gözle doğrulandı mı (FiftyOne, 5-10 sekans) |
| 3 | Tam eval: 2 model × 2 mod × 6 sorgu + FiftyOne akışı | Filtre AÇIK bileşik sorguda anlamlı kazanç veriyor mu |
| 4 | Bake-off (3+ model) + ölçek testi + (ops.) quantization | Kazanan model + ClickHouse gecikme eğrisi raporu |
| 5+ | Faz 5 planlaması, kurum verisi hazırlığı | Telemetri formatı doğrulandı mı |

**Donanım:** tek GPU'lu bir makine (24GB VRAM rahat; VisDrone-MOT ~33K kare → tüm embedding turu saatler mertebesi). GPU yoksa `xclip_hf` base varyantı CPU'da yavaş ama koşar; bake-off için GPU şart.

**Başarı tanımı:** POC "X-CLIP iyi mi" sorusuna değil şu üç soruya cevap verir — (1) hibrit mimari bileşik sorgularda saf-vektöre karşı ölçülebilir kazanç sağlıyor mu, (2) hangi açık model bu veride önde, (3) ClickHouse bu iş yükünü hangi ölçeğe kadar taşıyor. Üçü de evet/rakamla cevaplanınca, kurum verisiyle Adım 0'a girme kararı veri-destekli olur.
