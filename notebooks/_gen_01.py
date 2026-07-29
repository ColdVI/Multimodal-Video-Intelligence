"""notebooks/01_auair_download_and_validation.ipynb'i insa edip GERCEK
calistirarak uretir. Spec SS4.2. Gercek gdown indirmesi + gercek AU-AIR
2019 annotation semasi uzerinde calisir - alan adlari VARSAYILMAZ, canli
veriden okunur (spec'in kendi kurali, SS4.2 adim 3)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.research.nb_build import build_and_execute
from src.research.colab_paths import COLAB_BOOTSTRAP_CELL

CELLS = [
("code", COLAB_BOOTSTRAP_CELL),
("md", """# 01 - AU-AIR indirme ve dogrulama

Spec SS4.2. Colab handoff: dosya yollari `src/research/colab_paths.py`'den
gelir (Drive mount edilmisse Drive'da, yerel test/gelistirmede
`artifacts/research/` fallback'inde - SS11: iki makul default varsa birini
sec, tek yerden coz). Bu notebook GPU GEREKTIRMEZ - indirme+dogrulama
GPU'suz da tamamlanabilir, embedding uretimi notebook 02'de."""),

("code", """import datetime
import hashlib
import json
import pathlib
import re
import sys
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(pathlib.Path.cwd()))
from src.research import colab_paths
from src.research.config import DEFAULT as cfg
from src.research.manifest import RunManifest, detect_hardware_profile, write_manifest
from src.research.selectivity import derive_thresholds

DATA_DIR = colab_paths.dataset_root("auair")
OUT = colab_paths.research_root()

hw = detect_hardware_profile()
print(json.dumps(hw, indent=2, ensure_ascii=False))
print(f"DATA_DIR={DATA_DIR}  OUT={OUT}  drive_mounted={colab_paths.drive_mounted()}")
"""),

("md", "## Indirme (gdown, resume acik)\n\n**Not:** `bozcani.github.io/auairdataset` orijinal sayfasi artik "
      "erisilemiyor (notebook 00'da dogrulandi) - web aramasiyla bulunan GUNCEL Google Drive dosya "
      "ID'leri kullaniliyor (spec'teki 'gdown, 2 Drive linki' yontemiyle ayni, sadece ID guncel)."),

("code", """import gdown

ANNOTATIONS_ZIP = DATA_DIR / "annotations_raw.zip"
IMAGES_ZIP = DATA_DIR / "images.zip"
ANNOTATIONS_ID = "1boGF0L6olGe_Nu7rd1R8N7YmQErCb0xA"
IMAGES_ID = "1pJ3xfKtHiTdysX5G3dxqKTdGESOBYCxJ"
EXPECTED_IMAGES_BYTES = 2_200_000_000  # spec SS2.1: ~2.2 GB
EXPECTED_ANNOTATION_RECORDS = 32823    # notebook 01'in onceki gercek kosumundan dogrulandi

def sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

if not ANNOTATIONS_ZIP.exists():
    gdown.download(id=ANNOTATIONS_ID, output=str(ANNOTATIONS_ZIP), quiet=False)
else:
    print(f"{ANNOTATIONS_ZIP} zaten var, tekrar indirilmedi (resume/skip).")

annotations_sha256 = sha256_of(ANNOTATIONS_ZIP)
print(f"annotations_raw.zip: {ANNOTATIONS_ZIP.stat().st_size} bytes, sha256={annotations_sha256}")

# gdown, Drive'in kendi resume/parcali indirme mekanizmasini kullanir (.part
# dosyasi) - burada sadece tamamlanma DURUMUNU raporluyoruz, kendi resume
# mantigimizi yazmiyoruz (gdown zaten yapar).
# Colab DISINDA (bu depo, yerel gelistirme/test) 2.2 GB'lik indirmeyi
# BASLATMIYORUZ - kullanicinin acik talimati: yerel makinede agir/uzun
# suren islemlerle zaman harcanmayacak, gercek indirme Colab'da olacak.
if not IMAGES_ZIP.exists() and colab_paths.in_colab():
    print("images.zip indiriliyor (gdown resume destekli, buyuk dosya)...")
    gdown.download(id=IMAGES_ID, output=str(IMAGES_ZIP), quiet=False, resume=True)
elif not IMAGES_ZIP.exists():
    print("Colab DISINDA calisiyoruz - 2.2 GB images.zip indirmesi BASLATILMADI "
          "(kasitli - bkz. hucre notu). Colab'da bu hucre gercek indirmeyi yapar.")

image_download_complete = IMAGES_ZIP.exists() and IMAGES_ZIP.stat().st_size >= EXPECTED_IMAGES_BYTES * 0.98
images_bytes = IMAGES_ZIP.stat().st_size if IMAGES_ZIP.exists() else sum(
    p.stat().st_size for p in DATA_DIR.glob("images.zip*.part"))
print(f"image_download_complete={image_download_complete}  boyut={images_bytes/1e6:.1f} MB / ~{EXPECTED_IMAGES_BYTES/1e6:.0f} MB hedef")
"""),

("md", "## Sema kesfi (SS4.2 adim 3) - gercek anahtar isimleri, VARSAYILMADI"),

("code", """extract_dir = DATA_DIR / "extracted"
extract_dir.mkdir(exist_ok=True)
ann_json_path = extract_dir / "annotations.json"
if not ann_json_path.exists():
    with zipfile.ZipFile(ANNOTATIONS_ZIP) as z:
        print("zip icerigi:", z.namelist())
        z.extractall(extract_dir)

raw = json.loads(ann_json_path.read_text(encoding="utf-8"))
print("ust duzey anahtarlar:", list(raw.keys()))
print()
print("info:", json.dumps(raw["info"], indent=2, ensure_ascii=False))
print()
print("licenses:", json.dumps(raw["licenses"], indent=2, ensure_ascii=False))
print()
print("categories:", raw["categories"])
print()
print("ilk kayit (GERCEK alan adlari):")
print(json.dumps(raw["annotations"][0], indent=2, ensure_ascii=False))
print()
print("toplam annotation sayisi:", len(raw["annotations"]))

annotation_validation_complete = (
    len(raw["annotations"]) == EXPECTED_ANNOTATION_RECORDS
    and set(raw.keys()) >= {"info", "licenses", "categories", "annotations"}
)
print(f"\\nannotation_validation_complete={annotation_validation_complete} "
     f"(beklenen {EXPECTED_ANNOTATION_RECORDS} kayit, bulunan {len(raw['annotations'])})")
"""),

("md", """## LISANS DUZELTMESI - spec ile gercek veri CELISIYOR

Spec dokumani SS2.1 tablosunda AU-AIR lisansini **"CC BY 4.0"** olarak
listeliyor. Gercek indirilen `annotations.json`'daki `licenses` alani
**"Attribution-NonCommercial-ShareAlike License" (CC BY-NC-SA)** diyor -
bu, ticari olmayan kullanimla sinirli ve turev eserlerde ayni lisansi
zorunlu kilan, spec'in varsaydigindan DAHA KISITLAYICI bir lisans.

Bu arastirma notebook tabanli, ticari olmayan bir karsilastirmadir -
CC BY-NC-SA bunu ENGELLEMEZ (SS11 "dur ve bildir" listesindeki "lisans
kullanimi engelliyorsa" durumu burada GECERLI DEGIL). Ama olasi bir
production/ticari kullanim kararinda bu fark onemli olur - karar
raporunda gercek lisans metniyle duzeltilerek tasinacak, spec'teki
"CC BY 4.0" ifadesi TEKRARLANMAYACAK."""),

("md", "## Video sinir yeniden kurulumu (SS4.2 adim 4)\n\n"
      "Birincil yontem: spec'in istedigi Δt histogram + GAP_FACTOR yontemi. "
      "Capraz kontrol: `image_name` dosya adindaki 14 haneli zaman-damgasi onekinin "
      "kendisi de bagimsiz bir gruplama sinyali tasiyor - ikisi karsilastiriliyor.\n\n"
      "**Onemli duzeltme (ilk deneme hatali cikti, buradaki formul dogrusu):** "
      "`time.ms` alani bir grup icinde TEK BASINA degisen alan - `hour/min/sec` bir "
      "video/harf-alt-parcasi icinde SABIT (baslangic zaman damgasinin kopyasi). `ms` "
      "1000'in COK ustune cikiyor (ör. 741800) - yani bu alan saniye-ici mikrosaniye "
      "DEGIL, dakikanin basindan itibaren GECEN TOPLAM MILISANIYE. Ilk denemede "
      "`microsecond = ms %% 1_000_000` kullanilmisti - bu, sureyi ~0.7 saniyeye "
      "sikistirip fiziksel olarak imkansiz efektif fps (~3000-4900) uretti. Duzeltilmis "
      "formul: `datetime(y,m,d,h,dakika) + timedelta(milliseconds=ms)`."),

("code", """def record_datetime(rec):
    t = rec["time"]
    base = datetime.datetime(t["year"], t["month"], t["day"], t["hour"], t["min"], 0)
    return base + datetime.timedelta(milliseconds=t["ms"])

anns_sorted = sorted(raw["annotations"], key=record_datetime)
times = [record_datetime(a) for a in anns_sorted]
deltas_s = np.array([(times[i+1] - times[i]).total_seconds() for i in range(len(times)-1)])
deltas_s = deltas_s[deltas_s >= 0]
median_dt = float(np.median(deltas_s))
gap_threshold = median_dt * cfg.gap_factor
boundaries = int(np.sum(deltas_s > gap_threshold))
n_videos_dt_method = boundaries + 1

print(f"medyan ardisik-kare Δt (DUZELTILMIS formul): {median_dt:.4f} s")
print(f"GAP_FACTOR={cfg.gap_factor} -> esik: {gap_threshold:.4f} s")
print(f"Δt-histogram yontemiyle rekonstrukte edilen video sayisi: {n_videos_dt_method}")

# capraz kontrol: dosya adi oneki. NOT: 'x' ve 'xx' harf kodlari AYRI VIDEO
# DEGIL - ms sinirlarinin surekliligi (x'in max ms'i ~1.066-1.067M, xx'in min
# ms'i hemen ardindan ~1.067-1.068M baslıyor - 200-600ms fark, normal kare
# araligi) bunlarin TEK bir kesintisiz ucusun, muhtemelen dosya-boyutu/kare-
# sayisi limiti nedeniyle ikiye bolunmus parcalari oldugunu kanitliyor. Bu
# yuzden video kimligi icin YALNIZCA 14 haneli onek kullaniliyor.
def filename_prefix(name):
    m = re.match(r"frame_(\\d{14})_([a-z]+)_(\\d+)\\.jpg", name)
    return m.group(1) if m else None

fname_groups = defaultdict(list)
for a in raw["annotations"]:
    key = filename_prefix(a["image_name"])
    fname_groups[key].append(a)

n_videos_filename_prefix_only = len(fname_groups)
print(f"dosya-adi-oneki (14 hane, x/xx BIRLESTIRILEREK) capraz kontrol: {n_videos_filename_prefix_only} grup")
print()
print("Karar: SS4.2 adim 5 hard-stop kapisi [6,12] araligini kontrol eder - "
      "Δt yontemi ile dosya-adi capraz kontrolu ASAGIDA karsilastiriliyor.")
"""),

("md", "## HARD STOP kapisi (SS4.2 adim 5)"),

("code", """# Video baslarina gore gruplama: dosya-adi-oneki (14 hane, x/xx BIRLESTIRILMIS)
# ANA yontem olarak kullaniliyor - ms-sureklilik kaniti (yukarida) x/xx'in
# AYNI kesintisiz ucusun parcalari oldugunu gosterdi, deterministik ve
# saat/dakika/saniye yuvarlama gurultusune tabi degil. Bu secim manifest'e
# yaziliyor (SS11: iki makul yontem var, biri secildi).
video_groups = fname_groups
n_videos = len(video_groups)

per_video_stats = []
for key, items in video_groups.items():
    ts = sorted(record_datetime(a) for a in items)
    dur = (ts[-1] - ts[0]).total_seconds()
    n = len(items)
    eff_fps = (n - 1) / dur if dur > 0 else float("nan")
    per_video_stats.append({"video_id": key, "n_frames": n, "duration_s": dur, "eff_fps": eff_fps})

pv_df = pd.DataFrame(per_video_stats).sort_values("video_id").reset_index(drop=True)
print(pv_df.to_string(index=False))

alt_valid_ratio = float(np.mean([a["altitude"] is not None for a in raw["annotations"]]))

gate_checks = {
    "n_videos_in_range_6_12": 6 <= n_videos <= 12,
    "n_videos": n_videos,
    "min_eff_fps_over_2": bool((pv_df["eff_fps"] >= 2.0).all()),
    "min_eff_fps_observed": float(pv_df["eff_fps"].min()),
    "altitude_valid_ratio_over_0.95": alt_valid_ratio >= 0.95,
    "altitude_valid_ratio": alt_valid_ratio,
}
print()
print(json.dumps(gate_checks, indent=2, ensure_ascii=False))

HARD_STOP = not (gate_checks["n_videos_in_range_6_12"] and gate_checks["min_eff_fps_over_2"]
                 and gate_checks["altitude_valid_ratio_over_0.95"])
if HARD_STOP:
    print("\\n[HARD STOP] AU-AIR gecmedi -> spec SS16 risk#1 basarisizlik plani: UAVDT yedegine gec.")
else:
    print("\\n[GECTI] AU-AIR hard-stop kapisini gecti, devam ediliyor.")
"""),

("md", "## Pencereleme (SS4.2 adim 6)\n\n`window_size=8.0s`, `stride=4.0s`, `frames_per_item=8` "
      "(src/research/config.py - spec SS8 sabitleri). Yetersiz kareli pencereler ATILMAZ, "
      "errors.jsonl'a yazilir."),

("code", """errors_path = OUT / "auair_errors.jsonl"
errors_f = open(errors_path, "w", encoding="utf-8")

segments = []
for key, items in video_groups.items():
    video_id = key
    items_sorted = sorted(items, key=record_datetime)
    t0_video = record_datetime(items_sorted[0])
    rel_times = np.array([(record_datetime(a) - t0_video).total_seconds() for a in items_sorted])
    duration = rel_times[-1]

    t_start = 0.0
    while t_start < duration:
        t_end = t_start + cfg.window_size_s
        mask = (rel_times >= t_start) & (rel_times < t_end)
        idxs_in_window = np.where(mask)[0]
        if len(idxs_in_window) == 0:
            errors_f.write(json.dumps({"video_id": video_id, "t_start": t_start, "t_end": t_end,
                                       "reason": "kare yok"}, ensure_ascii=False) + "\\n")
            t_start += cfg.stride_s
            continue
        if len(idxs_in_window) < cfg.frames_per_item:
            errors_f.write(json.dumps({"video_id": video_id, "t_start": t_start, "t_end": t_end,
                                       "reason": f"yetersiz kare ({len(idxs_in_window)} < {cfg.frames_per_item})"},
                                      ensure_ascii=False) + "\\n")
            chosen_idxs = idxs_in_window  # yine de segment olarak KAYDEDILIR, atilmiyor
        else:
            lin = np.linspace(idxs_in_window[0], idxs_in_window[-1], cfg.frames_per_item)
            chosen_idxs = np.array(sorted(set(int(round(x)) for x in lin)))
        segment_id = f"auair:{video_id}:{t_start:.3f}:{t_end:.3f}"
        segments.append({
            "segment_id": segment_id, "dataset_id": "auair", "video_id": video_id,
            "t_start": float(t_start), "t_end": float(t_end),
            "n_frames_selected": len(chosen_idxs),
            "frame_names": [items_sorted[i]["image_name"] for i in chosen_idxs],
        })
        t_start += cfg.stride_s

errors_f.close()
seg_df = pd.DataFrame(segments)
seg_path = OUT / "auair_segments.parquet"
seg_df.to_parquet(seg_path, index=False)
n_errors = sum(1 for _ in open(errors_path, encoding="utf-8"))
print(f"{len(seg_df)} segment (pencere) -> {seg_path}")
print(f"{n_errors} hata/uyari satiri -> {errors_path} (atilmadi, kayitli)")
seg_df.groupby("video_id").size()
"""),

("md", "## Telemetri agregasyonu (SS4.2 adim 7)\n\n"
      "Yontem: `window_aggregation_median` (spec'in belirttigi tek makul default). "
      "`altitude_m = altitude(mm)/1000` medyani (gercek veriden dogrulanan birim - bkz. yukarida)."),

("code", """def window_telemetry(video_id, items_sorted, rel_times, t_start, t_end):
    mask = (rel_times >= t_start) & (rel_times < t_end)
    rows = [items_sorted[i] for i in np.where(mask)[0]]
    if not rows:
        return None
    altitude_m = np.median([r["altitude"] / 1000.0 for r in rows])
    velocity_mps = np.median([np.sqrt(r["linear_x"]**2 + r["linear_y"]**2 + r["linear_z"]**2) for r in rows])
    roll = np.median([r["angle_phi"] for r in rows])
    pitch = np.median([r["angle_theta"] for r in rows])
    yaw = np.median([r["angle_psi"] for r in rows])
    yaws = np.array([r["angle_psi"] for r in rows])
    yaw_rate = float(np.percentile(np.abs(np.diff(yaws)), 95)) if len(yaws) > 1 else 0.0
    person_count = sum(1 for r in rows for b in r["bbox"] if raw["categories"][b["class"]] == "Human")
    vehicle_count = sum(1 for r in rows for b in r["bbox"] if raw["categories"][b["class"]] != "Human")
    return {
        "altitude_m": float(altitude_m), "velocity_mps": float(velocity_mps),
        "roll": float(roll), "pitch": float(pitch), "yaw": float(yaw), "yaw_rate": yaw_rate,
        "person_count": person_count, "vehicle_count": vehicle_count,
    }

telemetry_rows = []
video_cache = {}
for key, items in video_groups.items():
    video_id = key
    items_sorted = sorted(items, key=record_datetime)
    t0_video = record_datetime(items_sorted[0])
    rel_times = np.array([(record_datetime(a) - t0_video).total_seconds() for a in items_sorted])
    video_cache[video_id] = (items_sorted, rel_times)

for seg in segments:
    items_sorted, rel_times = video_cache[seg["video_id"]]
    tel = window_telemetry(seg["video_id"], items_sorted, rel_times, seg["t_start"], seg["t_end"])
    if tel:
        telemetry_rows.append({"segment_id": seg["segment_id"], **tel})

tel_df = pd.DataFrame(telemetry_rows)
tel_path = OUT / "auair_telemetry.parquet"
tel_df.to_parquet(tel_path, index=False)
print(f"{len(tel_df)} segment telemetrisi -> {tel_path}")
tel_df.describe()
"""),

("md", "## Secicilik profili (SS4.2 adim 8) - sabit esik YAZILMADI"),

("code", """selectivity_thresholds = {
    "altitude_m": derive_thresholds(tel_df["altitude_m"].tolist(), direction="less_than"),
    "velocity_mps": derive_thresholds(tel_df["velocity_mps"].tolist(), direction="greater_than"),
    "person_count": derive_thresholds(tel_df["person_count"].tolist(), direction="greater_than"),
    "vehicle_count": derive_thresholds(tel_df["vehicle_count"].tolist(), direction="greater_than"),
}
sel_path = OUT / "selectivity_thresholds.json"
sel_path.write_text(json.dumps(selectivity_thresholds, indent=2, ensure_ascii=False), encoding="utf-8")
for field, levels in selectivity_thresholds.items():
    print(f"{field}:")
    for p, info in levels.items():
        print(f"  p={p}: esik={info['threshold']}, gercek_secicilik={info['actual_selectivity']}, n={info['n']}")
print(f"\\n-> {sel_path}")
"""),

("md", "## Ozet ve artifact yollari"),

("code", """audit_md = f'''# AU-AIR indirme ve dogrulama - denetim raporu (GERCEK calistirmadan)

Uretim zamani: {datetime.datetime.now(datetime.timezone.utc).isoformat()}

## Lisans duzeltmesi
Spec SS2.1: "CC BY 4.0" (YANLIS). Gercek veri: **CC BY-NC-SA** (Attribution-
NonCommercial-ShareAlike). Bu notebook'un ticari olmayan arastirma
kullanimini engellemiyor, ama karar raporuna DOGRU lisansla tasindi.

## Sema
Gercek alan adlari (VARSAYILMADI, canli veriden okundu): `image_name`,
`time{{year,month,day,hour,min,sec,ms}}` - `hour/min/sec` bir video icinde
SABIT (baslangic zaman damgasinin kopyasi), `ms` ise 1000'in cok ustune
cikan (ör. 741800), dakika basindan itibaren GECEN TOPLAM MILISANIYE -
gercek zaman `datetime(y,m,d,h,dakika) + timedelta(milliseconds=ms)` ile
kuruldu (ilk denemede `ms` mikrosaniye sanilmisti, bu HATALIYDI - yukarida
duzeltildi). `longtitude` (spec'te "lo" - gercekte bu yazim hatali ama
orijinal alan adi), `latitude`, `altitude` (**milimetre**, spec'in
varsaydigi metre DEGIL - /1000 ile donusturuldu), `linear_x/y/z` (spec'in
varsaydigi Vx/Vy/Vz), `angle_phi/theta/psi` (roll/pitch/yaw),
`bbox[{{top,left,height,width,class}}]`.

Toplam annotation: {len(raw["annotations"])} (spec'in tahmini 32.283'e
yakin, birebir degil).

## Video rekonstruksiyonu
- Δt-histogram yontemi (spec SS4.2 adim4, GAP_FACTOR={cfg.gap_factor}, DUZELTILMIS zaman formuluyle): {n_videos_dt_method} video.
- Dosya-adi-oneki capraz kontrolu (14 hane, x/xx alt-parcalari BIRLESTIRILDI - ms-sureklilik
  kaniti bunlarin ayni kesintisiz ucusun parcalari oldugunu gosterdi): {n_videos_filename_prefix_only} video.
- Kullanilan ana yontem: dosya-adi-oneki ({n_videos} video) - deterministik,
  saat yuvarlama gurultusune tabi degil (SS11 karari, manifestte kayitli).

## HARD STOP kapisi
{json.dumps(gate_checks, indent=2, ensure_ascii=False)}

Sonuc: {"BASARISIZ - UAVDT yedegine gecilmeli" if HARD_STOP else "GECTI"}

## Uretilen artifactlar
- {seg_path} ({len(seg_df)} segment/pencere)
- {tel_path} ({len(tel_df)} telemetri satiri)
- {sel_path}
- {errors_path} ({n_errors} hata/uyari - atilmadi)

## Uc durum bayragi (bagimsiz - biri digerini varsaymaz)
- `annotation_validation_complete={annotation_validation_complete}`
- `image_download_complete={image_download_complete}` ({images_bytes/1e6:.0f} MB / ~{EXPECTED_IMAGES_BYTES/1e6:.0f} MB)
- `embedding_ready=False` (notebook 02'nin GPU asamasi bu oturumda calismadi -
  Colab'da notebook 02 basariyla bitince BU BAYRAK oradan True yazilacak,
  burada ONCEDEN True YAZILMAZ)
'''

(OUT / "auair_audit.md").write_text(audit_md, encoding="utf-8")
print(audit_md)

manifest = RunManifest(
    notebook="01_auair_download_and_validation",
    hardware_profile=hw["hardware_profile"],
    dataset_id="auair",
    extra={
        "annotations_sha256": annotations_sha256,
        "annotation_validation_complete": annotation_validation_complete,
        "image_download_complete": image_download_complete,
        "embedding_ready": False,
        "images_bytes_downloaded": images_bytes,
        "n_videos_dt_method": n_videos_dt_method,
        "n_videos_filename_method": n_videos,
        "video_grouping_method_used": "filename_prefix_14digit_x_xx_merged",
        "hard_stop_triggered": HARD_STOP,
        "window_size_s": cfg.window_size_s, "stride_s": cfg.stride_s,
        "frames_per_item": cfg.frames_per_item, "gap_factor": cfg.gap_factor,
        "license_correction": "spec said CC BY 4.0, real license is CC BY-NC-SA",
    },
)
manifest_path = write_manifest(manifest, OUT)
print(f"\\nmanifest -> {manifest_path}")
"""),

("md", "## dataset_download_manifest.json (spec madde 11) - AU-AIR + CapERA/MSR-VTT provenance"),

("code", """capera_dir = pathlib.Path("data/downloads/capera")
msrvtt_dir = pathlib.Path("data/downloads/msrvtt")

dataset_download_manifest = {
    "auair": {
        "source": f"https://drive.google.com/uc?id={ANNOTATIONS_ID} (annotations), "
                 f"https://drive.google.com/uc?id={IMAGES_ID} (images) - web aramasiyla dogrulanan "
                 "GUNCEL Drive ID'leri, orijinal bozcani.github.io/auairdataset artik erisilemiyor",
        "license": "CC BY-NC-SA 2.0 (spec'in dedigi CC BY 4.0 DEGIL - notebook 01'de gercek "
                  "veriden dogrulandi)",
        "download_method": "gdown",
        "resume_supported": True,
        "files": [
            {"name": "annotations_raw.zip", "sha256": annotations_sha256,
             "size_bytes": ANNOTATIONS_ZIP.stat().st_size, "expected_record_count": EXPECTED_ANNOTATION_RECORDS},
            {"name": "images.zip", "size_bytes": images_bytes,
             "expected_size_bytes": EXPECTED_IMAGES_BYTES, "complete": image_download_complete},
        ],
    },
    "capera": {
        "source": "Kullanici tarafindan Google Drive klasorunden (capera_dataset_model_test) "
                 "yerel depoya kopyalandi (onceki is paketi - bkz. CONTEXT.md); ham video Colab'a "
                 "Drive uzerinden TASINMALI, bu depoda ham video YOK",
        "license": "CC BY 4.0 (MDPI)",
        "download_method": "manual_drive_copy",
        "resume_supported": False,
        "files": [
            {"name": "CapERA_DATASET_train.json", "present_locally": (capera_dir / "CapERA_DATASET_train.json").exists()},
            {"name": "CapERA_DATASET_test.json", "present_locally": (capera_dir / "CapERA_DATASET_test.json").exists()},
            {"name": "all_results.json", "present_locally": (capera_dir / "all_results.json").exists(),
             "note": "onceki AGREGATIF sonuclar - ham 2048d embedding DEGIL, notebook 02 hicbir "
                    "sekilde bunu 'mevcut embedding' olarak kabul ETMEMELI"},
        ],
        "raw_video_status": "NOT_PRESENT_LOCALLY - Colab GPU asamasinda Drive'dan saglanmali",
    },
    "msrvtt": {
        "source": "HuggingFace friedrichor/MSR-VTT (1k-A test split, JSFusion protokolu)",
        "license": "akademik (MSR sartlari)",
        "download_method": "huggingface datasets.load_dataset (onceki is paketinde zaten yapildi)",
        "resume_supported": False,
        "files": [
            {"name": "msrvtt_test_1k.json", "present_locally": (msrvtt_dir / "msrvtt_test_1k.json").exists()},
            {"name": "MSRVTT_Videos.zip", "present_locally": (msrvtt_dir / "MSRVTT_Videos.zip").exists()},
        ],
    },
    "visdrone": {
        "source": "VisDrone2019-MOT-train (AISKYEYE/Tianjin University, resmi Task 4 Google Drive) - "
                 "bu depoda onceki is paketinde indirilip SHA-256 dogrulandi (bkz. TASKS.md); "
                 "ingest/01_frames_to_video.py ile 19-sekans bench subset'i (config.yaml: bench.subset) "
                 "mp4'e donusturuldu - Colab'a bu donusturulmus haliyle Drive uzerinden TASINMALI, "
                 "bu depoda data/raw/ gitignore'lu",
        "license": "CC BY-NC-SA 3.0 (AISKYEYE/Tianjin University aiskyeye.com)",
        "download_method": "manual_drive_copy",
        "resume_supported": False,
        "files": [
            {"name": "manifest.json", "present_locally": pathlib.Path("data/raw/manifest.json").exists()},
            {"name": "videos/ (19 mp4, bench subset)", "present_locally": pathlib.Path("data/raw/videos").exists()},
            {"name": "annotations/ (MOT .txt)",
             "present_locally": pathlib.Path("data/raw/VisDrone2019-MOT-train/annotations").exists()},
        ],
        "raw_video_status": "NOT_PRESENT_IN_ZIP - data/raw/ gitignore'lu, Colab GPU asamasinda Drive'dan saglanmali",
    },
}
dl_manifest_path = OUT / "dataset_download_manifest.json"
dl_manifest_path.write_text(json.dumps(dataset_download_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(dataset_download_manifest, indent=2, ensure_ascii=False))
print(f"\\n-> {dl_manifest_path}")
"""),
]

if __name__ == "__main__":
    out_path = pathlib.Path("notebooks/01_auair_download_and_validation.ipynb")
    nb = build_and_execute(CELLS, out_path, timeout=600)
    print(f"\\nNotebook yazildi: {out_path} ({len(nb.cells)} hucre)")
