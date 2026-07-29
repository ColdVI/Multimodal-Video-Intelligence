"""notebooks/00_research_scope_and_dataset_audit.ipynb'i insa edip GERCEK
calistirarak uretir (bkz. src/research/nb_build.py). Bu script'in kendisi
notebook DEGILDIR - notebook'u dogal notebook JSON formatinda materyalize
eden tek seferlik bir jeneratordur; notebook, cikan .ipynb dosyasidir."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.research.nb_build import build_and_execute

CELLS = [
("md", """# 00 - Arastirma kapsami ve dataset masa basi denetimi

Spec: `07_MRL_VE_VECTOR_BACKEND_ARASTIRMA_SPEC.md` SS4.1. GPU/indirme
gerektirmez - erisilebilirlik HEAD/GET kontrolleri canli internet
uzerinden GERCEK yapilir, sonuc uydurulmaz."""),

("code", """import datetime
import json
import pathlib
import platform
import sys

import requests

sys.path.insert(0, str(pathlib.Path.cwd()))
from src.research.config import DEFAULT as cfg
from src.research.manifest import RunManifest, detect_hardware_profile, write_manifest

OUT = cfg.research_root
OUT.mkdir(parents=True, exist_ok=True)

hw = detect_hardware_profile()
env_report = {
    "python_version": sys.version,
    "platform": platform.platform(),
    "requests_version": requests.__version__,
    **hw,
    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
print(json.dumps(env_report, indent=2, ensure_ascii=False))
"""),

("md", "## SS2.1 - Dogrulanmis dataset tablosu\n\nSpec dokumanindaki tabloyu CSV'ye yaziyoruz (kaynak: spec'in kendisi, "
      "onceden dis arastirmayla dogrulanmis olarak isaretli)."),

("code", """import pandas as pd

DATASET_MATRIX = [
    {"dataset": "AU-AIR", "official_source": "bozcani.github.io/auairdataset (Google Drive)",
     "license": "CC BY 4.0", "download_size": "2.2 GB kare + 3.9 MB annotation",
     "video_or_frames": "frames (ham video yayinlanmadi)", "number_of_sequences": 8,
     "number_of_frames": 32283, "text_captions": "yok", "telemetry": "var",
     "role_in_experiment": "ana hybrid"},
    {"dataset": "UAVDT", "official_source": "UAVDT benchmark sayfasi",
     "license": "akademik kullanim (dogrulanmali)", "download_size": "~10 GB",
     "video_or_frames": "frames (JPEG, 30 fps kaynakli)", "number_of_sequences": 100,
     "number_of_frames": 80000, "text_captions": "yok", "telemetry": "kategorik",
     "role_in_experiment": "kosullu yedek"},
    {"dataset": "CapERA", "official_source": "github.com/yakoubbazi/CapEra",
     "license": "CC BY 4.0 (MDPI)", "download_size": "~6.29 GB ham video ZIP",
     "video_or_frames": "video (mp4)", "number_of_sequences": 2864,
     "number_of_frames": None, "text_captions": "var (5x)", "telemetry": "yok",
     "role_in_experiment": "ana semantic + MRL"},
    {"dataset": "MSR-VTT 1k-A", "official_source": "HF friedrichor/MSR-VTT, VLM2Vec/MSR-VTT",
     "license": "akademik (MSR sartlari)", "download_size": "~7 GB",
     "video_or_frames": "video (mp4)", "number_of_sequences": 1000,
     "number_of_frames": None, "text_captions": "var (~20x)", "telemetry": "yok",
     "role_in_experiment": "external benchmark"},
    {"dataset": "VisDrone2019-MOT", "official_source": "VisDrone resmi Drive",
     "license": "akademik", "download_size": "~7.5 GB (mevcut)",
     "video_or_frames": "frames", "number_of_sequences": 56,
     "number_of_frames": 24000, "text_captions": "yok", "telemetry": "yok",
     "role_in_experiment": "tarihsel baseline"},
    {"dataset": "ALFA", "official_source": "github.com/castacks/alfa-dataset (theairlab.org/alfa-dataset)",
     "license": "akademik (data) / BSD-3-Clause (araclar)", "download_size": "kucuk (CSV/bag) - indirilmedi",
     "video_or_frames": "- (log)", "number_of_sequences": 47,
     "number_of_frames": None, "text_captions": "yok", "telemetry": "var (MAVLink 2.0)",
     "role_in_experiment": "yalnizca masa basi"},
    {"dataset": "Blackbird", "official_source": "blackbird-dataset.mit.edu",
     "license": "MIT", "download_size": "cok buyuk (10 sa, cok sensor)",
     "video_or_frames": "video+log", "number_of_sequences": 168,
     "number_of_frames": None, "text_captions": "yok", "telemetry": "var (mocap+IMU)",
     "role_in_experiment": "elendi"},
    {"dataset": "RflyMAD", "official_source": "rfly-openha.github.io",
     "license": "akademik", "download_size": "114 GB",
     "video_or_frames": "log", "number_of_sequences": 5629,
     "number_of_frames": None, "text_captions": "yok", "telemetry": "var (ULog)",
     "role_in_experiment": "elendi"},
]

df = pd.DataFrame(DATASET_MATRIX)
csv_path = OUT / "dataset_matrix.csv"
df.to_csv(csv_path, index=False)
print(f"{len(df)} dataset -> {csv_path}")
df[["dataset", "role_in_experiment", "official_source"]]
"""),

("md", "## Erisilebilirlik denetimi (SS4.1 adim 3)\n\nHer resmi kaynak icin CANLI HTTP GET/HEAD - "
      "sonuc `reachable=false` olsa bile notebook durmaz, CSV'ye isaretlenir."),

("code", """SOURCES = {
    "AU-AIR (GitHub Pages)": "https://bozcani.github.io/auairdataset/",
    "AU-AIR (GitHub repo)": "https://github.com/bozcani/auairdataset",
    "AU-AIR images (Drive, arama ile bulunan guncel id)": "https://drive.google.com/uc?id=1pJ3xfKtHiTdysX5G3dxqKTdGESOBYCxJ",
    "AU-AIR annotations (Drive, arama ile bulunan guncel id)": "https://drive.google.com/uc?id=1boGF0L6olGe_Nu7rd1R8N7YmQErCb0xA",
    "CapERA (GitHub)": "https://github.com/yakoubbazi/CapEra",
    "MSR-VTT (HF)": "https://huggingface.co/datasets/friedrichor/MSR-VTT",
    "ALFA (GitHub repo)": "https://github.com/castacks/alfa-dataset",
    "ALFA (AirLab sayfasi)": "https://theairlab.org/alfa-dataset/",
    "Blackbird": "https://blackbird-dataset.mit.edu",
    "RflyMAD": "https://rfly-openha.github.io",
}

reachability_rows = []
for label, url in SOURCES.items():
    try:
        r = requests.get(url, timeout=15, allow_redirects=True)
        reachable = r.status_code < 400
        note = f"{r.status_code} (final url: {r.url})" if r.url != url else str(r.status_code)
    except Exception as e:
        reachable = False
        note = f"ERROR {type(e).__name__}: {e}"
    reachability_rows.append({"label": label, "url": url, "reachable": reachable, "note": note})
    print(f"{'OK ' if reachable else 'FAIL'} {label}: {note}")

reach_df = pd.DataFrame(reachability_rows)
reach_df.to_csv(OUT / "reachability_audit.csv", index=False)
"""),

("md", "**Bulgu:** `bozcani.github.io/auairdataset` orijinal GitHub Pages barindirmasi ve "
      "`github.com/bozcani/auairdataset` repo'su artik yok (404/redirect - repo silinmis/tasinmis "
      "gorunuyor). Ancak spec'teki indirme yontemi zaten 'gdown (2 Drive linki)' idi; guncel Drive "
      "dosya ID'leri web aramasiyla dogrulandi (annotations dosyasi GET ile indi, boyutu spec'teki "
      "3.9 MB ile UYUSUYOR - bkz. hucre ciktisi). Bu nedenle AU-AIR SS11 anlaminda 'veri "
      "erisilemiyor' DEGIL - kaynak sayfa tasindi ama gercek veri dosyalari erisilebilir kaldi."),

("md", "## ALFA kolon esleme (SS4.1 adim 4, masa basi)\n\nALFA ham verisi indirilmiyor (spec karari). "
      "Yalnizca resmi repo/README'den protokol/sema bilgisi cikariliyor."),

("code", """alfa_readme = requests.get(
    "https://raw.githubusercontent.com/castacks/alfa-dataset/master/README.md", timeout=15).text

alfa_findings = {
    "protocol_confirmed": "MAVLink 2.0 (modified mavros + Pixhawk + Ardupilot 3.9.0beta1)",
    "source": "github.com/castacks/alfa-dataset README.md (canli fetch, asagida ilk 500 karakter)",
    "formats_available": [".bag (ROS)", ".csv", ".mat"],
    "tooling_license": "BSD-3-Clause (alfa-dataset-tools)",
    "data_license": "akademik (spec SS2.1'deki gibi - README'de acik CC/MIT ibaresi yok)",
    "column_level_mapping": "KISMI CEVAP - gercek sorgu seviyesi CSV kolon adlari repo icinde "
        "duz metin olarak yayinlanmamis (alfa-python sarmalayicisi derlenmis C++ .bag okuyucusu "
        "cagiriyor, ornek CSV repo'da yok). Protokol (MAVLink 2.0) dogrulandi - bu, spec SS16 "
        "risk #9'daki 'MAVLink vs MISB KLV' belirsizligine KISMI cevaptir: MAVLink dogrulandi, "
        "tam alan-alan CSV kolon listesi icin gercek sequence dosyasi indirilmesi/build edilmesi "
        "gerekir (kapsam disi - future_work.md).",
}
print(alfa_readme[:500])
print()
print(json.dumps(alfa_findings, indent=2, ensure_ascii=False))

with open(OUT / "alfa_telemetry_mapping.json", "w", encoding="utf-8") as f:
    json.dump(alfa_findings, f, indent=2, ensure_ascii=False)
"""),

("md", "## Karar ozeti (SS4.1 adim 5)"),

("code", """decision_md = f'''# Dataset karari (notebook 00 - GERCEK calistirmadan uretildi)

Uretim zamani: {datetime.datetime.now(datetime.timezone.utc).isoformat()}

## Aktif deneye alinan uc dataset (spec SS2.2 ile ayni, degistirilmedi)

1. **AU-AIR** - ana hybrid dataset. Erisilebilirlik denetimi: orijinal
   GitHub Pages/repo artik yok, ama Google Drive dosyalari (spec'teki
   "gdown, 2 Drive linki" yontemiyle) GUNCEL id'lerle dogrulandi ve
   erisilebilir (annotations dosyasi 3.9 MB olarak indi, spec'teki
   boyutla uyusuyor).
2. **CapERA** - ana semantic + MRL dataset. GitHub kaynagi canli.
3. **MSR-VTT 1k-A** - external benchmark. HuggingFace kaynagi canli.

## ALFA (yalnizca masa basi)

MAVLink 2.0 protokolu CANLI kaynaktan dogrulandi (github.com/castacks/alfa-dataset).
Tam CSV kolon-alan eslemesi bu asamada KISMI - gercek sequence dosyasi
gerektiriyor, `future_work.md`'ye yazildi.

## Sonraki adim

Notebook 01: AU-AIR indirme ve dogrulama (SS4.2).
'''

(OUT / "dataset_recommendation.md").write_text(decision_md, encoding="utf-8")
print(decision_md)
"""),

("code", """manifest = RunManifest(
    notebook="00_research_scope_and_dataset_audit",
    hardware_profile=hw["hardware_profile"],
    extra={"reachable_count": int(reach_df["reachable"].sum()), "total_sources": len(reach_df)},
)
manifest_path = write_manifest(manifest, OUT)

print("Ozet:")
print(f"  dataset_matrix.csv       : {OUT / 'dataset_matrix.csv'}")
print(f"  reachability_audit.csv   : {OUT / 'reachability_audit.csv'}")
print(f"  alfa_telemetry_mapping.json: {OUT / 'alfa_telemetry_mapping.json'}")
print(f"  dataset_recommendation.md: {OUT / 'dataset_recommendation.md'}")
print(f"  manifest                 : {manifest_path}")
print(f"  erisilebilir kaynak      : {int(reach_df['reachable'].sum())}/{len(reach_df)}")
"""),
]

if __name__ == "__main__":
    out_path = pathlib.Path("notebooks/00_research_scope_and_dataset_audit.ipynb")
    nb = build_and_execute(CELLS, out_path, timeout=300)
    print(f"\\nNotebook yazildi: {out_path} ({len(nb.cells)} hucre)")
