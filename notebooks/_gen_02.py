"""notebooks/02_qwen2b_embedding_and_mrl.ipynb'i insa eder. Spec SS4.3 +
Colab handoff talebi: GERCEK Qwen3-VL-Embedding-2B uretim kodu, Drive'a
checkpoint/resume, 1024/512/256 MRL turetme. Bu makinede (GT 1030,
CPU-only torch) calistirilmiyor - kullanicinin acik talimati: 'GT 1030
uzerinde Qwen calistirmayi tamamen birak'. Notebook GERCEK kod icerir ve
buradan (nbclient ile) calistirilir ama HER agir hucre `gpu_available`
bayragiyla korunur - GPU yoksa (bu makine) hucre GERCEKTEN atlanir ve
bunu ACIKCA yazar, sahte sonuc URETMEZ. Colab'da (GPU var) ayni kod
gercek embedding'i uretir."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.research.nb_build import build_and_execute
from src.research.colab_paths import COLAB_BOOTSTRAP_CELL

CELLS = [
("code", COLAB_BOOTSTRAP_CELL),
("md", """# 02 - Qwen3-VL-Embedding-2B embedding uretimi + MRL turetme (GPU asamasi)

Spec SS4.3 + Colab handoff. **Bu notebook Colab GPU runtime'inda
calistirilmak uzere yazildi.** Her agir hucre `gpu_available` kontrolu
tasir - GPU yoksa (bu deponun kendi CI/dev makinesi gibi) hucre GERCEKTEN
atlanir, uydurma sonuc YAZILMAZ. Checkpoint'ler `colab_paths.checkpoints_root()`
(Drive mount edilmisse Drive'da) altina her `cfg.embedding_checkpoint_every`
(varsayilan 100) item'da bir yazilir; resume otomatik (zaten tamamlanmis
item'lar atlanir)."""),

("code", """import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path.cwd()))
from src.research import colab_paths
from src.research.config import DEFAULT as cfg
from src.research.manifest import RunManifest, detect_hardware_profile, write_manifest

OUT = colab_paths.research_root()
CKPT_ROOT = colab_paths.checkpoints_root()
EMB_ROOT = colab_paths.embeddings_root()
hw = detect_hardware_profile()
print(json.dumps(hw, indent=2, ensure_ascii=False))
print(f"OUT={OUT}  CKPT_ROOT={CKPT_ROOT}  drive_mounted={colab_paths.drive_mounted()}  in_colab={colab_paths.in_colab()}")
"""),

("md", "## GPU kapisi (gercek kontrol - bench/gpu_gate.py)"),

("code", """sys.path.insert(0, str(pathlib.Path.cwd()))
from bench.gpu_gate import QWEN_CPU_S_PER_WINDOW, QWEN_GPU_S_PER_WINDOW, require_gpu_for_qwen_windows

gpu_available = False
try:
    require_gpu_for_qwen_windows("notebook 02: Qwen3-VL-Embedding-2B 2048d uretimi", n_windows=1)
    gpu_available = True
    print("[GECTI] GPU mevcut - asagidaki tum uretim hucreleri GERCEKTEN calisacak.")
except SystemExit as e:
    print("[ATLANDI] " + str(e))
    print("\\nBu oturumda GPU yok - asagidaki hucreler GERCEK isi ATLAYACAK ve bunu "
         "acikca yazacak (sahte/kismi sonuc URETMEYECEK). Colab GPU runtime'inda "
         "ayni kod gpu_available=True ile gercek embedding uretir.")
"""),

("md", "## Model yukleme (yalniz GPU varsa)"),

("code", """embedder = None
if gpu_available:
    from models import get_embedder
    embedder = get_embedder("qwen3vl_emb_2048")
    print("Qwen3-VL-Embedding-2B yuklendi.")
else:
    print("[ATLANDI] Model yuklenmedi (GPU yok).")
"""),

("md", "## AU-AIR embedding (notebook 01'in GERCEK pencerelerinden)\n\n"
      "`images.zip`'i ELLE ACMANIZA GEREK YOK - asagidaki hucre Drive'daki "
      "`datasets/auair/images.zip`'i (notebook 01'in indirdigi) OTOMATIK bulup "
      "`/content`'e acar (Drive'a degil - hizli)."),

("code", """import zipfile

import cv2
import numpy as np
import pandas as pd

from src.research.embedding_checkpoint import CheckpointWriter, remaining_items

auair_seg_path = OUT / "auair_segments.parquet"
auair_ckpt = CKPT_ROOT / "auair_qwen2048.ndjson"
AUAIR_IMAGES_DIR = pathlib.Path("/content/auair_images")
AUAIR_IMAGES_ZIP = colab_paths.dataset_root("auair") / "images.zip"

def _ensure_auair_images_extracted():
    if AUAIR_IMAGES_DIR.exists() and any(AUAIR_IMAGES_DIR.rglob("*.jpg")):
        print(f"AU-AIR resimleri zaten acik: {AUAIR_IMAGES_DIR}")
        return True
    if not AUAIR_IMAGES_ZIP.exists():
        print(f"[ATLANDI] {AUAIR_IMAGES_ZIP} Drive'da yok - AU-AIR embedding atlaniyor "
             "(once notebook 01'i GPU runtime'inda calistirip images.zip'i indirtin).")
        return False
    print(f"AU-AIR resimleri aciliyor: {AUAIR_IMAGES_ZIP} -> {AUAIR_IMAGES_DIR} "
         "(2.2 GB, birkac dakika surebilir) ...")
    with zipfile.ZipFile(AUAIR_IMAGES_ZIP) as z:
        z.extractall(AUAIR_IMAGES_DIR)
    print("Acildi.")
    return True

auair_embedding_ready = False
if not auair_seg_path.exists():
    print(f"[ATLANDI] {auair_seg_path} yok - once notebook 01'i calistirin.")
elif not gpu_available:
    print("[ATLANDI] AU-AIR embedding uretimi (GPU yok).")
elif not _ensure_auair_images_extracted():
    pass
else:
    # zip'in ic klasor yapisi onceden bilinmiyor (AU-AIR'in kendi paketlemesi) -
    # dosya adina gore rglob ile bulup harita cikariyoruz, sabit bir alt-yol VARSAYMIYORUZ.
    _auair_image_index = {p.name: p for p in AUAIR_IMAGES_DIR.rglob("*") if p.is_file()}
    print(f"AU-AIR resim indeksi: {len(_auair_image_index)} dosya.")

    seg_df = pd.read_parquet(auair_seg_path)
    all_ids = seg_df["segment_id"].tolist()
    todo = remaining_items(all_ids, auair_ckpt)
    print(f"AU-AIR: {len(all_ids)} pencere, {len(all_ids) - len(todo)} zaten tamamlanmis (resume), "
         f"{len(todo)} kaldi.")
    seg_by_id = seg_df.set_index("segment_id")
    n_missing_frames = 0
    with CheckpointWriter(auair_ckpt, flush_every=cfg.embedding_checkpoint_every) as writer:
        for i, seg_id in enumerate(todo):
            row = seg_by_id.loc[seg_id]
            frames = []
            for name in row["frame_names"]:
                p = _auair_image_index.get(name)
                if p is None:
                    continue
                img = cv2.imread(str(p))
                if img is not None:
                    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not frames:
                n_missing_frames += 1
                continue
            vec = embedder.embed_video(frames)
            writer.add(seg_id, vec.tolist())
            if (i + 1) % cfg.embedding_checkpoint_every == 0:
                print(f"  AU-AIR {i+1}/{len(todo)} ({writer.n_flushed} Drive'a yazildi, "
                     f"{n_missing_frames} pencere kare bulamadi)")
    if n_missing_frames:
        print(f"UYARI: {n_missing_frames} AU-AIR penceresi icin hic kare bulunamadi "
             "(image_name/zip icerigi uyusmuyor olabilir).")
    auair_embedding_ready = len(remaining_items(all_ids, auair_ckpt)) == 0
    print(f"AU-AIR embedding_ready={auair_embedding_ready} -> {auair_ckpt}")
"""),

("md", "## CapERA embedding (dataset_adapters.capera.CapERAAdapter - ham video Drive'dan)\n\n"
      "ERA_Dataset.zip'i ELLE ACMANIZA GEREK YOK - asagidaki hucre, "
      "`/content/ERA_Dataset/Videos` yoksa Drive'daki `datasets/capera/ERA_Dataset.zip`'i "
      "OTOMATIK bulup `/content`'e acar (Drive'a degil - hizli). Zip'i Drive'da "
      "`.../phase6_mrl_vector_backend/datasets/capera/ERA_Dataset.zip` yoluna atmaniz yeterli. "
      "AYRICA `CapERA_DATASET_train.json` ve `CapERA_DATASET_test.json` (kucuk caption "
      "dosyalari, config.yaml'daki train_split/test_split) da AYNI `datasets/capera/` "
      "klasorune atin - bu depoda `data/downloads/` altinda gitignore'lu oldugu icin ZIP'in "
      "icinde GELMEZ, Drive'a ayrica koymaniz gerekir."),

("code", """import zipfile

from common import load_config as _load_raw_config
from dataset_adapters.capera import CapERAAdapter, FIXED_DURATION_S
from ingest.frame_io import read_window_frames

capera_ckpt = CKPT_ROOT / "capera_qwen2048.ndjson"
capera_embedding_ready = False

CAPERA_VIDEOS_DIR = pathlib.Path("/content/ERA_Dataset/Videos")
CAPERA_DIR_IN_DRIVE = colab_paths.dataset_root("capera")
CAPERA_ZIP_IN_DRIVE = CAPERA_DIR_IN_DRIVE / "ERA_Dataset.zip"

def _ensure_capera_extracted():
    if CAPERA_VIDEOS_DIR.exists():
        print(f"CapERA video zaten acik: {CAPERA_VIDEOS_DIR}")
        return True
    if not CAPERA_ZIP_IN_DRIVE.exists():
        print(f"[ATLANDI] {CAPERA_ZIP_IN_DRIVE} Drive'da yok - CapERA video "
             "atlaniyor (AU-AIR/MSR-VTT etkilenmez).")
        return False
    print(f"CapERA video aciliyor: {CAPERA_ZIP_IN_DRIVE} -> /content/ERA_Dataset ...")
    with zipfile.ZipFile(CAPERA_ZIP_IN_DRIVE) as z:
        z.extractall("/content/ERA_Dataset")
    print("Acildi.")
    return CAPERA_VIDEOS_DIR.exists()

# NOT: config.yaml'daki train_split/test_split yerel gelistirme yoluna
# (data/downloads/capera/...) sabit - o klasor .gitignore'da, Colab'a hic
# gitmez. MSR-VTT hucresiyle AYNI desen: config'e degil, dogrudan
# colab_paths.dataset_root()'a (Drive) bakiyoruz.
capera_train = CAPERA_DIR_IN_DRIVE / "CapERA_DATASET_train.json"
capera_test = CAPERA_DIR_IN_DRIVE / "CapERA_DATASET_test.json"

capera_cfg = None
if not gpu_available:
    print("[ATLANDI] CapERA embedding uretimi (GPU yok).")
elif not _ensure_capera_extracted():
    pass
elif not (capera_train.exists() and capera_test.exists()):
    print(f"[ATLANDI] {capera_train} / {capera_test} Drive'da yok - CapERA "
         "caption JSON'larini ERA_Dataset.zip'in yanina (datasets/capera/) atin.")
else:
    capera_cfg = _load_raw_config()
    capera_cfg["datasets"]["capera"]["train_split"] = str(capera_train)
    capera_cfg["datasets"]["capera"]["test_split"] = str(capera_test)

if capera_cfg is not None:
    adapter = CapERAAdapter(cfg=capera_cfg)
    all_ids = adapter.list_sequences()
    todo = remaining_items(all_ids, capera_ckpt)
    print(f"CapERA: {len(all_ids)} video, {len(all_ids) - len(todo)} zaten tamamlanmis, {len(todo)} kaldi.")
    n_missing_video = 0
    with CheckpointWriter(capera_ckpt, flush_every=cfg.embedding_checkpoint_every) as writer:
        for i, seq_id in enumerate(todo):
            video_path = adapter.load_video(seq_id)
            if not video_path.exists():
                n_missing_video += 1
                continue
            frames = read_window_frames(str(video_path), 0.0, FIXED_DURATION_S, n=cfg.frames_per_item)
            if not frames:
                continue
            vec = embedder.embed_video(frames)
            writer.add(seq_id, vec.tolist())
            if (i + 1) % cfg.embedding_checkpoint_every == 0:
                print(f"  CapERA {i+1}/{len(todo)} ({writer.n_flushed} Drive'a yazildi, {n_missing_video} video eksik)")
    if n_missing_video:
        print(f"UYARI: {n_missing_video} CapERA videosu Drive'da bulunamadi - "
             "scripts/verify_drive_inputs.py ile onceden kontrol edin.")
    capera_embedding_ready = len(remaining_items(all_ids, capera_ckpt)) == 0
    print(f"CapERA embedding_ready={capera_embedding_ready} -> {capera_ckpt}")
"""),

("md", "## MSR-VTT embedding (scripts/validate_msrvtt.py::embed_all_videos - mevcut, per-item cache)"),

("code", """msrvtt_ckpt = CKPT_ROOT / "msrvtt_qwen2048.ndjson"
msrvtt_embedding_ready = False

if not gpu_available:
    print("[ATLANDI] MSR-VTT embedding uretimi (GPU yok).")
else:
    from scripts.validate_msrvtt import embed_all_videos, load_test_split

    msrvtt_dir = colab_paths.dataset_root("msrvtt")
    test_split_path = msrvtt_dir / "msrvtt_test_1k.json"
    MSRVTT_LOCAL_VIDEOS_DIR = pathlib.Path("/content/msrvtt_videos")
    MSRVTT_VIDEOS_ZIP = msrvtt_dir / "videos.zip"

    def _resolve_msrvtt_videos_dir():
        # Drive'da dogrudan tek tek mp4 dosyalari varsa (kucuk sayida, elle
        # yuklendiyse) onu kullan; yoksa videos.zip'i (varsa) /content'e ac -
        # zip'in ic klasor yapisi bilinmiyor, en cok .mp4 iceren alt-klasoru sec.
        drive_videos_dir = msrvtt_dir / "videos"
        if drive_videos_dir.exists() and any(drive_videos_dir.glob("*.mp4")):
            return drive_videos_dir
        if not MSRVTT_VIDEOS_ZIP.exists():
            return None
        if not (MSRVTT_LOCAL_VIDEOS_DIR.exists() and any(MSRVTT_LOCAL_VIDEOS_DIR.rglob("*.mp4"))):
            print(f"MSR-VTT videolari aciliyor: {MSRVTT_VIDEOS_ZIP} -> {MSRVTT_LOCAL_VIDEOS_DIR} ...")
            with zipfile.ZipFile(MSRVTT_VIDEOS_ZIP) as z:
                z.extractall(MSRVTT_LOCAL_VIDEOS_DIR)
            print("Acildi.")
        counts = {}
        for p in MSRVTT_LOCAL_VIDEOS_DIR.rglob("*.mp4"):
            counts[p.parent] = counts.get(p.parent, 0) + 1
        return max(counts, key=counts.get) if counts else None

    if not test_split_path.exists():
        print(f"[ATLANDI] {test_split_path} yok.")
    else:
        entries = load_test_split(str(test_split_path))
        videos_dir = _resolve_msrvtt_videos_dir()
        if videos_dir is None:
            print(f"[ATLANDI] {msrvtt_dir}/videos/*.mp4 veya {MSRVTT_VIDEOS_ZIP} Drive'da yok.")
        else:
            embeddings = embed_all_videos(entries, videos_dir, "qwen3vl_emb_2048",
                                          n_frames=32, cache_file=msrvtt_ckpt)
            msrvtt_embedding_ready = len(embeddings) == len(entries)
            print(f"MSR-VTT: {len(embeddings)}/{len(entries)} video embed edildi "
                 f"(embedding_ready={msrvtt_embedding_ready}) -> {msrvtt_ckpt}")
            if not msrvtt_embedding_ready:
                print(f"UYARI: {len(entries) - len(embeddings)} MSR-VTT videosu "
                     f"{videos_dir} altinda bulunamadi (embed_all_videos sessizce atlar).")
"""),

("md", "## VisDrone-MOT embedding (bench subset - ingest/02_windowing.py ile AYNI pencereleme formulu)\n\n"
      "Ham video/annotasyon bu depoda YOK (`data/raw/` gitignore'lu) - Drive'da "
      "`datasets/visdrone/` altina `manifest.json`, `videos/*.mp4` (ingest/01_frames_to_video.py "
      "ciktisi, 19 sekans bench subset - config.yaml: bench.subset), `annotations/*.txt` "
      "KULLANICI TARAFINDAN yerlestirilmeli. Yoksa hucre sadece atlanir."),

("code", """from common import load_config as _load_raw_config
from ingest.frame_io import read_window_frames

visdrone_ckpt = CKPT_ROOT / "visdrone_qwen2048.ndjson"
visdrone_embedding_ready = False

VISDRONE_DIR = colab_paths.dataset_root("visdrone")
visdrone_manifest_path = VISDRONE_DIR / "manifest.json"

if not gpu_available:
    print("[ATLANDI] VisDrone embedding uretimi (GPU yok).")
elif not visdrone_manifest_path.exists():
    print(f"[ATLANDI] {visdrone_manifest_path} Drive'da yok - VisDrone atlaniyor.")
else:
    visdrone_manifest = json.loads(visdrone_manifest_path.read_text(encoding="utf-8"))

    # ingest/02_windowing.py ile AYNI formul. window_size_s/stride_s research
    # cfg'de config.yaml'daki degerlerle AYNI (kilitli sabit) - min_window_s
    # ResearchConfig'te yok, ham config.yaml'dan okunuyor.
    _min_win = _load_raw_config()["window"]["min_window_s"]
    visdrone_windows = []
    for vid, m in visdrone_manifest.items():
        t = 0.0
        while t < m["duration_s"]:
            t_end = min(t + cfg.window_size_s, m["duration_s"])
            if t_end - t >= _min_win:
                visdrone_windows.append({"video_id": vid, "t_start": round(t, 2), "t_end": round(t_end, 2)})
            t += cfg.stride_s

    def _visdrone_window_id(w):
        return f"{w['video_id']}__{w['t_start']:.2f}_{w['t_end']:.2f}"

    windows_by_id = {_visdrone_window_id(w): w for w in visdrone_windows}
    all_ids = list(windows_by_id)
    todo = remaining_items(all_ids, visdrone_ckpt)
    print(f"VisDrone: {len(visdrone_manifest)} video, {len(all_ids)} pencere, "
         f"{len(all_ids) - len(todo)} zaten tamamlanmis, {len(todo)} kaldi.")
    n_missing_video = 0
    with CheckpointWriter(visdrone_ckpt, flush_every=cfg.embedding_checkpoint_every) as writer:
        for i, win_id in enumerate(todo):
            w = windows_by_id[win_id]
            video_path = VISDRONE_DIR / "videos" / f"{w['video_id']}.mp4"
            if not video_path.exists():
                n_missing_video += 1
                continue
            frames = read_window_frames(str(video_path), w["t_start"], w["t_end"], n=cfg.frames_per_item)
            if not frames:
                continue
            vec = embedder.embed_video(frames)
            writer.add(win_id, vec.tolist())
            if (i + 1) % cfg.embedding_checkpoint_every == 0:
                print(f"  VisDrone {i+1}/{len(todo)} ({writer.n_flushed} Drive'a yazildi, {n_missing_video} video eksik)")
    if n_missing_video:
        print(f"UYARI: {n_missing_video} VisDrone videosu Drive'da bulunamadi.")
    visdrone_embedding_ready = len(remaining_items(all_ids, visdrone_ckpt)) == 0
    print(f"VisDrone embedding_ready={visdrone_embedding_ready} -> {visdrone_ckpt}")
"""),

("md", "## MRL turetme (SS3.3) - GPU GEREKTIRMEZ, mevcut 2048d checkpoint'lerden herhangi biri varsa calisir"),

("code", """from src.research.mrl import derive_all_dims, validate_mrl_derivation
from src.research.embedding_checkpoint import load_cached

MRL_DIMS = (1024, 512, 256)
mrl_summary = {}

for dataset_id, ckpt_path in [("auair", auair_ckpt), ("capera", capera_ckpt), ("msrvtt", msrvtt_ckpt),
                              ("visdrone", visdrone_ckpt)]:
    if not ckpt_path.exists():
        mrl_summary[dataset_id] = {"status": "ATLANDI - 2048d checkpoint yok"}
        print(f"{dataset_id}: [ATLANDI] {ckpt_path} yok.")
        continue
    e2048_by_id = load_cached(ckpt_path)
    if not e2048_by_id:
        mrl_summary[dataset_id] = {"status": "ATLANDI - checkpoint bos"}
        continue
    problems_total = 0
    derived_out = {d: {} for d in MRL_DIMS}
    for item_id, e2048 in e2048_by_id.items():
        derived = derive_all_dims(e2048, MRL_DIMS)
        problems = validate_mrl_derivation(e2048, derived)
        problems_total += len(problems)
        for d in MRL_DIMS:
            derived_out[d][item_id] = derived[d]
    for d in MRL_DIMS:
        out_path = EMB_ROOT / f"{dataset_id}_qwen{d}.json"
        out_path.write_text(json.dumps(derived_out[d], ensure_ascii=False), encoding="utf-8")
    mrl_summary[dataset_id] = {"status": "TAMAMLANDI", "n_items": len(e2048_by_id),
                               "n_validation_problems": problems_total,
                               "dims_written": list(MRL_DIMS)}
    print(f"{dataset_id}: {len(e2048_by_id)} item, {problems_total} dogrulama sorunu, "
         f"1024/512/256 -> {EMB_ROOT}")

print()
print(json.dumps(mrl_summary, indent=2, ensure_ascii=False))
"""),

("md", "## Ozet ve manifest"),

("code", """embedding_status = {
    "auair": auair_embedding_ready,
    "capera": capera_embedding_ready,
    "msrvtt": msrvtt_embedding_ready,
    "visdrone": visdrone_embedding_ready,
}
report_md = f'''# Notebook 02 - Qwen3-VL-Embedding-2B GPU asamasi sonucu

## GPU durumu
gpu_available={gpu_available} (hardware_profile={hw["hardware_profile"]})

## embedding_ready bayraklari (dataset basina, birbirinden BAGIMSIZ)
{json.dumps(embedding_status, indent=2, ensure_ascii=False)}

## MRL turetme ozeti
{json.dumps(mrl_summary, indent=2, ensure_ascii=False)}

## Checkpoint yollari (resume destekli - hucreyi tekrar calistirmak kaldigi yerden devam eder)
- AU-AIR: {auair_ckpt}
- CapERA: {capera_ckpt}
- MSR-VTT: {msrvtt_ckpt}
- VisDrone: {visdrone_ckpt}
'''
(OUT / "qwen2b_mrl_report.md").write_text(report_md, encoding="utf-8")
print(report_md)

manifest = RunManifest(
    notebook="02_qwen2b_embedding_and_mrl",
    hardware_profile=hw["hardware_profile"],
    extra={
        "gpu_available": gpu_available,
        "embedding_ready": embedding_status,
        "mrl_summary": mrl_summary,
        "checkpoint_paths": {"auair": str(auair_ckpt), "capera": str(capera_ckpt), "msrvtt": str(msrvtt_ckpt),
                            "visdrone": str(visdrone_ckpt)},
    },
)
manifest_path = write_manifest(manifest, OUT)
print(f"\\nmanifest -> {manifest_path}")
"""),
]

if __name__ == "__main__":
    out_path = pathlib.Path("notebooks/02_qwen2b_embedding_and_mrl.ipynb")
    nb = build_and_execute(CELLS, out_path, timeout=300)
    print(f"\\nNotebook yazildi: {out_path} ({len(nb.cells)} hucre)")
