"""Kullanilan tum checkpoint'leri (YOLO .pt + HF snapshot dizinleri) tek bir
weights_manifest.json (model ID, revision, SHA-256, boyut) ile weights/
altinda toplar. Air-gapped kuruluma tasinacak paket budur; weights/
.gitignore'da oldugu icin git'e commit edilmez.

Model onceden calistirilip HF cache'ine (.runtime/huggingface) veya repo
koku (*.pt) indirilmis olmalidir - bu script agdan indirme yapmaz, yalnizca
zaten var olan agirliklari paketler."""
import hashlib
import json
import os
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import REPO_ROOT, configure_runtime_environment

HF_MODEL_IDS = [
    "microsoft/xclip-base-patch16-zero-shot",
    "google/siglip2-so400m-patch14-384",
]
YOLO_CHECKPOINTS = [
    "yolo26x.pt",
]


def file_sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_hf_snapshot(hf_home: pathlib.Path, model_id: str):
    cache_name = "models--" + model_id.replace("/", "--")
    snapshots_dir = hf_home / "hub" / cache_name / "snapshots"
    if not snapshots_dir.exists():
        return None
    revisions = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not revisions:
        return None
    revisions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return revisions[0]


def package_hf_model(model_id, snapshot_dir, dest_root):
    revision = snapshot_dir.name
    dest = dest_root / "hf" / model_id.replace("/", "__") / revision
    dest.mkdir(parents=True, exist_ok=True)
    files = []
    total_bytes = 0
    for src_file in sorted(snapshot_dir.rglob("*")):
        if src_file.is_dir():
            continue
        rel = src_file.relative_to(snapshot_dir)
        dest_file = dest / rel
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_file, dest_file)
        size = dest_file.stat().st_size
        files.append({
            "path": str(rel).replace("\\", "/"),
            "bytes": size,
            "sha256": file_sha256(dest_file),
        })
        total_bytes += size
    return {
        "type": "hf",
        "model_id": model_id,
        "revision": revision,
        "dest": str(dest.relative_to(dest_root)).replace("\\", "/"),
        "bytes": total_bytes,
        "files": files,
    }


def package_yolo_checkpoint(name, dest_root):
    src = REPO_ROOT / name
    if not src.exists():
        return None
    dest = dest_root / "yolo" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    return {
        "type": "yolo",
        "model_id": name,
        "revision": None,
        "dest": str(dest.relative_to(dest_root)).replace("\\", "/"),
        "bytes": dest.stat().st_size,
        "sha256": file_sha256(dest),
    }


def main():
    configure_runtime_environment()
    hf_home = pathlib.Path(os.environ["HF_HOME"])
    dest_root = REPO_ROOT / "weights"
    dest_root.mkdir(parents=True, exist_ok=True)

    entries = []
    missing = []

    for model_id in HF_MODEL_IDS:
        snapshot_dir = find_hf_snapshot(hf_home, model_id)
        if snapshot_dir is None:
            missing.append({"type": "hf", "model_id": model_id})
            continue
        entries.append(package_hf_model(model_id, snapshot_dir, dest_root))

    for name in YOLO_CHECKPOINTS:
        entry = package_yolo_checkpoint(name, dest_root)
        if entry is None:
            missing.append({"type": "yolo", "model_id": name})
        else:
            entries.append(entry)

    manifest = {"generated_by": "scripts/package_weights.py", "entries": entries, "missing": missing}
    manifest_path = dest_root / "weights_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    for entry in entries:
        print(f"paketlendi: {entry['model_id']} ({entry.get('revision') or '-'}) "
              f"{entry['bytes'] / 1e6:.1f} MB -> weights/{entry['dest']}")
    for m in missing:
        print(f"EKSIK: {m['type']} {m['model_id']} bulunamadi "
              f"(once modeli calistirip cache'e/repo kokune indirin)")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
