"""Resmi VisDrone-MOT train setini indirir, dogrular ve data/raw'a acar."""
import hashlib
import pathlib
import zipfile


FILE_ID = "1-qX2d-P1Xr64ke6nTdlm33om1VxCUTSh"
URL = f"https://drive.google.com/uc?id={FILE_ID}"
EXPECTED_BYTES = 8_080_572_990
EXPECTED_SHA256 = "566d08fb53fff4e539f386f5a408ccf17854fd53814dc756bdede2de1dbb4014"
DATASET_NAME = "VisDrone2019-MOT-train"
EXPECTED_SEQUENCES = 56
EXPECTED_FRAMES = 24_201


def file_sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archive(path, expected_bytes=EXPECTED_BYTES,
                     expected_sha256=EXPECTED_SHA256):
    path = pathlib.Path(path)
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"ZIP boyutu yanlis: {path.stat().st_size} != {expected_bytes}"
        )
    if expected_sha256 is not None:
        actual = file_sha256(path)
        if actual != expected_sha256:
            raise RuntimeError(f"ZIP SHA-256 yanlis: {actual}")

    required = {
        f"{DATASET_NAME}/sequences/": False,
        f"{DATASET_NAME}/annotations/": False,
    }
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            member = pathlib.PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise RuntimeError(f"guvensiz ZIP yolu: {info.filename}")
            for prefix in required:
                if info.filename.startswith(prefix):
                    required[prefix] = True
        if not all(required.values()):
            raise RuntimeError(f"ZIP veri sozlesmesi eksik: {required}")
        return len(archive.infolist())


def validate_dataset(root):
    root = pathlib.Path(root)
    sequences = sorted((root / "sequences").glob("*/"))
    annotations = sorted((root / "annotations").glob("*.txt"))
    frames = sum(1 for _ in (root / "sequences").glob("*/*.jpg"))
    seq_names = {path.name for path in sequences}
    ann_names = {path.stem for path in annotations}
    if (len(sequences), len(annotations), frames) != (
            EXPECTED_SEQUENCES, EXPECTED_SEQUENCES, EXPECTED_FRAMES):
        raise RuntimeError(
            "veri sayilari yanlis: "
            f"sequences={len(sequences)} annotations={len(annotations)} "
            f"frames={frames}"
        )
    if seq_names != ann_names:
        raise RuntimeError("sekans ve annotation adlari eslesmiyor")
    return len(sequences), len(annotations), frames


def main():
    repo = pathlib.Path(__file__).resolve().parents[1]
    dataset = repo / "data" / "raw" / DATASET_NAME
    if dataset.exists():
        counts = validate_dataset(dataset)
        print(f"veri hazir: sequences={counts[0]} annotations={counts[1]} "
              f"frames={counts[2]}")
        return

    downloads = repo / "data" / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive_path = downloads / f"{DATASET_NAME}.zip"
    if not archive_path.exists():
        import gdown
        result = gdown.download(URL, str(archive_path), quiet=False, resume=True)
        if not result:
            raise RuntimeError("Google Drive indirmesi tamamlanamadi")

    entries = validate_archive(archive_path)
    print(f"ZIP dogrulandi: entries={entries} sha256={EXPECTED_SHA256}")
    raw_dir = repo / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(raw_dir)
    counts = validate_dataset(dataset)
    print(f"veri hazir: sequences={counts[0]} annotations={counts[1]} "
          f"frames={counts[2]}")


if __name__ == "__main__":
    main()
