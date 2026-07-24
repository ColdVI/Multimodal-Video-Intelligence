"""VisDrone kare dizilerini mp4'e cevirir.
Girdi: data/raw/VisDrone2019-MOT-train/sequences/<seq>/*.jpg
Cikti: data/raw/videos/<seq>.mp4 + data/raw/manifest.json"""
import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_config

FPS = 25  # VisDrone tipik kare hizi; sekans bazinda farkliysa duzeltin


def select_sequences(raw: pathlib.Path, requested=None):
    sequences = sorted(p for p in raw.iterdir() if p.is_dir())
    if not requested:
        return sequences

    by_name = {p.name: p for p in sequences}
    missing = sorted(set(requested) - set(by_name))
    if missing:
        raise ValueError(f"sekans bulunamadi: {', '.join(missing)}")
    return [by_name[name] for name in requested]


def resolve_ffmpeg() -> str:
    configured = os.environ.get('FFMPEG_BINARY')
    if configured:
        found = shutil.which(configured)
        if found:
            return found
        path = pathlib.Path(configured)
        if path.is_file():
            return str(path)
        raise RuntimeError(f'FFMPEG_BINARY bulunamadi: {configured}')

    found = shutil.which('ffmpeg')
    if found:
        return found

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            'ffmpeg bulunamadi; requirements.txt ile imageio-ffmpeg kurun'
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def build_ffmpeg_command(ffmpeg, seq, out):
    return [
        ffmpeg, "-y", "-framerate", str(FPS),
        "-i", str(seq / "%07d.jpg"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sequence", action="append", dest="sequences",
        help="yalnizca bu sekansi donustur; birden cok kez verilebilir",
    )
    args = ap.parse_args()

    cfg = load_config()
    raw = pathlib.Path("data/raw/VisDrone2019-MOT-train/sequences")
    out_dir = pathlib.Path(cfg["paths"]["videos_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw.exists():
        print(f"HATA: {raw} yok. VisDrone-MOT'u aiskyeye.com'dan kayitla "
              f"indirip buraya cikarin (README.md 'Manuel veri adimi').")
        raise SystemExit(1)

    ffmpeg = resolve_ffmpeg()
    manifest = {}
    try:
        sequences = select_sequences(raw, args.sequences)
    except ValueError as exc:
        ap.error(str(exc))
    for seq in sequences:
        out = out_dir / f"{seq.name}.mp4"
        subprocess.run(build_ffmpeg_command(ffmpeg, seq, out), check=True)
        n_frames = len(list(seq.glob("*.jpg")))
        manifest[seq.name] = {"fps": FPS, "n_frames": n_frames,
                               "duration_s": n_frames / FPS}

    manifest_path = pathlib.Path(cfg["paths"]["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(manifest_path, "w"), indent=1)
    print(f"{len(manifest)} sekans -> {manifest_path}")


if __name__ == "__main__":
    main()
