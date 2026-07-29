"""Faz 6 Colab handoff - kosum sonunda SADECE sonuc artifact'lerini Drive'a
kopyalar. Cogu artifact zaten dogrudan Drive'a yaziliyor (colab_paths.
research_root() Drive mount edilmisken Drive'i doner) - bu script'in
GERCEK isi, KASITLI olarak yerel/ephemeral `/content` oturumunda kalan iki
sey icin: (1) `artifacts/research/environment_capability_report.json`
(scripts/colab_preflight.py bunu bilerek repo-yerel yola yazar, SESSION
tanisi oldugu icin), (2) calistirilmis `.ipynb` dosyalarinin kendisi
(notebooks/ klasoru /content'teki gecici repo kopyasinda, Drive'da DEGIL -
oturum kapaninca KAYBOLUR). Veritabani calisma dizinleri (SS5) BURADA DA
kopyalanmaz - zaten /content/vector_bench'te ve gecicidir."""
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.research import colab_paths

LOCAL_ONLY_FILES = [
    pathlib.Path("artifacts/research/environment_capability_report.json"),
]
NOTEBOOKS_DIR = pathlib.Path("notebooks")


def copy_local_only_files(dest_root: pathlib.Path) -> list:
    copied = []
    for src in LOCAL_ONLY_FILES:
        if not src.exists():
            continue
        dest = dest_root / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.resolve() == src.resolve():
            copied.append(f"{dest} (kaynakla ayni - kopyalama atlandi, Drive mount edilmemis)")
            continue
        shutil.copy2(src, dest)
        copied.append(str(dest))
    return copied


def copy_executed_notebooks(dest_root: pathlib.Path) -> list:
    dest_dir = dest_root / "executed_notebooks"
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for nb in sorted(NOTEBOOKS_DIR.glob("*.ipynb")):
        dest = dest_dir / nb.name
        shutil.copy2(nb, dest)
        copied.append(str(dest))
    return copied


def main():
    if not colab_paths.drive_mounted():
        print("[UYARI] Drive mount edilmemis - kopyalama yerel fallback'e "
             f"yapilacak ({colab_paths.research_root()}), Drive'a DEGIL. "
             "Colab'da Drive'i mount edip tekrar calistirin.")
    dest_root = colab_paths.research_root()

    copied_local = copy_local_only_files(dest_root)
    copied_notebooks = copy_executed_notebooks(dest_root)

    print(f"Yerel-sadece dosyalar kopyalandi ({len(copied_local)}):")
    for p in copied_local:
        print(f"  {p}")
    print(f"\nCalistirilmis notebook'lar kopyalandi ({len(copied_notebooks)}):")
    for p in copied_notebooks:
        print(f"  {p}")

    print(f"\nNOT: diger tum sonuc artifact'leri (CSV/Parquet/JSON/MD) zaten "
         f"dogrudan Drive'a yaziliyordu (colab_paths.research_root()={dest_root}) - "
         f"burada AYRICA kopyalanmadi (cift yazma degil, tek kaynak).")


if __name__ == "__main__":
    main()
