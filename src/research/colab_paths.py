"""Faz 6 Colab handoff: TEK Drive kok config degiskeni + yerel/Colab ikili
yol cozumu. Butun notebooklar ve script'ler dosya yollarini BURADAN okur -
Drive kokunu degistirmek icin tek yer burasi.

Iki calisma modu:
- Colab (gercek calistirma): /content/drive/MyDrive/... mount edilmis -
  DRIVE_ROOT gercekten var, buyuk/kalici dosyalar oraya yazilir.
- Yerel gelistirme/test (bu depo, GPU'suz makine): DRIVE_ROOT'un ebeveyni
  yok - artifacts/research/ altina duser (mevcut testler ve onceki
  notebook 00-01-03-06 kosumlari bu fallback'le calisti, kirilmadi).

Veritabani calisma dizinleri ASLA Drive'da tutulmaz (spec madde 5) -
LOCAL_SCRATCH_ROOT hep yerel/ephemeral diskte kalir (Colab'da /content/
vector_bench, yerelde scratch/vector_bench)."""
import pathlib

DRIVE_ROOT = pathlib.Path("/content/drive/MyDrive/VidEmbedd/phase6_mrl_vector_backend")
LOCAL_SCRATCH_ROOT_COLAB = pathlib.Path("/content/vector_bench")

# Kod/repo'nun (bu ZIP'in) Drive'da CIKARILMASI beklenen SABIT konum -
# DRIVE_ROOT (VERI kokü) ile KARISTIRILMAMALI. COLAB_RUNBOOK.md kullaniciya
# ZIP'i tam olarak buraya cikarmasini soyler; her notebook'un bootstrap
# hucresi calisma dizinini buna gore ayarlar.
REPO_EXTRACT_PATH = pathlib.Path("/content/drive/MyDrive/VidEmbedd/phase6_repo")

# Her notebook'un ILK hucresi budur (spec/kullanici madde 13: dosya yolu
# duzenleme, dependency cozme kullaniciya BIRAKILMAZ). Colab disinda
# (google.colab yok) sessizce atlar - bu depodaki mevcut calisma dizinini
# repo koku varsayar, boylece yerel test/gelistirme de calismaya devam eder.
COLAB_BOOTSTRAP_CELL = f'''import subprocess
import sys

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    REPO_ROOT = "{REPO_EXTRACT_PATH.as_posix()}"
    import os
    if not os.path.isdir(REPO_ROOT):
        raise FileNotFoundError(
            f"{{REPO_ROOT}} yok - COLAB_RUNBOOK.md'ye gore ZIP'i once bu klasore cikarin.")
    os.chdir(REPO_ROOT)
    sys.path.insert(0, REPO_ROOT)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"],
                   check=False)
    print(f"[Colab bootstrap] repo kok: {{REPO_ROOT}} - calisma dizini ayarlandi, "
         "bagimliliklar kuruldu.")
except ImportError:
    print("[Colab bootstrap] google.colab yok - Colab-disi ortam, ATLANDI "
         "(mevcut calisma dizini repo koku varsayiliyor).")
'''

_LOCAL_FALLBACK_RESEARCH_ROOT = pathlib.Path("artifacts/research")
_LOCAL_FALLBACK_SCRATCH_ROOT = pathlib.Path("scratch/vector_bench")


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def drive_mounted() -> bool:
    """DRIVE_ROOT'un ebeveyni (.../VidEmbedd) erisilebilir mi - Drive
    gercekten mount edilmis mi kontrolu. Colab DISINDA hep False."""
    return DRIVE_ROOT.parent.exists()


def research_root() -> pathlib.Path:
    """Buyuk/kalici dosyalarin (raw dataset, parquet, embedding checkpoint,
    manifest, sonuc CSV/Parquet/Markdown) yazildigi kok. Drive mount
    edilmisse Drive'da, degilse yerel fallback'te (test/gelistirme)."""
    root = DRIVE_ROOT if drive_mounted() else _LOCAL_FALLBACK_RESEARCH_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def local_scratch_root() -> pathlib.Path:
    """Veritabani calisma dizinleri (ClickHouse/Qdrant/pgvector data/) -
    HICBIR ZAMAN Drive'a yazilmaz, hep yerel/ephemeral disktedir."""
    root = LOCAL_SCRATCH_ROOT_COLAB if in_colab() else _LOCAL_FALLBACK_SCRATCH_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def dataset_root(dataset_id: str) -> pathlib.Path:
    root = research_root() / "datasets" / dataset_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def embeddings_root() -> pathlib.Path:
    root = research_root() / "embeddings"
    root.mkdir(parents=True, exist_ok=True)
    return root


def checkpoints_root() -> pathlib.Path:
    root = research_root() / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    return root


def results_root() -> pathlib.Path:
    """Notebook 03-06'nin bitmis sonuc CSV/Parquet/Markdown dosyalari."""
    root = research_root() / "results"
    root.mkdir(parents=True, exist_ok=True)
    return root


__all__ = ["DRIVE_ROOT", "LOCAL_SCRATCH_ROOT_COLAB", "in_colab", "drive_mounted",
          "research_root", "local_scratch_root", "dataset_root", "embeddings_root",
          "checkpoints_root", "results_root"]
