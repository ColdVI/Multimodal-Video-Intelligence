"""Faz 6 Colab handoff - drive_manifest.json'daki klasor/dosya beklentilerini
GERCEK Drive icerigiyle karsilastirir. Notebook 01/02/04 baslamadan once
(veya elle) calistirilir - eksik girdi varsa ACIKCA listeler, sessizce
devam etmez."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.research import colab_paths

DRIVE_MANIFEST_PATH = pathlib.Path("drive_manifest.json")


def _check_folder(folder_rel: str, expected_files: list) -> dict:
    # research_root() DRIVE_ROOT'a (Colab) ya da yerel fallback'e cozer;
    # datasets/checkpoints/embeddings hepsi bunun ALTINDA - colab_paths.
    # dataset_root()/checkpoints_root() vb. ile AYNI sema. "." (kok) ozel
    # durum: sadece DOGRUDAN cocuklar (sonuc dosyalari), ic ice datasets/
    # checkpoints/embeddings alt klasorlerini TEKRAR SAYMAMAK icin recursive
    # DEGIL.
    folder = colab_paths.research_root() / folder_rel
    exists = folder.exists()
    found_files = []
    if exists:
        globber = folder.iterdir() if folder_rel == "." else folder.rglob("*")
        found_files = [p.name if folder_rel == "." else str(p.relative_to(folder))
                      for p in globber if p.is_file()]
    return {"folder": str(folder), "exists": exists, "n_files_found": len(found_files),
           "expected_files_hint": expected_files, "sample_found": found_files[:10]}


def main():
    if not DRIVE_MANIFEST_PATH.exists():
        print(f"HATA: {DRIVE_MANIFEST_PATH} yok.")
        raise SystemExit(1)

    manifest = json.loads(DRIVE_MANIFEST_PATH.read_text(encoding="utf-8"))
    print(f"Drive kok: {colab_paths.DRIVE_ROOT}")
    print(f"drive_mounted={colab_paths.drive_mounted()}  research_root={colab_paths.research_root()}")
    print()

    results = {}
    all_ok = True
    for folder_rel, info in manifest["folders"].items():
        r = _check_folder(folder_rel, info["expected_files"])
        results[folder_rel] = r
        status = "OK" if r["exists"] and r["n_files_found"] > 0 else "EKSIK/BOS"
        if status != "OK":
            all_ok = False
        print(f"[{status}] {folder_rel}: {r['n_files_found']} dosya bulundu "
             f"(beklenen ornekler: {info['expected_files']})")

    dl_manifest_path = colab_paths.research_root() / "dataset_download_manifest.json"
    if dl_manifest_path.exists():
        dl_manifest = json.loads(dl_manifest_path.read_text(encoding="utf-8"))
        print("\ndataset_download_manifest.json bulundu - lisans/kaynak/resume bilgisi:")
        for ds, info in dl_manifest.items():
            print(f"  {ds}: lisans={info.get('license')}, yontem={info.get('download_method')}")
    else:
        print(f"\n[BILGI] {dl_manifest_path} henuz yok - notebook 01 calistirilmamis olabilir.")

    print(f"\nGENEL DURUM: {'TUM KLASORLER HAZIR' if all_ok else 'BAZI KLASORLER EKSIK/BOS - yukaridaki listeye bakin'}")
    return results


if __name__ == "__main__":
    main()
