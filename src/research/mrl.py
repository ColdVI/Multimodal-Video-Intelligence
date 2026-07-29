"""MRL (Matryoshka) turetme + dogrulama (spec SS3.3). Turetme mantigi
scripts/mrl_truncate_embeddings.py::truncate_and_renormalize ile AYNI
fonksiyon (import edilir, kopyalanmaz) - bu modul ona SS3.3'teki zorunlu
hard-assert dogrulamalarini ekler."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.mrl_truncate_embeddings import truncate_and_renormalize  # noqa: E402

NORM_TOL = 1e-5


def derive_all_dims(e_2048: list, dims: tuple) -> dict:
    """{d: e_d} - her d icin truncate_and_renormalize(e_2048, d)."""
    return {d: truncate_and_renormalize(e_2048, d) for d in dims}


def validate_mrl_derivation(e_2048: list, derived: dict) -> list:
    """SS3.3 hard-assert'leri. Basarisiz olanlarin metin listesini dondurur
    (bos liste = hepsi gecti) - notebook hucresi bunu assert ile raise'e
    cevirir, boylece testler mesaji da dogrulayabilir."""
    problems = []
    e_2048_arr = np.asarray(e_2048, dtype=np.float32)
    for d, e_d in derived.items():
        arr = np.asarray(e_d, dtype=np.float32)
        if arr.shape[0] != d:
            problems.append(f"d={d}: uzunluk {arr.shape[0]} != {d}")
            continue
        norm = float(np.linalg.norm(arr))
        if abs(norm - 1.0) > NORM_TOL and norm != 0.0:
            problems.append(f"d={d}: ||e_d||_2={norm:.7f}, 1.0 +/- {NORM_TOL} disinda")
        if not np.all(np.isfinite(arr)):
            problems.append(f"d={d}: NaN/Inf var")
        prefix = e_2048_arr[:d]
        prefix_norm = np.linalg.norm(prefix)
        if prefix_norm > 0:
            expected_direction = prefix / prefix_norm
            # cosine sıralaması normalizasyondan etkilenmez: yon (birim vektor)
            # e_d ile AYNI olmali, bit-identical olmasi beklenmiyor (SS3.3).
            cos = float(np.dot(arr, expected_direction))
            if cos < 1.0 - 1e-4:
                problems.append(f"d={d}: yon(e_d) != normalize(prefix) (cos={cos:.6f})")
    return problems


__all__ = ["derive_all_dims", "validate_mrl_derivation", "truncate_and_renormalize", "NORM_TOL"]
