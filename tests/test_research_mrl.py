import numpy as np

from src.research.mrl import derive_all_dims, validate_mrl_derivation


def _random_unit_vec(dim=2048, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def test_derive_all_dims_returns_unit_norm_vectors():
    e = _random_unit_vec()
    derived = derive_all_dims(e, (1024, 512, 256))
    for d, e_d in derived.items():
        assert len(e_d) == d
        assert abs(np.linalg.norm(e_d) - 1.0) < 1e-5


def test_validate_mrl_derivation_passes_for_correct_derivation():
    e = _random_unit_vec()
    derived = derive_all_dims(e, (1024, 512, 256))
    problems = validate_mrl_derivation(e, derived)
    assert problems == []


def test_validate_mrl_derivation_catches_wrong_length():
    e = _random_unit_vec()
    derived = derive_all_dims(e, (512,))
    derived[512] = derived[512][:500]  # bozuk uzunluk
    problems = validate_mrl_derivation(e, derived)
    assert any("uzunluk" in p for p in problems)


def test_validate_mrl_derivation_catches_wrong_norm():
    e = _random_unit_vec()
    derived = {512: (np.asarray(e[:512], dtype=np.float32) * 3.0).tolist()}  # normalize edilmemis
    problems = validate_mrl_derivation(e, derived)
    assert any("||e_d||_2" in p for p in problems)


def test_validate_mrl_derivation_catches_nan():
    e = _random_unit_vec()
    derived = derive_all_dims(e, (256,))
    derived[256][0] = float("nan")
    problems = validate_mrl_derivation(e, derived)
    assert any("NaN" in p for p in problems)


def test_prefix_not_bit_identical_but_same_ranking_direction():
    """SS3.3: e_2048[:512] ile e_512 bit-identical DEGIL (normalizasyon farki)
    ama yonleri (dolayisiyla cosine siralamasi) ayni olmali."""
    e = _random_unit_vec()
    derived = derive_all_dims(e, (512,))
    raw_prefix = e[:512]
    assert derived[512] != raw_prefix  # bit-identical degil
    problems = validate_mrl_derivation(e, derived)
    assert problems == []  # ama yon ayni oldugu icin dogrulama gecer
