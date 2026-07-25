import math

from scripts.mrl_truncate_embeddings import truncate_and_renormalize


def test_truncate_keeps_first_n_dims_direction():
    vec = [3.0, 4.0, 0.0, 0.0]  # norm 5
    out = truncate_and_renormalize(vec, 2)
    assert len(out) == 2
    assert math.isclose(out[0], 0.6, rel_tol=1e-5)
    assert math.isclose(out[1], 0.8, rel_tol=1e-5)


def test_truncate_output_is_unit_norm():
    vec = [1.0, 2.0, 3.0, 4.0, 5.0]
    out = truncate_and_renormalize(vec, 3)
    norm = sum(x * x for x in out) ** 0.5
    assert math.isclose(norm, 1.0, rel_tol=1e-5)


def test_truncate_handles_zero_vector():
    out = truncate_and_renormalize([0.0, 0.0, 0.0], 2)
    assert out == [0.0, 0.0]
