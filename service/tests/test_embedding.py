import numpy as np

from app.embedding.synthetic import synthetic_embedding
from app.mrl import truncate_and_normalize


def test_synthetic_contract_is_deterministic_and_normalized():
    a=synthetic_embedding("same",2048); b=synthetic_embedding("same",2048); c=synthetic_embedding("different",2048)
    assert a.dtype==np.float32 and np.array_equal(a,b) and not np.array_equal(a,c)
    np.testing.assert_allclose(np.linalg.norm(a),1.0,atol=1e-6)


def test_mrl_all_dimensions_are_finite_unit_vectors():
    base=synthetic_embedding("item")
    for dim in (2048,1024,512,256):
        value=truncate_and_normalize(base,dim)
        assert value.shape==(dim,) and np.isfinite(value).all()
        np.testing.assert_allclose(np.linalg.norm(value),1.0,atol=1e-6)
