import numpy as np

from models.base import feature_tensor


def test_feature_tensor_supports_transformers_4_and_5_outputs():
    direct = np.ones((1, 3), dtype=np.float32)
    assert feature_tensor(direct) is direct

    wrapped = type("WrappedOutput", (), {"pooler_output": direct})()
    assert feature_tensor(wrapped) is direct

    try:
        feature_tensor(object())
    except TypeError as exc:
        assert "Model feature çıktısı" in str(exc)
    else:
        raise AssertionError("Desteklenmeyen çıktı tipi TypeError üretmeliydi")
