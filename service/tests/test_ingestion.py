from __future__ import annotations

import numpy as np

from app.ingestion.load_dataset import _build_vectors, load_auair_bundle


def test_auair_contract_and_altitude_unit():
    bundle = load_auair_bundle()
    assert bundle.dataset[0] == "auair"
    assert len(bundle.segments) == 1866
    assert len(bundle.telemetry) == 1866
    altitudes = bundle.quantile_frame["altitude_m"]
    assert 1 < altitudes.min() < altitudes.max() < 100


def test_build_vectors_derives_all_mrl_dimensions(monkeypatch):
    bundle = load_auair_bundle()
    bundle.vector_payload = bundle.vector_payload[:2]
    vectors = _build_vectors(bundle)
    assert set(vectors) == {2048, 1024, 512, 256}
    for dimension, rows in vectors.items():
        assert len(rows) == 2
        assert len(rows[0]["embedding"]) == dimension
        assert np.linalg.norm(rows[0]["embedding"]) == pytest.approx(1.0, abs=1e-5)


import pytest

