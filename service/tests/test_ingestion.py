from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.ingestion.load_dataset import _build_vectors, load_auair_bundle, load_capera_bundle


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


@pytest.mark.skipif(
    not Path("data/downloads/capera/CapERA_DATASET_train.json").exists(),
    reason="CapERA source data is not present; contract acceptance is NOT RUN",
)
def test_capera_bundle_is_test_only_split_qualified_and_has_exact_unknown_gt():
    bundle = load_capera_bundle()
    assert len(bundle.videos) == len(bundle.segments) == 1391
    assert len(bundle.groundtruth) == 6955
    assert all(row[0].startswith("capera:test__") for row in bundle.segments)
    assert all(row[4].startswith("test__") for row in bundle.groundtruth)
    assert {row[7] for row in bundle.groundtruth} == {"unknown"}
    assert len({row[1] for row in bundle.groundtruth}) == 6955
