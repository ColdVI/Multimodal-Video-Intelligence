from pathlib import Path

import pytest

from app.ingestion.load_dataset import load_capera_bundle


@pytest.mark.skipif(
    not Path("data/downloads/capera/CapERA_DATASET_train.json").exists(),
    reason="CapERA source data is not present; integrity acceptance is NOT RUN",
)
def test_t7_capera_test_split_bundle_integrity():
    bundle = load_capera_bundle()
    assert len(bundle.segments) == 1391
    assert len(bundle.groundtruth) == 6955
    assert len({row[0] for row in bundle.segments}) == 1391
    assert len({row[1] for row in bundle.groundtruth}) == 6955
    assert {row[7] for row in bundle.groundtruth} == {"unknown"}
