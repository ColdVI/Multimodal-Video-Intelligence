from app.ingestion.load_dataset import load_capera_bundle


def test_t7_capera_test_split_bundle_integrity():
    bundle = load_capera_bundle()
    assert len(bundle.segments) == 1391
    assert len(bundle.groundtruth) == 6955
    assert len({row[0] for row in bundle.segments}) == 1391
    assert len({row[1] for row in bundle.groundtruth}) == 6955
    assert {row[7] for row in bundle.groundtruth} == {"unknown"}
