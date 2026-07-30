from app.bench.protocol import baseline_comparable


def test_t9_baselines_require_same_mode_and_hardware():
    baseline = {"embedding_mode": "synthetic", "hardware_profile": "cpu-a"}
    assert baseline_comparable(baseline, dict(baseline)) is True
    assert baseline_comparable(
        {"embedding_mode": "cached", "hardware_profile": "cpu-a"}, baseline
    ) is False
    assert baseline_comparable(
        {"embedding_mode": "synthetic", "hardware_profile": "cpu-b"}, baseline
    ) is False
