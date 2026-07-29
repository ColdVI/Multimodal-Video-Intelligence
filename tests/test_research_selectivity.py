from src.research.selectivity import derive_thresholds


def test_less_than_threshold_matches_target_selectivity_on_uniform_data():
    values = list(range(1000))  # 0..999 uniform
    out = derive_thresholds(values, levels=(0.5, 0.1))
    assert abs(out[0.5]["actual_selectivity"] - 0.5) < 0.01
    assert abs(out[0.1]["actual_selectivity"] - 0.1) < 0.01


def test_greater_than_threshold_matches_target_selectivity():
    values = list(range(1000))
    out = derive_thresholds(values, levels=(0.1,), direction="greater_than")
    assert abs(out[0.1]["actual_selectivity"] - 0.1) < 0.01


def test_no_fixed_threshold_narrow_range_still_derives_from_data():
    """spec SS0 bulgu #2: AU-AIR irtifasi 5-30m dar araliginda - sabit esik
    yazilamaz, veriden turetilmeli. Dar aralikli veri icin de calismali."""
    values = [5.0 + i * 0.01 for i in range(2500)]  # 5.0..30.0 dar aralik
    out = derive_thresholds(values, levels=(0.5,))
    assert 5.0 <= out[0.5]["threshold"] <= 30.0


def test_empty_values_returns_none_not_crash():
    out = derive_thresholds([], levels=(0.5,))
    assert out[0.5]["threshold"] is None
    assert out[0.5]["n"] == 0


def test_none_values_are_filtered_before_quantile():
    values = [1, 2, None, 3, None, 4]
    out = derive_thresholds(values, levels=(0.5,))
    assert out[0.5]["n"] == 4
