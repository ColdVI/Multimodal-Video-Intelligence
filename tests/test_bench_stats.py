import pytest

from bench.stats import _norm_ppf, minimum_detectable_effect, paired_bootstrap_ci


def test_norm_ppf_matches_known_z_scores():
    # standart tablo degerleri: z_0.975 (alpha=0.05 iki-kuyruklu), z_0.80 (power=0.8)
    assert _norm_ppf(0.975) == pytest.approx(1.95996, abs=1e-4)
    assert _norm_ppf(0.80) == pytest.approx(0.84162, abs=1e-4)
    assert _norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)


def test_norm_ppf_rejects_out_of_range():
    with pytest.raises(ValueError):
        _norm_ppf(0.0)
    with pytest.raises(ValueError):
        _norm_ppf(1.0)


def test_minimum_detectable_effect_matches_hand_computation():
    # MDE = (z_0.975 + z_0.80) * std * sqrt(2/n), n=28, std=0.2 (bu projedeki
    # gercek yaklasik std'ye yakin, bkz. TASKS.md Faz 6 "saptanabilir minimum
    # fark ~0.20" notu)
    n, std = 28, 0.2
    expected = (1.95996 + 0.84162) * std * (2 / n) ** 0.5
    assert minimum_detectable_effect(n, std) == pytest.approx(expected, abs=1e-3)


def test_minimum_detectable_effect_shrinks_with_more_queries():
    small = minimum_detectable_effect(n=28, std=0.2)
    large = minimum_detectable_effect(n=150, std=0.2)
    assert large < small  # daha fazla sorgu -> daha kucuk farklari saptayabiliriz


def test_minimum_detectable_effect_zero_n_is_infinite():
    assert minimum_detectable_effect(n=0, std=0.2) == float("inf")


def test_paired_bootstrap_ci_identical_arrays_is_zero_diff():
    values = [0.5, 0.6, 0.7, 0.4, 0.8]
    result = paired_bootstrap_ci(values, values, n_resamples=500, seed=0)
    assert result["mean_diff"] == 0.0
    assert result["ci_lo"] == 0.0
    assert result["ci_hi"] == 0.0


def test_paired_bootstrap_ci_constant_offset_recovers_true_diff():
    values_b = [0.5, 0.6, 0.7, 0.4, 0.8, 0.55, 0.65]
    values_a = [v + 0.1 for v in values_b]
    result = paired_bootstrap_ci(values_a, values_b, n_resamples=2000, seed=0)
    assert result["mean_diff"] == pytest.approx(0.1, abs=1e-9)
    # sabit fark - tum resample'larda ayni deger, CI tek noktaya sikismali
    assert result["ci_lo"] == pytest.approx(0.1, abs=1e-9)
    assert result["ci_hi"] == pytest.approx(0.1, abs=1e-9)


def test_paired_bootstrap_ci_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1, 2, 3], [1, 2])


def test_paired_bootstrap_ci_empty_input_does_not_crash():
    result = paired_bootstrap_ci([], [])
    assert result["n"] == 0
    assert result["mean_diff"] == 0.0


def test_paired_bootstrap_ci_single_pair_is_degenerate_but_valid():
    result = paired_bootstrap_ci([0.7], [0.5])
    assert result["n"] == 1
    assert result["mean_diff"] == pytest.approx(0.2)
    assert result["ci_lo"] == result["ci_hi"] == pytest.approx(0.2)


def test_paired_bootstrap_ci_noisy_data_ci_contains_true_mean():
    import random
    rng = random.Random(42)
    # gercek fark 0.1, gurultu ekli - CI gercek degeri icermeli (istatistiksel
    # olarak %95 CI'nin bunu %95 ihtimalle icermesi beklenir, tek bir kosumda
    # deterministik garanti degil ama sabit seed ile tekrarlanabilir)
    values_b = [rng.gauss(0.5, 0.05) for _ in range(50)]
    values_a = [v + 0.1 + rng.gauss(0, 0.02) for v in values_b]
    result = paired_bootstrap_ci(values_a, values_b, n_resamples=3000, seed=1)
    assert result["ci_lo"] < 0.1 < result["ci_hi"]
