import numpy as np
import pytest

from scripts.validate_msrvtt import (
    ZERO_SHOT_BASELINE,
    compute_retrieval_metrics,
    mean_rank_vs_chance,
    red_flag_check,
)


def test_compute_retrieval_metrics_perfect_diagonal_is_r1_100():
    sim = np.eye(5)  # caption i en cok video i'ye benziyor, mukemmel eslesme
    m = compute_retrieval_metrics(sim)
    assert m["R@1"] == 100.0
    assert m["R@5"] == 100.0
    assert m["MedR"] == 1.0
    assert m["MeanR"] == 1.0
    assert m["n"] == 5


def test_compute_retrieval_metrics_known_ranks():
    # 3 caption, 3 video. caption0->video0 rank1 (dogru en yuksek),
    # caption1->video1 rank2 (bir yanlis daha yuksek skorlu),
    # caption2->video2 rank3 (en dusuk skor tam dogru video icin)
    sim = np.array([
        [0.9, 0.1, 0.1],   # dogru (video0) en yuksek -> rank1
        [0.5, 0.4, 0.3],   # dogru (video1) ikinci -> rank2
        [0.9, 0.5, 0.1],   # dogru (video2) ucuncu -> rank3
    ])
    m = compute_retrieval_metrics(sim)
    assert m["R@1"] == pytest.approx(100 / 3, abs=0.01)
    assert m["R@5"] == 100.0  # hepsi top-5'te (n=3)
    assert m["MedR"] == 2.0


def test_red_flag_check_flags_large_deviation():
    measured = {"R@1": 10.0, "R@5": 20.0, "R@10": 30.0}
    baseline = {"R@1": 44.5, "R@5": 71.4, "R@10": 81.6}
    flags = red_flag_check(measured, baseline)
    assert len(flags) == 3
    assert "R@1" in flags[0]


def test_red_flag_check_no_flags_when_close():
    measured = {"R@1": 44.0, "R@5": 71.0, "R@10": 81.0}
    baseline = {"R@1": 44.5, "R@5": 71.4, "R@10": 81.6}
    assert red_flag_check(measured, baseline) == []


def test_red_flag_check_ignores_baseline_keys_not_present_like_medr():
    measured = {"R@1": 44.0, "R@5": 71.0, "R@10": 81.0}
    baseline = {"R@1": 44.5, "R@5": 71.4, "R@10": 81.6, "MedR": 2.0}
    # measured'da MedR yok ama KeyError atmamali
    assert red_flag_check(measured, baseline) == []


def test_mean_rank_vs_chance_perfect_retrieval_is_far_better_than_chance():
    # n=100, MeanR=1 (mukemmel) -> chance=(100+1)/2=50.5 -> 50.5x
    msg = mean_rank_vs_chance(1.0, 100)
    assert "MeanR=1.0" in msg
    assert "50.5" in msg
    assert "50.5x" in msg


def test_mean_rank_vs_chance_at_chance_level_is_about_1x():
    # MeanR tam rastgele beklentiye esitse oran ~1x olmali
    n = 999
    chance = (n + 1) / 2
    msg = mean_rank_vs_chance(chance, n)
    assert "1.0x" in msg


def test_mean_rank_vs_chance_matches_real_measured_run():
    # gercek 1000-video kosumundan (artifacts/pipeline_validation.json):
    # MeanR=75.056, n=1000 -> chance=500.5 -> ~6.7x
    msg = mean_rank_vs_chance(75.056, 1000)
    assert "MeanR=75.1" in msg
    assert "6.7x" in msg


def test_zero_shot_baseline_is_the_verified_t2v_row_not_v2t():
    # Portillo-Quintero ve ark. 2021 Tablo 2, T2V yonu (bkz. arXiv:2102.12443
    # ve CLIP4Clip Tablo 1(b) capraz dogrulamasi) - V2T (27.2/51.7/62.6/5)
    # ile KARISTIRILMAMALI, bizim protokolumuz T2V.
    assert ZERO_SHOT_BASELINE == {"R@1": 31.2, "R@5": 53.7, "R@10": 64.2, "MedR": 4.0}


def test_red_flag_check_against_real_measured_run_and_verified_baseline():
    # gercek 1000-video xclip_hf_zeroshot kosumu: R@1=21.5 R@5=42.4 R@10=52.3
    measured = {"R@1": 21.5, "R@5": 42.4, "R@10": 52.3}
    flags = red_flag_check(measured, ZERO_SHOT_BASELINE)
    # R@1 farki 9.7 puan (esigin altinda), R@5/R@10 farki >10 puan (bayrakli)
    assert len(flags) == 2
    assert any("R@5" in f for f in flags)
    assert any("R@10" in f for f in flags)
    assert not any("R@1:" in f for f in flags)
