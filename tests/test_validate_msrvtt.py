import numpy as np
import pytest

from scripts.validate_msrvtt import compute_retrieval_metrics, red_flag_check


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
