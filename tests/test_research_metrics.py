import numpy as np
import pytest

from src.research.metrics import corpus_retrieval_metrics, paired_bootstrap_ci


def test_perfect_retrieval_gives_r_at_1_100():
    sim = np.eye(4, dtype=np.float32)  # sorgu i en cok item i'ye benziyor
    gt = [[0], [1], [2], [3]]
    out = corpus_retrieval_metrics(sim, gt)
    assert out["R@1"] == 100.0
    assert out["MeanRank"] == 1.0
    assert out["MedianRank"] == 1.0
    assert out["MRR"] == 1.0


def test_worst_case_retrieval_gives_low_r_at_1():
    n = 5
    sim = 1.0 - np.eye(n, dtype=np.float32)  # dogru cevap her zaman EN DUSUK skor
    gt = [[i] for i in range(n)]
    out = corpus_retrieval_metrics(sim, gt)
    assert out["R@1"] == 0.0
    assert out["MeanRank"] == n  # her zaman son sirada


def test_multi_gt_per_query_capera_style():
    """CapERA: bir video 5 caption'a karsilik gelebilir - GT KUMESI (tek index
    degil). Sorgu 0'in GT'si {0, 1} olsun; index 1 en yuksek skor alsa da
    rank=1 sayilmali (kumede)."""
    sim = np.array([[0.1, 0.9, 0.0]], dtype=np.float32)
    gt = [[0, 1]]
    out = corpus_retrieval_metrics(sim, gt)
    assert out["R@1"] == 100.0


def test_empty_gt_excluded_from_valid_and_does_not_crash():
    sim = np.array([[0.5, 0.3], [0.1, 0.9]], dtype=np.float32)
    gt = [[], [1]]
    out = corpus_retrieval_metrics(sim, gt)
    assert out["n"] == 2
    assert out["n_valid"] == 1


def test_paired_bootstrap_ci_reexported_and_works():
    a = [0.5, 0.6, 0.7, 0.8]
    b = [0.4, 0.5, 0.6, 0.7]
    result = paired_bootstrap_ci(a, b, n_resamples=200, seed=1)
    assert result["mean_diff"] == pytest.approx(0.1)
