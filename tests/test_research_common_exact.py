import numpy as np
import pytest

from src.research.common_exact import exact_ranking, recall_at_k_vs_exact, topk_agreement


def test_exact_ranking_is_deterministic_and_sorted_by_score():
    corpus = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32)
    query = np.array([1.0, 0.0], dtype=np.float32)
    order = exact_ranking(corpus, query)
    assert list(order) == [0, 2, 1]  # index0 en yakin, sonra index2, sonra index1


def test_exact_ranking_stable_for_tied_scores():
    corpus = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    query = np.array([1.0, 0.0], dtype=np.float32)
    order1 = exact_ranking(corpus, query)
    order2 = exact_ranking(corpus, query)
    assert list(order1) == list(order2) == [0, 1, 2]  # giris sirasi korunur


def test_recall_at_k_vs_exact_full_overlap_is_one():
    exact_order = np.array([0, 1, 2, 3, 4])
    backend_ids = [0, 1, 2]
    assert recall_at_k_vs_exact(backend_ids, exact_order, k=3) == 1.0


def test_recall_at_k_vs_exact_partial_overlap():
    exact_order = np.array([0, 1, 2, 3, 4])
    backend_ids = [0, 5, 6]
    assert recall_at_k_vs_exact(backend_ids, exact_order, k=3) == pytest.approx(1 / 3)


def test_topk_agreement_identical_sets_is_one():
    assert topk_agreement([1, 2, 3], [3, 2, 1], k=3) == 1.0


def test_topk_agreement_no_overlap_is_zero():
    assert topk_agreement([1, 2, 3], [4, 5, 6], k=3) == 0.0
