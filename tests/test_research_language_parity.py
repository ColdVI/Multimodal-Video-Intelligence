from __future__ import annotations

import numpy as np

from bench.language_parity import QueryPair, load_pairs, measure_alignment, DEFAULT_FIXTURE_PATH


def test_fixture_loads_and_skips_entries_without_both_languages():
    pairs = load_pairs()
    ids = {pair.query_id for pair in pairs}
    assert "S20" not in ids  # empty tr/en edge case
    assert "S23" not in ids  # generator-only entry, no tr/en fields
    assert "S01" in ids
    assert any(pair.evaluation == "exploratory" for pair in pairs)  # S17-S19
    assert any(pair.evaluation == "binding" for pair in pairs)


def test_perfect_cross_lingual_embedder_scores_1_0_on_everything():
    pairs = [
        QueryPair("a", "tr metin a", "en text a", "t", "binding"),
        QueryPair("b", "tr metin b", "en text b", "t", "binding"),
        QueryPair("c", "tr metin c", "en text c", "t", "binding"),
    ]
    # A "perfect" embedder maps each pair's tr/en text to the exact same vector,
    # distinct per pair -- the idealized case this metric should recognize as 1.0.
    fixed = {"tr metin a": [1, 0, 0], "en text a": [1, 0, 0],
             "tr metin b": [0, 1, 0], "en text b": [0, 1, 0],
             "tr metin c": [0, 0, 1], "en text c": [0, 0, 1]}
    result = measure_alignment(pairs, lambda text: np.array(fixed[text], dtype=np.float32))
    assert result["overall"]["mean_same_pair_cosine"] == 1.0
    assert result["overall"]["tr_to_en_top1_accuracy"] == 1.0
    assert result["overall"]["en_to_tr_top1_accuracy"] == 1.0


def test_language_blind_embedder_that_ignores_meaning_scores_near_chance():
    """If the embedder maps every text to the same constant vector regardless of
    language OR meaning, same-pair cosine is trivially 1.0 (useless on its own) but
    top-1 cross-lingual retrieval accuracy must NOT be 1.0 -- every EN vector is equally
    close to every TR vector, so argmax ties resolve to index 0 for all queries, which
    this test locks in as the correct degenerate behavior rather than silently "passing"."""
    pairs = [
        QueryPair("a", "tr a", "en a", "t", "binding"),
        QueryPair("b", "tr b", "en b", "t", "binding"),
        QueryPair("c", "tr c", "en c", "t", "binding"),
    ]
    constant = np.array([1.0, 0.0], dtype=np.float32)
    result = measure_alignment(pairs, lambda text: constant)
    assert result["overall"]["mean_same_pair_cosine"] == 1.0
    assert result["overall"]["tr_to_en_top1_accuracy"] < 1.0
    per_pair_ids = [row["tr_nearest_en_query_id"] for row in result["per_pair"]]
    assert all(value == "a" for value in per_pair_ids)  # all ties resolve to the first pair


def test_exploratory_and_binding_pairs_are_scored_separately():
    pairs = [
        QueryPair("a", "tr a", "en a", "t", "binding"),
        QueryPair("b", "tr b", "en b", "t", "exploratory"),
    ]
    result = measure_alignment(pairs, lambda text: np.array([hash(text) % 97, 1.0], dtype=np.float32))
    assert result["binding"]["n"] == 1
    assert result["exploratory"]["n"] == 1
    assert result["overall"]["n"] == 2


def test_fixture_path_is_the_real_shared_fixture_not_a_private_copy():
    assert DEFAULT_FIXTURE_PATH.name == "queries_semantic.json"
    assert DEFAULT_FIXTURE_PATH.exists()
