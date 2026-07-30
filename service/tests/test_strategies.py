from __future__ import annotations

import pytest

from app.search.strategies import (
    SUPPORTED_STRATEGIES,
    clickhouse_settings,
    pgvector_session_settings,
    qdrant_search_params,
    validate_strategy,
)


def test_every_required_backend_strategy_is_registered():
    assert set(SUPPORTED_STRATEGIES["clickhouse"]) == {"exact", "ann", "prefilter", "postfilter"}
    assert set(SUPPORTED_STRATEGIES["qdrant"]) == {"exact", "ann", "ann_high_ef", "prefilter"}
    assert set(SUPPORTED_STRATEGIES["pgvector"]) == {"exact", "ann", "iterative_scan", "iterative_strict"}
    for backend, strategies in SUPPORTED_STRATEGIES.items():
        for strategy in strategies:
            validate_strategy(backend, strategy)


def test_clickhouse_top_k_200_raises_query_scoped_limit():
    settings, notes = clickhouse_settings("ann", 200)
    assert settings["max_limit_for_vector_search_queries"] == 200
    assert any("query-scoped" in note for note in notes)


def test_strategy_specific_settings():
    assert clickhouse_settings("exact", 10)[0]["query_plan_try_use_vector_search"] == 0
    assert qdrant_search_params("exact") == {"exact": True}
    assert qdrant_search_params("ann_high_ef") == {"hnsw_ef": 512}
    assert any("relaxed_order" in value for value in pgvector_session_settings("iterative_scan"))


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        validate_strategy("clickhouse", "iterative_scan")

