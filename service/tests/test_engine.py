from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.search import engine
from app.search.strategies import SUPPORTED_STRATEGIES


def _request(backend="clickhouse", strategy="exact", telemetry_filters=None, top_k=10):
    return SimpleNamespace(
        query="traffic",
        dataset_id="mini",
        backend=backend,
        strategy=strategy,
        dimension=512,
        adaptive_mrl=SimpleNamespace(enabled=False, base_dim=256, top_n=100),
        metadata_filters={},
        telemetry_filters=telemetry_filters or {},
        pattern="C" if backend == "pgvector" else "A",
        top_k=top_k,
        repeats=1,
    )


@pytest.fixture
def fake_corpus(monkeypatch):
    ids = [f"s{i:03d}" for i in range(200)]

    def filter_ids(dataset_id, metadata_filters, telemetry_filters):
        if telemetry_filters.get("altitude_m") == [-100, -50]:
            return []
        return ids

    def fake_search(dataset_id, dimension, query_vector, top_k, strategy, candidate_ids):
        selected = list(candidate_ids or ids)[:top_k]
        return ([{"segment_id": value, "score": 1.0 - index / 1000} for index, value in enumerate(selected)],
                {"plan_used_vector_index": strategy != "exact", "indexed_vectors_count": 200, "notes": []})

    def hydrate(segment_ids):
        return [{"segment_id": value, "video_id": value, "t_start": 0.0, "t_end": 1.0} for value in segment_ids]

    monkeypatch.setattr(engine.postgres, "filter_segment_ids", filter_ids)
    monkeypatch.setattr(
        engine.postgres, "dataset_info",
        lambda dataset_id: {
            "dataset_id": dataset_id, "has_telemetry": True, "has_captions": False,
            "vector_provenance": "real",
        },
    )
    monkeypatch.setattr(engine.postgres, "hydrate", hydrate)
    monkeypatch.setattr(engine, "embed_query", lambda text, dim: np.ones(dim, dtype=np.float32) / np.sqrt(dim))
    for backend in engine.BACKENDS:
        monkeypatch.setitem(engine.BACKENDS, backend, fake_search)
    return ids


def test_every_backend_strategy_combination_runs_on_200_item_corpus(fake_corpus):
    for backend in ("clickhouse", "qdrant", "pgvector", "numpy_exact"):
        for strategy in SUPPORTED_STRATEGIES[backend]:
            response = engine.search(_request(backend, strategy))
            assert response["diagnostics"]["candidate_count"] == 200
            assert response["diagnostics"]["returned_count"] == 10
            assert response["diagnostics"]["filter_correctness"] is True


def test_negative_control_is_underfilled_and_quality_is_null(fake_corpus):
    response = engine.search(_request(telemetry_filters={"altitude_m": [-100, -50]}))
    assert response["diagnostics"]["returned_count"] == 0
    assert response["diagnostics"]["underfilled"] is True
    assert response["diagnostics"]["underfilled_reason"] == "candidate_shortage"
    assert response["diagnostics"]["candidate_shortage"] is True
    assert response["diagnostics"]["ann_filter_loss"] is False
    assert response["diagnostics"]["quality_vs_groundtruth"] is None
    assert response["diagnostics"]["r_at_1"] is None
    assert response["diagnostics"]["ndcg"] is None


def test_top_k_200_does_not_fail(fake_corpus):
    response = engine.search(_request(top_k=200))
    assert response["diagnostics"]["returned_count"] == 200


def test_underfilled_with_enough_candidates_is_ann_filter_loss(fake_corpus, monkeypatch):
    def short_search(dataset_id, dimension, query_vector, top_k, strategy, candidate_ids):
        return ([{"segment_id": value, "score": 1.0} for value in candidate_ids[:3]], {
            "plan_used_vector_index": True, "indexed_vectors_count": 200, "notes": [],
        })

    monkeypatch.setitem(engine.BACKENDS, "clickhouse", short_search)
    response = engine.search(_request(top_k=10))
    assert response["diagnostics"]["candidate_count"] == 200
    assert response["diagnostics"]["underfilled"] is True
    assert response["diagnostics"]["underfilled_reason"] == "ann_filter_loss"
    assert response["diagnostics"]["underfilled_expected"] is False
