from __future__ import annotations

from types import SimpleNamespace
from dataclasses import replace

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
    monkeypatch.setattr(engine.postgres, "get_active_run_snapshot", lambda dataset_id: None)
    monkeypatch.setattr(engine, "embed_query", lambda text, dim: np.ones(dim, dtype=np.float32) / np.sqrt(dim))
    monkeypatch.setattr(
        engine, "embed_query_multi",
        lambda text, dims: {dim: np.ones(dim, dtype=np.float32) / np.sqrt(dim) for dim in dims},
    )
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


def test_active_run_snapshot_is_read_once_and_reused_across_repeats(fake_corpus, monkeypatch):
    snapshots = [{
        "run_id": "00000000-0000-0000-0000-000000000001", "vector_provenance": "real",
        "model_id": "model", "model_revision": "revision", "source_commit": "commit",
    }, {
        "run_id": "00000000-0000-0000-0000-000000000002", "vector_provenance": "real",
    }]
    observed = []

    def snapshot(dataset_id):
        return snapshots.pop(0)

    def filter_run(dataset_id, run_id, metadata_filters, telemetry_filters):
        observed.append(("filter", run_id))
        return fake_corpus

    def active_search(dataset_id, dimension, vector, top_k, strategy, candidate_ids, *, run_id):
        observed.append(("search", run_id))
        return ([{"segment_id": value, "score": 1.0} for value in candidate_ids[:top_k]], {
            "plan_used_vector_index": True, "indexed_vectors_count": 200, "notes": [],
        })

    def active_hydrate(ids, *, run_id):
        observed.append(("hydrate", run_id))
        return [{"segment_id": value} for value in ids]

    monkeypatch.setattr(engine.postgres, "get_active_run_snapshot", snapshot)
    monkeypatch.setattr(engine.postgres, "filter_run_segment_ids", filter_run)
    monkeypatch.setattr(engine.postgres, "hydrate", active_hydrate)
    monkeypatch.setitem(engine.BACKENDS, "clickhouse", active_search)
    request = _request()
    request.repeats = 2
    request.filter_execution_mode = "legacy_candidate_ids"
    response = engine.search(request)
    assert response["run_id"] == "00000000-0000-0000-0000-000000000001"
    assert snapshots  # second value was never read
    assert {value for _, value in observed} == {response["run_id"]}


def test_active_pushdown_never_materializes_candidate_ids(fake_corpus, monkeypatch):
    run_id = "00000000-0000-0000-0000-000000000003"
    monkeypatch.setattr(engine.postgres, "get_active_run_snapshot", lambda dataset_id: {
        "run_id": run_id, "vector_provenance": "real", "model_id": "m",
        "model_revision": "r", "source_commit": "s",
    })
    monkeypatch.setattr(
        engine.postgres, "filter_run_segment_ids",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("candidate IDs must not be loaded")),
    )

    def native_search(
        dataset_id, dimension, vector, top_k, strategy, candidate_ids, *, run_id,
        metadata_filters, telemetry_filters,
    ):
        assert candidate_ids is None
        assert telemetry_filters == {"altitude_m": [4, 6]}
        return ([{"segment_id": "s005", "score": 0.9}], {
            "candidate_count": 7, "plan_used_vector_index": True,
            "indexed_vectors_count": 200, "notes": [],
        })

    monkeypatch.setitem(engine.BACKENDS, "clickhouse", native_search)
    monkeypatch.setattr(engine.postgres, "hydrate", lambda ids, *, run_id: [{
        "segment_id": "s005", "altitude_m": 5.0,
    }])
    request = _request(telemetry_filters={"altitude_m": [4, 6]})
    request.filter_execution_mode = "pushdown"
    response = engine.search(request)
    assert response["filter_execution_mode"] == "pushdown"
    assert response["diagnostics"]["candidate_count"] == 7
    assert response["diagnostics"]["filter_correctness"] is True


def test_legacy_candidate_mode_fails_before_embedding_when_limit_exceeded(fake_corpus, monkeypatch):
    monkeypatch.setattr(engine, "settings", replace(engine.settings, legacy_candidate_limit=10))
    request = _request()
    request.filter_execution_mode = "legacy_candidate_ids"
    with pytest.raises(ValueError, match="legacy candidate limit exceeded"):
        engine.search(request)


def test_adaptive_pushdown_reuses_predicate_and_only_passes_bounded_rerank_ids(fake_corpus, monkeypatch):
    run_id = "00000000-0000-0000-0000-000000000004"
    monkeypatch.setattr(engine.postgres, "get_active_run_snapshot", lambda dataset_id: {
        "run_id": run_id, "vector_provenance": "real", "model_id": "m",
        "model_revision": "r", "source_commit": "s",
    })
    observed = []

    def native_search(dataset_id, dimension, vector, top_k, strategy, candidate_ids, **kwargs):
        observed.append((dimension, candidate_ids, kwargs["telemetry_filters"]))
        ids = [f"s{index:03d}" for index in range(min(top_k, 4))] if candidate_ids is None else candidate_ids[:top_k]
        return ([{"segment_id": value, "score": 1.0} for value in ids], {
            "candidate_count": 50, "plan_used_vector_index": True,
            "indexed_vectors_count": 50, "notes": [],
        })

    monkeypatch.setitem(engine.BACKENDS, "clickhouse", native_search)
    monkeypatch.setattr(engine.postgres, "hydrate", lambda ids, *, run_id: [
        {"segment_id": value, "altitude_m": 5.0} for value in ids
    ])
    request = _request(telemetry_filters={"altitude_m": [4, 6]})
    request.filter_execution_mode = "pushdown"
    request.adaptive_mrl.enabled = True
    request.adaptive_mrl.top_n = 4
    engine.search(request)
    assert observed[0] == (256, None, {"altitude_m": [4, 6]})
    assert observed[1][0] == 512 and len(observed[1][1]) <= 4
    assert observed[1][2] == observed[0][2]
