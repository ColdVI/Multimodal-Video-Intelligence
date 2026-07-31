from __future__ import annotations

import numpy as np

from app.embedding import router


def test_embed_query_multi_calls_raw_embedding_exactly_once(monkeypatch):
    """Phase -1.2 regression: adaptive MRL needs both a base_dim and a full dimension
    vector for one query. Before the fix, engine.py called embed_query() twice, and in
    `real` mode each call reruns the full Qwen forward pass from scratch (no caching
    exists there). embed_query_multi must derive both dimensions from one raw call."""
    calls = []

    def fake_raw(text):
        calls.append(text)
        return np.arange(2048, dtype=np.float32) + 1.0

    monkeypatch.setattr(router, "embed_query_raw", fake_raw)
    vectors = router.embed_query_multi("a query", (512, 256))
    assert calls == ["a query"]
    assert set(vectors) == {512, 256}
    assert vectors[512].shape == (512,)
    assert vectors[256].shape == (256,)
    assert np.isclose(np.linalg.norm(vectors[512]), 1.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(vectors[256]), 1.0, atol=1e-5)
    # 256d is a true prefix-truncation of the same raw vector as 512d, not an
    # independently-derived embedding -- this is the MRL property the fix relies on.
    ratio = vectors[512][:256] / vectors[256]
    assert np.allclose(ratio, ratio[0], atol=1e-4)


def test_embed_query_single_dimension_still_works_via_raw(monkeypatch):
    calls = []
    monkeypatch.setattr(router, "embed_query_raw", lambda text: calls.append(text) or np.ones(2048, dtype=np.float32))
    vector = router.embed_query("q", 128)
    assert calls == ["q"]
    assert vector.shape == (128,)


def test_engine_adaptive_search_triggers_exactly_one_raw_embedding_call(monkeypatch):
    from types import SimpleNamespace

    from app.search import engine

    calls = []
    monkeypatch.setattr(router, "embed_query_raw", lambda text: calls.append(text) or (np.arange(2048, dtype=np.float32) + 1.0))
    monkeypatch.setattr(engine.postgres, "dataset_info", lambda dataset_id: {
        "dataset_id": dataset_id, "has_telemetry": True, "has_captions": False, "vector_provenance": "real",
    })
    monkeypatch.setattr(engine.postgres, "get_active_run_snapshot", lambda dataset_id: None)
    monkeypatch.setattr(engine.postgres, "filter_segment_ids", lambda *a, **k: [f"s{i:03d}" for i in range(50)])
    monkeypatch.setattr(engine.postgres, "hydrate", lambda ids: [{"segment_id": i} for i in ids])

    def fake_search(dataset_id, dimension, query_vector, top_k, strategy, candidate_ids):
        selected = list(candidate_ids)[:top_k]
        return ([{"segment_id": v, "score": 1.0} for v in selected],
                {"plan_used_vector_index": True, "indexed_vectors_count": 50, "notes": []})

    monkeypatch.setitem(engine.BACKENDS, "clickhouse", fake_search)
    request = SimpleNamespace(
        query="traffic", dataset_id="mini", backend="clickhouse", strategy="ann", dimension=512,
        adaptive_mrl=SimpleNamespace(enabled=True, base_dim=256, top_n=20),
        metadata_filters={}, telemetry_filters={}, pattern="A", top_k=10, repeats=3,
    )
    engine.search(request)
    # repeats=3 -> 3 real _one_run calls, each must do exactly one raw embedding call
    # (not two), regardless of how many MRL dimensions that run needs.
    assert calls == ["traffic", "traffic", "traffic"]
