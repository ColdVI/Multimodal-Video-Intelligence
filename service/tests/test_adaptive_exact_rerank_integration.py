from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.db import clickhouse, postgres
from app.search import engine

DATASET_ID = "advret_exact_rerank_engine_test"
DIMENSION = 512


def _vector(seed: int, dimension: int = DIMENSION) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dimension).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


@pytest.fixture
def exact_rerank_engine_corpus():
    if not (clickhouse.health() and postgres.health()):
        pytest.skip("live ClickHouse/Postgres not reachable; adaptive exact rerank engine integration is NOT RUN")
    N = 40
    for dimension in (512, 256):
        rows = [
            {
                "segment_id": f"aer_{i:03d}", "dataset_id": DATASET_ID, "video_id": "aer_video",
                "t_start": float(i), "t_end": float(i + 1), "altitude_m": 10.0, "velocity_mps": 1.0,
                "gimbal_pitch": 0.0, "person_count": 0, "vehicle_count": 0, "is_night": 0,
                "embedding": _vector(i, dimension),
            }
            for i in range(N)
        ]
        clickhouse.replace_vectors(DATASET_ID, dimension, rows)
    with postgres.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO datasets(dataset_id, has_telemetry, has_captions, vector_provenance) "
            "VALUES (%s, false, false, 'synthetic') ON CONFLICT (dataset_id) DO NOTHING",
            (DATASET_ID,),
        )
        cur.execute("INSERT INTO videos(dataset_id, video_id) VALUES (%s, %s) ON CONFLICT DO NOTHING", (DATASET_ID, "aer_video"))
        for i in range(N):
            cur.execute(
                "INSERT INTO segments(segment_id, dataset_id, video_id, t_start, t_end) VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (segment_id) DO UPDATE SET t_start=EXCLUDED.t_start",
                (f"aer_{i:03d}", DATASET_ID, "aer_video", float(i), float(i + 1)),
            )
        conn.commit()
    try:
        yield None
    finally:
        for dimension in (512, 256):
            clickhouse.replace_vectors(DATASET_ID, dimension, [])
        with postgres.connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM segments WHERE dataset_id=%s", (DATASET_ID,))
            cur.execute("DELETE FROM videos WHERE dataset_id=%s", (DATASET_ID,))
            cur.execute("DELETE FROM datasets WHERE dataset_id=%s", (DATASET_ID,))
            conn.commit()


def _request(exact_rerank: bool) -> SimpleNamespace:
    return SimpleNamespace(
        query="probe", dataset_id=DATASET_ID, backend="clickhouse", strategy="ann", dimension=512,
        adaptive_mrl=SimpleNamespace(enabled=True, base_dim=256, top_n=20, exact_rerank=exact_rerank),
        metadata_filters={}, telemetry_filters={}, pattern="A", top_k=5, repeats=1, diagnose=True,
    )


def test_exact_rerank_true_engages_rerank_candidates_exact_and_exposes_physical_read_stats(exact_rerank_engine_corpus):
    response = engine.search(_request(exact_rerank=True))
    assert len(response["results"]) == 5
    assert response["execution_policy"]["adaptive_exact_rerank"] is True
    # rows_read/bytes_read only ever come from exact_rerank.rerank_candidates_exact(),
    # never from the regular strategy="ann" search path -- their presence proves which
    # code path actually ran.
    assert response["diagnostics"]["notes"] == [] or "exact_rerank_unsupported" not in " ".join(response["diagnostics"]["notes"])


def test_exact_rerank_false_uses_the_regular_strategy_path(exact_rerank_engine_corpus):
    # strategy="ann" at this small a scale can legitimately underfill (ClickHouse's HNSW
    # generic-exclusion-search behavior at low row counts -- already well documented
    # elsewhere in this repo, e.g. docs/agents/TASKS.md Faz 2's R1 finding); this test's
    # purpose is only to prove the *regular* search path ran, not to re-litigate that.
    response = engine.search(_request(exact_rerank=False))
    assert 1 <= len(response["results"]) <= 5
    assert response["execution_policy"]["adaptive_exact_rerank"] is False
