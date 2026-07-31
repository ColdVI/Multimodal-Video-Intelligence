from __future__ import annotations

import threading

import numpy as np
import pytest

from app.db import clickhouse

DATASET_ID = "advret_phase_neg1_client_test"
DIMENSION = 512


def _vector(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(DIMENSION).astype(np.float32)
    return (values / np.linalg.norm(values)).tolist()


@pytest.fixture(autouse=True)
def _clear_thread_local_client():
    clickhouse._reset_client()
    yield
    clickhouse._reset_client()


def test_client_is_reused_across_calls_on_the_same_thread(monkeypatch):
    created = []

    class FakeClient:
        pass

    def fake_new_client():
        instance = FakeClient()
        created.append(instance)
        return instance

    monkeypatch.setattr(clickhouse, "_new_client", fake_new_client)
    first = clickhouse.client()
    second = clickhouse.client()
    third = clickhouse.client()
    assert first is second is third
    assert len(created) == 1  # not one new connection per call


def test_each_thread_gets_its_own_client_instance_no_cross_thread_sharing(monkeypatch):
    """The safety property this design relies on: a Client is never shared across
    threads, so clickhouse_connect's per-instance concurrent-query thread-safety is
    never actually exercised -- each worker thread in FastAPI's threadpool has its own."""
    created_count = {"n": 0}
    lock = threading.Lock()

    class FakeClient:
        def __init__(self, owner_thread):
            self.owner_thread = owner_thread

    def fake_new_client():
        with lock:
            created_count["n"] += 1
        return FakeClient(threading.current_thread().ident)

    monkeypatch.setattr(clickhouse, "_new_client", fake_new_client)
    results: dict[int, object] = {}
    results_lock = threading.Lock()

    def worker():
        instance = clickhouse.client()
        with results_lock:
            results[threading.current_thread().ident] = instance

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert created_count["n"] == 10  # one client per thread, never reused across threads
    instances = list(results.values())
    assert len({id(instance) for instance in instances}) == 10  # all distinct objects
    for thread_id, instance in results.items():
        assert instance.owner_thread == thread_id  # each thread only ever saw its own


def test_transient_connection_error_resets_client_and_recovers(monkeypatch):
    from clickhouse_connect.driver.exceptions import OperationalError

    attempts = {"n": 0}
    created = []

    class FlakyThenFineClient:
        def command(self, sql):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise OperationalError("simulated connection drop")
            return 1

    def fake_new_client():
        instance = FlakyThenFineClient()
        created.append(instance)
        return instance

    monkeypatch.setattr(clickhouse, "_new_client", fake_new_client)
    assert clickhouse.health() is True
    assert len(created) == 2  # first (broken) client discarded, second one used to recover
    assert attempts["n"] == 2


def test_non_transient_error_does_not_retry_or_reset_client(monkeypatch):
    """A real programming/data error must propagate immediately, not trigger a pointless
    reconnect-and-retry that doubles latency on genuine failures."""
    calls = {"n": 0}

    class AlwaysRaisesClient:
        def query(self, *args, **kwargs):
            calls["n"] += 1
            raise ValueError("not a connection problem")

    monkeypatch.setattr(clickhouse, "_new_client", lambda: AlwaysRaisesClient())
    with pytest.raises(ValueError):
        clickhouse.table_count(DATASET_ID, DIMENSION)
    assert calls["n"] == 1  # no retry


@pytest.fixture
def live_corpus():
    if not clickhouse.health():
        pytest.skip("live ClickHouse is not reachable; concurrency evidence is NOT RUN")
    clickhouse._reset_client()
    rows = []
    for index in range(30):
        rows.append({
            "segment_id": f"advret_client_seg_{index:03d}", "dataset_id": DATASET_ID, "video_id": "v",
            "t_start": float(index), "t_end": float(index + 1), "altitude_m": 10.0, "velocity_mps": 1.0,
            "gimbal_pitch": 0.0, "person_count": index % 3, "vehicle_count": 0, "is_night": 0,
            "embedding": _vector(index),
        })
    clickhouse.replace_vectors(DATASET_ID, DIMENSION, rows)
    try:
        yield rows
    finally:
        clickhouse.replace_vectors(DATASET_ID, DIMENSION, [])


@pytest.mark.parametrize("concurrency", [1, 5, 10])
def test_concurrent_live_searches_do_not_cross_contaminate_results_or_settings(live_corpus, concurrency):
    """Each of N concurrent threads searches with a candidate_ids restriction unique to
    that thread and a distinct strategy (varying clickhouse `settings=` per call); if
    session/settings state leaked across threads sharing a connection, results would be
    wrong or an exception would surface. Real evidence against the live container, not a
    mock -- this is exactly the scenario Phase -1.4 needs proof for."""
    strategies = ["exact", "prefilter", "postfilter", "ann"]
    outcomes: list[tuple[int, list[str], list[str]]] = []
    lock = threading.Lock()
    errors = []

    def worker(index: int):
        try:
            candidate_ids = [row["segment_id"] for row in live_corpus[index * 2:index * 2 + 3]]
            strategy = strategies[index % len(strategies)]
            rows, diagnostics = clickhouse.search_vectors(
                DATASET_ID, DIMENSION, _vector(1000 + index), top_k=3, strategy=strategy,
                candidate_ids=candidate_ids, diagnose=True,
            )
            returned_ids = [row["segment_id"] for row in rows]
            with lock:
                outcomes.append((index, candidate_ids, returned_ids))
        except Exception as exc:  # noqa: BLE001 -- collected and asserted below, not swallowed
            with lock:
                errors.append((index, exc))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(outcomes) == concurrency
    for index, candidate_ids, returned_ids in outcomes:
        assert set(returned_ids).issubset(set(candidate_ids)), (
            f"thread {index} got results outside its own candidate set -- "
            "possible cross-thread session/settings contamination"
        )
