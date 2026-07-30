from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app.embedding.cache import CachedEmbeddingStore, _cache_key


def test_t6_query_cache_key_includes_revision_and_text():
    assert _cache_key("r1", "same") != _cache_key("r2", "same")
    assert _cache_key("r1", "same") != _cache_key("r1", "different")


def test_t6_query_cache_writes_are_atomic_and_thread_safe(tmp_path):
    store = CachedEmbeddingStore(tmp_path)
    vector = np.zeros(2048, dtype=np.float32)
    vector[0] = 1.0

    def write(index: int):
        store.put_query(f"text-{index}", "revision", vector)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(20)))
    assert store.query_cache_count() == 20
    for index in range(20):
        np.testing.assert_array_equal(store.query(f"text-{index}", "revision"), vector)
