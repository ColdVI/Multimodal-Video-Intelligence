from scripts.msrvtt_embedding_cache import append_cached, cache_key, load_cached


def test_cache_key_deterministic_for_identical_inputs():
    args = dict(dataset_id="msrvtt_1ka", split="1k-A test", dataset_hash="abc",
               model_id="Qwen/Qwen3-VL-Embedding-2B", model_revision="unknown",
               n_sample=6, frame_sampling_version="v1", dtype="float32",
               dimension_source="native_2048")
    assert cache_key(**args) == cache_key(**args)


def test_cache_key_changes_when_dataset_hash_changes():
    base = dict(dataset_id="msrvtt_1ka", split="1k-A test", dataset_hash="abc",
               model_id="Qwen/Qwen3-VL-Embedding-2B", model_revision="unknown",
               n_sample=6, frame_sampling_version="v1", dtype="float32",
               dimension_source="native_2048")
    changed = {**base, "dataset_hash": "xyz"}
    assert cache_key(**base) != cache_key(**changed)


def test_cache_key_changes_when_model_revision_changes():
    base = dict(dataset_id="msrvtt_1ka", split="1k-A test", dataset_hash="abc",
               model_id="Qwen/Qwen3-VL-Embedding-2B", model_revision="rev1",
               n_sample=6, frame_sampling_version="v1", dtype="float32",
               dimension_source="native_2048")
    changed = {**base, "model_revision": "rev2"}
    assert cache_key(**base) != cache_key(**changed)


def test_load_cached_missing_file_returns_empty_dict(tmp_path):
    assert load_cached(tmp_path / "does_not_exist.ndjson") == {}


def test_append_then_load_round_trips(tmp_path):
    path = tmp_path / "cache.ndjson"
    append_cached(path, "video1", [0.1, 0.2, 0.3])
    append_cached(path, "video2", [0.4, 0.5, 0.6])
    loaded = load_cached(path)
    assert loaded == {"video1": [0.1, 0.2, 0.3], "video2": [0.4, 0.5, 0.6]}


def test_load_cached_skips_corrupt_trailing_line_but_keeps_earlier_ones(tmp_path):
    path = tmp_path / "cache.ndjson"
    append_cached(path, "video1", [0.1, 0.2])
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"video_id": "video2", "embed')  # crash ortasinda kesilmis satir
    loaded = load_cached(path)
    assert loaded == {"video1": [0.1, 0.2]}


def test_append_is_incremental_not_buffered(tmp_path):
    path = tmp_path / "cache.ndjson"
    append_cached(path, "video1", [0.1])
    assert load_cached(path) == {"video1": [0.1]}  # ilk yazimdan hemen sonra okunabilir
    append_cached(path, "video2", [0.2])
    assert load_cached(path) == {"video1": [0.1], "video2": [0.2]}
