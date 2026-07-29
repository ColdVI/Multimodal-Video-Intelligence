"""embed_all_videos()'un onbellek/resume davranisini GERCEK yerel MSR-VTT
videolarina karsi test eder (embedder sahtelenir - gercek Qwen modeli
gerektirmez, ama video I/O gercektir)."""
import pathlib

import numpy as np
import pytest

from scripts import validate_msrvtt

_HAS_MSRVTT = pathlib.Path("data/downloads/msrvtt/msrvtt_test_1k.json").exists()
pytestmark = pytest.mark.skipif(not _HAS_MSRVTT, reason="MSR-VTT indirilmemis")


class _FakeEmbedder:
    def __init__(self):
        self.embed_video_calls = []

    def embed_video(self, frames):
        self.embed_video_calls.append(1)
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def embed_text(self, text):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


@pytest.fixture
def two_entries():
    entries = validate_msrvtt.load_test_split("data/downloads/msrvtt/msrvtt_test_1k.json")[:2]
    return entries


def test_embed_all_videos_without_cache_file_behaves_as_before(monkeypatch, two_entries):
    fake = _FakeEmbedder()
    monkeypatch.setattr(validate_msrvtt, "get_embedder", lambda name: fake)
    out = validate_msrvtt.embed_all_videos(
        two_entries, pathlib.Path("data/downloads/msrvtt/videos"), "qwen3vl_emb_2048")
    assert len(out) == 2
    assert len(fake.embed_video_calls) == 2


def test_embed_all_videos_with_cache_file_resumes_and_skips_cached(monkeypatch, two_entries, tmp_path):
    cache_file = tmp_path / "cache.ndjson"
    already_done_id = two_entries[0]["video_id"]
    validate_msrvtt.append_cached(cache_file, already_done_id, [9.0, 9.0, 9.0])

    fake = _FakeEmbedder()
    monkeypatch.setattr(validate_msrvtt, "get_embedder", lambda name: fake)
    out = validate_msrvtt.embed_all_videos(
        two_entries, pathlib.Path("data/downloads/msrvtt/videos"), "qwen3vl_emb_2048",
        cache_file=cache_file)

    assert len(out) == 2
    # onbellekten gelen deger DEGISMEDI (yeniden embed edilmedi)
    assert list(out[already_done_id]) == [9.0, 9.0, 9.0]
    # sadece 1 GERCEK embed_video cagrisi yapildi (2. video onbellekte degildi)
    assert len(fake.embed_video_calls) == 1


def test_embed_all_videos_writes_new_results_incrementally_to_cache(monkeypatch, two_entries, tmp_path):
    cache_file = tmp_path / "cache.ndjson"
    fake = _FakeEmbedder()
    monkeypatch.setattr(validate_msrvtt, "get_embedder", lambda name: fake)
    validate_msrvtt.embed_all_videos(
        two_entries, pathlib.Path("data/downloads/msrvtt/videos"), "qwen3vl_emb_2048",
        cache_file=cache_file)

    reloaded = validate_msrvtt.load_cached(cache_file)
    assert set(reloaded) == {e["video_id"] for e in two_entries}


def test_qwen_cache_file_is_deterministic_for_same_inputs():
    p1 = validate_msrvtt.qwen_cache_file("qwen3vl_emb_2048", "unused", n_frames=32)
    p2 = validate_msrvtt.qwen_cache_file("qwen3vl_emb_2048", "unused", n_frames=32)
    assert p1 == p2


def test_qwen_cache_file_differs_when_n_frames_differs():
    p1 = validate_msrvtt.qwen_cache_file("qwen3vl_emb_2048", "unused", n_frames=32)
    p2 = validate_msrvtt.qwen_cache_file("qwen3vl_emb_2048", "unused", n_frames=16)
    assert p1 != p2


def test_qwen_cache_file_differs_between_native_and_truncated_dimension_source():
    p_2048 = validate_msrvtt.qwen_cache_file("qwen3vl_emb_2048", "unused", n_frames=32)
    p_512 = validate_msrvtt.qwen_cache_file("qwen3vl_emb_512", "unused", n_frames=32)
    assert p_2048 != p_512
