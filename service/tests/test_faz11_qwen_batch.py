from __future__ import annotations

import sys
import types
from dataclasses import replace

import numpy as np
import pytest

from app.embedding import qwen


class _Result:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self.value


class _Embedder:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def process(self, items):
        self.calls.append(items)
        return _Result(self.result)


def test_embed_videos_uses_one_real_batch_call_and_normalizes(monkeypatch):
    raw = np.vstack([
        np.full(2048, 2.0, dtype=np.float64),
        np.arange(1, 2049, dtype=np.float64),
    ])
    embedder = _Embedder(raw)
    monkeypatch.setattr(qwen, "get_embedder", lambda: embedder)

    result = qwen.embed_videos([["a.jpg", "b.jpg"], ["c.jpg"]])

    assert len(embedder.calls) == 1
    assert embedder.calls[0] == [
        {"video": ["a.jpg", "b.jpg"]}, {"video": ["c.jpg"]},
    ]
    assert result.shape == (2, 2048)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), np.ones(2), atol=1e-6)


def test_embed_video_is_single_item_compatibility_wrapper(monkeypatch):
    expected = np.ones((1, 2048), dtype=np.float32)
    monkeypatch.setattr(qwen, "embed_videos", lambda videos: expected if videos == ["clip.mp4"] else None)
    np.testing.assert_array_equal(qwen.embed_video("clip.mp4"), expected[0])


@pytest.mark.parametrize("raw", [np.zeros((1, 2048)), np.full((1, 2048), np.nan), np.ones((1, 4))])
def test_embed_videos_rejects_invalid_model_output(monkeypatch, raw):
    monkeypatch.setattr(qwen, "get_embedder", lambda: _Embedder(raw))
    with pytest.raises(RuntimeError):
        qwen.embed_videos(["clip.mp4"])


def test_real_embedder_fails_closed_without_cuda(monkeypatch, tmp_path):
    source = tmp_path / "source"
    model = tmp_path / "model"
    source.mkdir()
    model.mkdir()
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(qwen, "settings", replace(qwen.settings, qwen_repo_path=source, qwen_model_path=model))
    qwen.get_embedder.cache_clear()
    with pytest.raises(RuntimeError, match="requires CUDA"):
        qwen.get_embedder()
    qwen.get_embedder.cache_clear()
