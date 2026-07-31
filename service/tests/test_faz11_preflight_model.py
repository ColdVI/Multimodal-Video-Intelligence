from __future__ import annotations

import sys
import types

from app.preflight import _append_model_checks, _flash_attention_check
from app.config import Settings


def _settings(**overrides):
    base = {"EMBEDDING_MODE": "real", "MODEL_BUNDLE_ROOT": "/nonexistent-bundle-root-for-tests"}
    base.update(overrides)
    return Settings.from_env(base)


def test_flash_attention_check_fails_when_package_missing(monkeypatch):
    monkeypatch.delitem(sys.modules, "flash_attn", raising=False)
    real_import = __import__

    def blocked_import(name, *args, **kwargs):
        if name == "flash_attn":
            raise ImportError("no module named flash_attn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked_import)
    check = _flash_attention_check()
    assert check.status == "fail"
    assert check.category == "gpu"
    assert "flash_attn not importable" in check.detail


def test_flash_attention_check_fails_on_low_compute_capability(monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_attn", types.SimpleNamespace())
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True, get_device_capability=lambda: (7, 5)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    check = _flash_attention_check()
    assert check.status == "fail"
    assert "compute_capability=(7, 5)" in check.detail


def test_flash_attention_check_fails_without_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_attn", types.SimpleNamespace())
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    check = _flash_attention_check()
    assert check.status == "fail"
    assert "CUDA is not available" in check.detail


def test_flash_attention_check_passes_when_supported(monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_attn", types.SimpleNamespace())
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True, get_device_capability=lambda: (8, 0)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    check = _flash_attention_check()
    assert check.status == "pass"


def test_sdpa_selection_never_triggers_flash_attention_check():
    checks: list = []
    _append_model_checks(checks, _settings(ATTN_IMPL="sdpa"))
    assert not any(item.check_id == "flash_attention_2_support" for item in checks)


def test_flash_attention_selection_appends_check_even_without_bundle():
    checks: list = []
    _append_model_checks(checks, _settings(ATTN_IMPL="flash_attention_2"))
    assert any(item.check_id == "flash_attention_2_support" for item in checks)
