from __future__ import annotations

import json
import sys
import types

from scripts import gpu_smoke


def test_gpu_smoke_writes_honest_not_run_without_cuda(tmp_path, monkeypatch):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        version=types.SimpleNamespace(cuda=None),
        __version__="fixture",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(gpu_smoke, "_driver_version", lambda: None)
    output = tmp_path / "gpu_smoke.json"
    payload, code = gpu_smoke.run(tmp_path / "dataset.yaml", tmp_path, output)
    assert code == 4
    assert payload["status"] == "not_run"
    assert payload["windows_embedded"] == 0
    assert "required_command" in payload and "expected_environment" in payload
    assert json.loads(output.read_text(encoding="utf-8"))["result"] == "not_run"
