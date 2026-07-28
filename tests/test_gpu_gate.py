import pytest

from bench import gpu_gate


def test_require_gpu_raises_when_no_cuda(monkeypatch):
    monkeypatch.setattr(gpu_gate.torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as exc:
        gpu_gate.require_gpu("test-task", cpu_estimate_h=17.7, gpu_estimate_min=41)
    msg = str(exc.value)
    assert "test-task" in msg
    assert "41" in msg
    assert "17.7" in msg


def test_require_gpu_passes_silently_when_cuda_available(monkeypatch):
    monkeypatch.setattr(gpu_gate.torch.cuda, "is_available", lambda: True)
    gpu_gate.require_gpu("test-task", cpu_estimate_h=17.7, gpu_estimate_min=41)  # raise atmamali


def test_require_gpu_for_qwen_windows_uses_real_measured_rates(monkeypatch):
    monkeypatch.setattr(gpu_gate.torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as exc:
        # 73 pencere, TASKS.md'deki gercek 73-pencere/1062-dakika olcumunu yeniden uretmeli
        gpu_gate.require_gpu_for_qwen_windows("ingest test", n_windows=73)
    msg = str(exc.value)
    assert "17.7" in msg  # 73 * 870s / 3600 = 17.65 saat ~ 17.7
    assert "3" in msg  # 73 * 2.443s / 60 = ~2.97 dk -> "3 dk" (yuvarlanmis)


def test_require_gpu_for_qwen_windows_zero_windows_is_noop_when_gpu_present(monkeypatch):
    monkeypatch.setattr(gpu_gate.torch.cuda, "is_available", lambda: True)
    gpu_gate.require_gpu_for_qwen_windows("noop test", n_windows=0)  # raise atmamali
