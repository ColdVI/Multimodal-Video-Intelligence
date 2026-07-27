from scripts.dtype_arch_probe import _time_calls, probe_hardware


def test_probe_hardware_on_cpu_returns_warning_without_crashing():
    info = probe_hardware()
    assert info["cuda_available"] is False
    assert "warning" in info
    # GPU-only alanlar CPU'da hic yazilmamali (yanlis/None deger yazip
    # olceklendirmek yerine acikca yok olmalilar)
    assert "compute_capability" not in info


def test_time_calls_returns_median_and_p95_for_normal_function():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1

    result = _time_calls(fn, warmup=2, n=5)
    assert calls["n"] == 7  # 2 warmup + 5 olcum
    assert "median_s" in result and "p95_s" in result
    assert result["n"] == 5


def test_time_calls_catches_exception_without_raising():
    def broken():
        raise RuntimeError("simulated failure")

    result = _time_calls(broken, warmup=1, n=1)
    assert "error" in result
    assert "simulated failure" in result["error"]
