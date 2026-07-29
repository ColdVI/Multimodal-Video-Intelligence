import pytest

from dataset_adapters import registry


def test_dataset_configs_validates_required_fields(monkeypatch):
    monkeypatch.setattr(registry, "load_config", lambda path="config.yaml": {
        "datasets": {"broken": {"adapter": "x"}}  # eksik alanlar
    })
    with pytest.raises(ValueError, match="eksik"):
        registry.dataset_configs()


def test_dataset_config_unknown_id_raises_keyerror(monkeypatch):
    monkeypatch.setattr(registry, "load_config", lambda path="config.yaml": {"datasets": {}})
    with pytest.raises(KeyError):
        registry.dataset_config("nope")


def test_dataset_configs_real_config_has_visdrone_and_msrvtt():
    configs = registry.dataset_configs()
    assert "visdrone" in configs
    assert "msrvtt_1ka" in configs
    assert configs["visdrone"]["query_count"] == 28
    assert configs["msrvtt_1ka"]["query_count"] == 1000


def test_supports_strategy_clickhouse_backend_supports_everything():
    assert registry.supports_strategy("visdrone", "prefilter")
    assert registry.supports_strategy("visdrone", "postfilter_rescore")
    assert registry.supports_strategy("visdrone", "telemetry_filter")


def test_supports_strategy_artifact_matrix_rejects_filter_dependent_strategies():
    assert not registry.supports_strategy("msrvtt_1ka", "prefilter")
    assert not registry.supports_strategy("msrvtt_1ka", "postfilter_rescore")
    assert not registry.supports_strategy("msrvtt_1ka", "telemetry_filter")
    assert registry.supports_strategy("msrvtt_1ka", "auto")
    assert registry.supports_strategy("msrvtt_1ka", "exact")


def test_run_strategy_or_unsupported_calls_run_fn_when_supported():
    calls = []
    result = registry.run_strategy_or_unsupported(
        "visdrone", "prefilter", lambda: calls.append(1) or {"ok": True})
    assert result == {"ok": True}
    assert calls == [1]


def test_run_strategy_or_unsupported_never_calls_run_fn_when_not_supported():
    calls = []
    result = registry.run_strategy_or_unsupported(
        "msrvtt_1ka", "prefilter", lambda: calls.append(1) or {"ok": True})
    assert result["unsupported_strategy"] is True
    assert result["dataset_id"] == "msrvtt_1ka"
    assert result["strategy"] == "prefilter"
    assert calls == []  # run_fn HIC cagrilmadi - sessiz fallback yok
