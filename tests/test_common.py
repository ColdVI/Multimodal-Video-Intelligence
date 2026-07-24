import common


def test_offline_mode_enabled_respects_hf_hub_offline_env(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert common.offline_mode_enabled() is True


def test_offline_mode_enabled_respects_config_flag(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(common, "load_config", lambda path="config.yaml": {"offline_mode": True})
    assert common.offline_mode_enabled() is True


def test_offline_mode_disabled_by_default(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(common, "load_config", lambda path="config.yaml": {"offline_mode": False})
    assert common.offline_mode_enabled() is False


def test_offline_mode_env_overrides_config_flag(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(common, "load_config", lambda path="config.yaml": {"offline_mode": False})
    assert common.offline_mode_enabled() is True
