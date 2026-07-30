from scripts.preflight import _api_exposure_safe, _combined_exit_code, _read_env


def test_env_reader_loads_values_without_overwriting_process_env(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("A=from-file\nB=second\n", encoding="utf-8")
    monkeypatch.setenv("A", "from-process")
    values = _read_env(path)
    assert values["A"] == "from-process"
    assert values["B"] == "second"


def test_preflight_exit_codes_are_stable():
    base = {"data_preflight": {"checks": []}}
    assert _combined_exit_code({**base, "checks": []}) == 0
    for category, expected in (
        ("configuration", 2), ("data", 3), ("gpu", 4), ("model", 5), ("resources", 6),
    ):
        report = {
            **base,
            "checks": [{"category": category, "status": "fail"}],
        }
        assert _combined_exit_code(report) == expected


def test_missing_env_file_is_not_created(tmp_path):
    path = tmp_path / "missing.env"
    _read_env(path)
    assert not path.exists()


def test_public_api_bind_requires_optional_token_to_be_configured():
    assert _api_exposure_safe({"BIND_HOST": "127.0.0.1", "API_TOKEN": ""})
    assert not _api_exposure_safe({"BIND_HOST": "0.0.0.0", "API_TOKEN": ""})
    assert _api_exposure_safe({"BIND_HOST": "0.0.0.0", "API_TOKEN": "secret"})
