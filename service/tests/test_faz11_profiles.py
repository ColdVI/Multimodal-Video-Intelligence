from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from fastapi import HTTPException

from app import main
from app.config import Settings


def _settings(**overrides: str) -> Settings:
    env = {"POSTGRES_PASSWORD": "test-pg", "CLICKHOUSE_PASSWORD": "test-ch"}
    env.update(overrides)
    return Settings.from_env(env)


def test_institution_profile_is_the_default():
    configured = Settings.from_env({})
    assert configured.enabled_vector_backends == ("clickhouse",)
    assert configured.enabled_dimensions == (512,)
    assert configured.filter_execution_mode == "pushdown"


def test_benchmark_profile_preserves_all_backends_and_dimensions():
    configured = _settings(
        ENABLED_VECTOR_BACKENDS="clickhouse,qdrant,pgvector",
        ENABLED_DIMENSIONS="2048,1024,512,256",
    )
    assert configured.enabled_vector_backends == ("clickhouse", "qdrant", "pgvector")
    assert configured.enabled_dimensions == (2048, 1024, 512, 256)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("ENABLED_VECTOR_BACKENDS", "clickhouse,unknown", "unknown ENABLED_VECTOR_BACKENDS"),
        ("ENABLED_DIMENSIONS", "4096", "unsupported ENABLED_DIMENSIONS"),
        ("FILTER_EXECUTION_MODE", "python_ids", "FILTER_EXECUTION_MODE"),
    ],
)
def test_profile_validation_fails_fast(name, value, message):
    with pytest.raises(ValueError, match=message):
        _settings(**{name: value})


def test_secure_runtime_rejects_empty_credentials():
    with pytest.raises(ValueError, match="POSTGRES_PASSWORD, CLICKHOUSE_PASSWORD"):
        Settings.from_env({"REQUIRE_SECURE_CREDENTIALS": "true"})


def test_api_import_does_not_require_capera_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"datasets": {"auair": {"enabled": True}}}), encoding="utf-8")
    env = os.environ.copy()
    env["PROJECT_CONFIG_PATH"] = str(config_path)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-c", "import app.main; print(app.main.app.version)"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "11.0.0"


def test_strategies_exposes_only_enabled_profile(monkeypatch):
    monkeypatch.setattr(main, "settings", _settings())
    response = main.strategies()
    assert response["enabled_backends"] == ["clickhouse"]
    assert response["enabled_dimensions"] == [512]
    assert set(response["strategies"]) == {"clickhouse"}


def test_health_does_not_probe_disabled_backends(monkeypatch):
    monkeypatch.setattr(main, "settings", _settings())
    monkeypatch.setattr(main.postgres, "health", lambda: True)
    monkeypatch.setattr(main, "enabled_health", lambda: {"clickhouse": True})
    response = main.health()
    assert response["status"] == "ok"
    assert response["vector_backends"] == {"clickhouse": True}
    assert response["disabled_backends"] == ["qdrant", "pgvector"]
    assert "qdrant" not in response


def test_search_rejects_disabled_backend_before_db_access(monkeypatch):
    monkeypatch.setattr(main, "settings", _settings())
    request = main.SearchRequest(query="traffic", dataset_id="mini", backend="qdrant")
    with pytest.raises(HTTPException) as error:
        main.search(request)
    assert error.value.status_code == 400
    assert "disabled" in str(error.value.detail)


def test_search_rejects_disabled_dimension_before_db_access(monkeypatch):
    monkeypatch.setattr(main, "settings", _settings())
    request = main.SearchRequest(query="traffic", dataset_id="mini", dimension=2048)
    with pytest.raises(HTTPException) as error:
        main.search(request)
    assert error.value.status_code == 400
    assert "disabled" in str(error.value.detail)
