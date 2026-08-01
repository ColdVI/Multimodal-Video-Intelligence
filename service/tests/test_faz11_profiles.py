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
    assert configured.default_vector_backend == "clickhouse"
    assert configured.enabled_dimensions == (512,)
    assert configured.filter_execution_mode == "pushdown"


def test_benchmark_profile_preserves_all_backends_and_dimensions():
    configured = _settings(
        ENABLED_VECTOR_BACKENDS="clickhouse,qdrant,pgvector",
        ENABLED_DIMENSIONS="2048,1024,512,256",
    )
    assert configured.enabled_vector_backends == ("clickhouse", "qdrant", "pgvector")
    assert configured.enabled_dimensions == (2048, 1024, 512, 256)


def test_default_vector_backend_must_be_enabled():
    configured = _settings(ENABLED_VECTOR_BACKENDS="qdrant", DEFAULT_VECTOR_BACKEND="qdrant")
    assert configured.default_vector_backend == "qdrant"
    with pytest.raises(ValueError, match="DEFAULT_VECTOR_BACKEND"):
        _settings(ENABLED_VECTOR_BACKENDS="qdrant", DEFAULT_VECTOR_BACKEND="clickhouse")


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


def test_qdrant_secure_profile_does_not_require_clickhouse_credentials():
    configured = Settings.from_env({
        "REQUIRE_SECURE_CREDENTIALS": "true",
        "POSTGRES_PASSWORD": "pg-secret",
        "ENABLED_VECTOR_BACKENDS": "qdrant",
        "DEFAULT_VECTOR_BACKEND": "qdrant",
    })
    assert configured.ch_password == ""


def test_media_and_token_settings_validate_without_exposing_secret():
    configured = _settings(API_TOKEN="secret", MEDIA_URL_TTL_S="60", MEDIA_H264_CRF="24")
    assert configured.api_token == "secret"
    assert "secret" not in repr(configured)
    with pytest.raises(ValueError, match="MEDIA_URL_TTL_S"):
        _settings(MEDIA_URL_TTL_S="0")


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


def test_search_request_uses_profile_backend_and_backend_aware_strategy(monkeypatch):
    monkeypatch.setattr(
        main, "settings", _settings(
            ENABLED_VECTOR_BACKENDS="qdrant", DEFAULT_VECTOR_BACKEND="qdrant",
            ENABLED_DIMENSIONS="512,256",
        )
    )
    observed = {}
    monkeypatch.setattr("app.search.engine.search", lambda request: observed.update(
        backend=request.backend, strategy=request.strategy
    ) or {"ok": True})
    response = main.search(main.SearchRequest(query="traffic", dataset_id="mini"))
    assert response == {"ok": True}
    assert observed == {"backend": "qdrant", "strategy": "ann"}


def test_qdrant_adaptive_exact_rerank_is_rejected_before_search(monkeypatch):
    monkeypatch.setattr(
        main, "settings", _settings(
            ENABLED_VECTOR_BACKENDS="qdrant", DEFAULT_VECTOR_BACKEND="qdrant",
            ENABLED_DIMENSIONS="512,256",
        )
    )
    request = main.SearchRequest(
        query="traffic", dataset_id="mini", dimension=512,
        adaptive_mrl={"enabled": True, "base_dim": 256, "top_n": 10, "exact_rerank": True},
    )
    with pytest.raises(HTTPException, match="exact_rerank") as error:
        main.search(request)
    assert error.value.status_code == 400


def test_request_relaxation_controls_preserve_zero_min_results():
    request = main.SearchRequest(
        query="traffic", dataset_id="mini", min_results=0,
        max_relaxation_passes=2, relaxation_timeout_ms=250, allow_semantic_only_fallback=True,
    )
    assert request.min_results == 0
    assert request.max_relaxation_passes == 2
    assert request.relaxation_timeout_ms == 250
    assert request.allow_semantic_only_fallback is True
