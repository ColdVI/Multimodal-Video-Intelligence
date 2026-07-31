from __future__ import annotations

import pytest
from fastapi import HTTPException

from app import main
from app.config import Settings


def _settings(**overrides: str):
    env = {
        "POSTGRES_PASSWORD": "test-pg", "CLICKHOUSE_PASSWORD": "test-ch",
        "ENABLED_DIMENSIONS": "2048,1024,512,256",
    }
    env.update(overrides)
    return Settings.from_env(env)


def _request(**overrides):
    payload = {
        "query": "traffic", "dataset_id": "mini", "backend": "clickhouse",
        "dimension": 512, "adaptive_mrl": {"enabled": True, "base_dim": 256, "top_n": 100},
        "top_k": 10,
    }
    payload.update(overrides)
    return main.SearchRequest(**payload)


@pytest.mark.parametrize("base_dim,dimension", sorted(main.ADAPTIVE_MRL_ALLOWED_PAIRS))
def test_every_allow_listed_pair_passes_validation_before_reaching_the_engine(monkeypatch, base_dim, dimension):
    monkeypatch.setattr(main, "settings", _settings())
    called = {}

    def fake_run_search(request):
        called["ran"] = True
        return {"ok": True}

    monkeypatch.setattr("app.search.engine.search", fake_run_search)
    request = _request(dimension=dimension, adaptive_mrl={"enabled": True, "base_dim": base_dim, "top_n": 100})
    main.search(request)
    assert called.get("ran") is True


def test_base_dim_not_in_enabled_dimensions_is_rejected(monkeypatch):
    monkeypatch.setattr(main, "settings", _settings(ENABLED_DIMENSIONS="512"))
    request = _request(dimension=512, adaptive_mrl={"enabled": True, "base_dim": 256, "top_n": 100})
    with pytest.raises(HTTPException) as error:
        main.search(request)
    assert error.value.status_code == 400
    assert "base_dim" in str(error.value.detail)
    assert "disabled" in str(error.value.detail)


def test_base_dim_greater_than_or_equal_to_dimension_is_rejected(monkeypatch):
    monkeypatch.setattr(main, "settings", _settings())
    request = _request(dimension=256, adaptive_mrl={"enabled": True, "base_dim": 512, "top_n": 100})
    with pytest.raises(HTTPException) as error:
        main.search(request)
    assert error.value.status_code == 400
    assert "must be smaller than" in str(error.value.detail)


def test_base_dim_equal_to_dimension_is_rejected(monkeypatch):
    monkeypatch.setattr(main, "settings", _settings())
    request = _request(dimension=512, adaptive_mrl={"enabled": True, "base_dim": 512, "top_n": 100})
    with pytest.raises(HTTPException) as error:
        main.search(request)
    assert error.value.status_code == 400
    assert "must be smaller than" in str(error.value.detail)


def test_top_n_below_top_k_is_rejected(monkeypatch):
    monkeypatch.setattr(main, "settings", _settings())
    request = _request(
        dimension=512, top_k=50, adaptive_mrl={"enabled": True, "base_dim": 256, "top_n": 10},
    )
    with pytest.raises(HTTPException) as error:
        main.search(request)
    assert error.value.status_code == 400
    assert "top_n" in str(error.value.detail)


def test_adaptive_disabled_skips_all_new_validation(monkeypatch):
    """Non-adaptive requests must be completely unaffected by this validation, even with
    a nonsensical base_dim/top_n combination sitting unused in the request body."""
    monkeypatch.setattr(main, "settings", _settings())
    called = {}
    monkeypatch.setattr("app.search.engine.search", lambda request: called.setdefault("ran", True) or {"ok": True})
    request = _request(
        dimension=256, adaptive_mrl={"enabled": False, "base_dim": 512, "top_n": 1}, top_k=999,
    )
    main.search(request)
    assert called.get("ran") is True
