from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import auth
from app.config import Settings


def _app(token: str) -> TestClient:
    app = FastAPI()
    app.add_middleware(auth.TokenAuthMiddleware, token=token)

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/private")
    def private():
        return {"ok": True}

    @app.get("/media/{segment_id}")
    def media(segment_id: str):
        return {"segment_id": segment_id}

    return TestClient(app)


def test_empty_token_preserves_local_development_compatibility():
    assert _app("").get("/private").status_code == 200


def test_health_open_and_private_routes_require_bearer_token():
    client = _app("secret")
    assert client.get("/health").status_code == 200
    unauthorized = client.get("/private")
    assert unauthorized.status_code == 401
    assert unauthorized.headers["www-authenticate"] == "Bearer"
    assert client.get("/private", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/private", headers={"Authorization": "Bearer secret"}).status_code == 200


def test_signed_media_url_is_scoped_and_expires(monkeypatch):
    monkeypatch.setattr(auth, "settings", SimpleNamespace(api_token="secret", media_url_ttl_s=300))
    monkeypatch.setattr(auth.time, "time", lambda: 1000)
    url = auth.signed_media_url("dataset:video:0.000:8.000", "run-1", now=1000)
    client = _app("secret")
    assert client.get(url).status_code == 200
    app = FastAPI()
    request_scope = {
        "type": "http", "method": "GET", "scheme": "http", "path": "/media/dataset:video:0.000:8.000",
        "raw_path": b"/media/dataset%3Avideo%3A0.000%3A8.000", "query_string": url.split("?", 1)[1].encode(),
        "headers": [], "client": ("test", 1), "server": ("test", 80), "root_path": "", "http_version": "1.1",
    }
    from fastapi import Request

    request = Request(request_scope)
    assert auth.valid_media_signature(request, "secret", now=1300)
    assert not auth.valid_media_signature(request, "secret", now=1301)


def test_api_token_is_not_exposed_in_settings_repr():
    configured = Settings.from_env({"API_TOKEN": "top-secret"})
    assert "top-secret" not in repr(configured)


def test_signed_media_url_does_not_contain_api_token(monkeypatch):
    monkeypatch.setattr(auth, "settings", SimpleNamespace(api_token="top-secret-bearer-token", media_url_ttl_s=300))
    monkeypatch.setattr(auth.time, "time", lambda: 1000)
    url = auth.signed_media_url("dataset:video:0.000:8.000", "run-1", now=1000)
    assert "top-secret-bearer-token" not in url
    assert "signature=" in url and "expires=" in url


def test_signed_media_url_signature_is_useless_for_other_paths(monkeypatch):
    """A signature is bound to the exact request path; it must not authorize a
    different /media/{other_segment} or a non-media endpoint."""
    monkeypatch.setattr(auth, "settings", SimpleNamespace(api_token="secret", media_url_ttl_s=300))
    monkeypatch.setattr(auth.time, "time", lambda: 1000)
    url = auth.signed_media_url("segment-a", "run-1", now=1000)
    query_string = url.split("?", 1)[1].encode()
    from fastapi import Request

    other_path_scope = {
        "type": "http", "method": "GET", "scheme": "http", "path": "/media/segment-b",
        "raw_path": b"/media/segment-b", "query_string": query_string,
        "headers": [], "client": ("test", 1), "server": ("test", 80), "root_path": "", "http_version": "1.1",
    }
    assert not auth.valid_media_signature(Request(other_path_scope), "secret", now=1000)
    non_media_scope = {**other_path_scope, "path": "/stats", "raw_path": b"/stats"}
    assert not auth.valid_media_signature(Request(non_media_scope), "secret", now=1000)
