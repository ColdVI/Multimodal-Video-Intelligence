from __future__ import annotations

import hashlib
import hmac
import time
from urllib.parse import quote, urlencode

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.config import settings


def _signature(path: str, run_id: str | None, expires: int, token: str) -> str:
    payload = f"{path}\n{run_id or ''}\n{expires}".encode("utf-8")
    return hmac.new(token.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def signed_media_url(segment_id: str, run_id: str | None = None, *, now: float | None = None) -> str:
    path = f"/media/{segment_id}"
    encoded_path = f"/media/{quote(segment_id, safe='')}"
    query: dict[str, str | int] = {}
    if run_id:
        query["run_id"] = run_id
    if settings.api_token:
        expires = int(time.time() if now is None else now) + settings.media_url_ttl_s
        query["expires"] = expires
        query["signature"] = _signature(path, run_id, expires, settings.api_token)
    return encoded_path if not query else f"{encoded_path}?{urlencode(query)}"


def valid_media_signature(request: Request, token: str, *, now: float | None = None) -> bool:
    if request.method != "GET" or not request.url.path.startswith("/media/") or request.url.path.endswith("/info"):
        return False
    expires_raw = request.query_params.get("expires")
    supplied = request.query_params.get("signature")
    if not expires_raw or not supplied:
        return False
    try:
        expires = int(expires_raw)
    except ValueError:
        return False
    current = int(time.time() if now is None else now)
    if expires < current:
        return False
    expected = _signature(request.url.path, request.query_params.get("run_id"), expires, token)
    return hmac.compare_digest(supplied, expected)


class TokenAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, token: str):
        super().__init__(app)
        self.token = token

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not self.token or request.url.path == "/health":
            return await call_next(request)
        authorization = request.headers.get("authorization", "")
        supplied = authorization[7:] if authorization.startswith("Bearer ") else ""
        if (supplied and hmac.compare_digest(supplied, self.token)) or valid_media_signature(request, self.token):
            return await call_next(request)
        return JSONResponse(
            status_code=401,
            content={"detail": "valid Bearer token required"},
            headers={"WWW-Authenticate": "Bearer"},
        )


__all__ = ["TokenAuthMiddleware", "signed_media_url", "valid_media_signature"]
