"""Additive FastMCP HTTP transport settings.

stdio remains the default. HTTP binds loopback only by default and fail-closes
unless GROK_SEARCH_MCP_TOKEN or (GROK_SEARCH_MCP_VERIFY_URL and GROK_SEARCH_MCP_INTERNAL_TOKEN)
is set. This token is never taken from GUDA_API_KEY / GROK_API_KEY / TAVILY_API_KEY / FIRECRAWL_API_KEY.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import httpx
from fastmcp.server.auth import AccessToken, TokenVerifier
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier

Transport = Literal["stdio", "http"]

DEFAULT_TRANSPORT: Transport = "stdio"
DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8800
DEFAULT_HTTP_PATH = "/mcp"

DEFAULT_ALLOWED_HOSTS: tuple[str, ...] = (
    "search.karldigi.dev",
    "127.0.0.1",
    "localhost",
)

DEFAULT_UVICORN_CONFIG: dict[str, Any] = {
    "forwarded_allow_ips": "127.0.0.1",
    "proxy_headers": True,
}

_ENV_TRANSPORT = "GROK_SEARCH_MCP_TRANSPORT"
_ENV_HOST = "GROK_SEARCH_MCP_HOST"
_ENV_PORT = "GROK_SEARCH_MCP_PORT"
_ENV_PATH = "GROK_SEARCH_MCP_PATH"
_ENV_TOKEN = "GROK_SEARCH_MCP_TOKEN"
_ENV_VERIFY_URL = "GROK_SEARCH_MCP_VERIFY_URL"
_ENV_INTERNAL_TOKEN = "GROK_SEARCH_MCP_INTERNAL_TOKEN"

class McpHttpConfigError(ValueError):
    """Raised when HTTP MCP is requested with an invalid or missing setup."""


@dataclass(frozen=True)
class McpHttpBind:
    host: str
    port: int
    path: str


class GatewayTokenVerifier(TokenVerifier):
    """TokenVerifier that validates tokens against an upstream loopback gateway."""

    def __init__(
        self,
        verify_url: str,
        internal_token: str,
        required_scopes: list[str] | None = None,
        timeout: float = 3.0,
        negative_cache_ttl: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        super().__init__(required_scopes=required_scopes)
        self.verify_url = verify_url
        self.internal_token = internal_token
        self.timeout = timeout
        self.negative_cache_ttl = negative_cache_ttl
        self._negative_cache: dict[str, float] = {}
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            transport=transport,
        )

    def _token_hash(self, token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    async def verify_token(self, token: str) -> AccessToken | None:
        token_hash = self._token_hash(token)
        now = time.time()

        # Check negative cache (never store raw token)
        expiry = self._negative_cache.get(token_hash)
        if expiry is not None:
            if expiry > now:
                return None
            self._negative_cache.pop(token_hash, None)

        try:
            response = await self._client.post(
                self.verify_url,
                headers={"X-Internal-Token": self.internal_token},
                json={"token": token},
            )
        except (httpx.RequestError, Exception):
            # Gateway blip must not lock out good keys -> do not negatively cache
            return None

        if response.status_code == 200:
            try:
                data = response.json()
            except Exception:
                return None

            if not isinstance(data, dict):
                return None

            client_id = data.get("client_id") or data.get("name") or "gateway-key"
            raw_scopes = data.get("scopes")
            scopes = raw_scopes if isinstance(raw_scopes, list) else ["mcp"]

            if self.required_scopes:
                token_scopes = set(scopes)
                required = set(self.required_scopes)
                if not required.issubset(token_scopes):
                    return None

            return AccessToken(
                token=token,
                client_id=str(client_id),
                scopes=scopes,
                claims=data,
            )

        if response.status_code == 401:
            self._negative_cache[token_hash] = now + self.negative_cache_ttl
            return None

        # 403, 5xx, or other status codes: return None without negative-caching
        return None


def _env(environ: Mapping[str, str] | None, name: str, default: str | None = None) -> str | None:
    source = os.environ if environ is None else environ
    if name not in source:
        return default
    value = source[name]
    if value is None:
        return default
    stripped = value.strip()
    return stripped if stripped else default


def resolve_transport(environ: Mapping[str, str] | None = None) -> Transport:
    raw = _env(environ, _ENV_TRANSPORT, DEFAULT_TRANSPORT)
    value = (raw or DEFAULT_TRANSPORT).lower()
    if value not in ("stdio", "http"):
        raise McpHttpConfigError(
            f"{_ENV_TRANSPORT} must be 'stdio' or 'http', got {raw!r}"
        )
    return value  # type: ignore[return-value]


def resolve_http_bind(environ: Mapping[str, str] | None = None) -> McpHttpBind:
    host = _env(environ, _ENV_HOST, DEFAULT_HTTP_HOST) or DEFAULT_HTTP_HOST
    port_raw = _env(environ, _ENV_PORT, str(DEFAULT_HTTP_PORT)) or str(DEFAULT_HTTP_PORT)
    try:
        port = int(port_raw)
    except ValueError as exc:
        raise McpHttpConfigError(
            f"{_ENV_PORT} must be an integer, got {port_raw!r}"
        ) from exc
    if not (1 <= port <= 65535):
        raise McpHttpConfigError(f"{_ENV_PORT} out of range: {port}")

    path = _env(environ, _ENV_PATH, DEFAULT_HTTP_PATH) or DEFAULT_HTTP_PATH
    if not path.startswith("/"):
        path = f"/{path}"
    return McpHttpBind(host=host, port=port, path=path)


def require_http_token(environ: Mapping[str, str] | None = None) -> str:
    """Return the inbound HTTP bearer. Fail closed; never fall back to other keys."""
    token = _env(environ, _ENV_TOKEN, None)
    if not token:
        raise McpHttpConfigError(
            f"{_ENV_TOKEN} is required when {_ENV_TRANSPORT}=http "
            "(fail closed). Do not reuse GUDA_API_KEY, GROK_API_KEY, "
            "TAVILY_API_KEY, or FIRECRAWL_API_KEY."
        )
    return token


def build_static_token_verifier(token: str) -> StaticTokenVerifier:
    return StaticTokenVerifier(
        tokens={
            token: {
                "client_id": "grok-search-http",
                "scopes": ["mcp"],
            }
        }
    )


def build_gateway_token_verifier(
    verify_url: str,
    internal_token: str,
    required_scopes: list[str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> GatewayTokenVerifier:
    return GatewayTokenVerifier(
        verify_url=verify_url,
        internal_token=internal_token,
        required_scopes=required_scopes,
        transport=transport,
    )


def apply_http_auth(
    mcp,
    auth_or_token: str | TokenVerifier,
) -> TokenVerifier:
    if isinstance(auth_or_token, TokenVerifier):
        verifier = auth_or_token
    else:
        verifier = build_static_token_verifier(auth_or_token)
    mcp.auth = verifier
    return verifier


@dataclass(frozen=True)
class McpRunSettings:
    transport: Transport
    host: str | None = None
    port: int | None = None
    path: str | None = None
    token: str | None = None
    verify_url: str | None = None
    internal_token: str | None = None
    allowed_hosts: tuple[str, ...] | None = None
    uvicorn_config: dict[str, Any] | None = None


def resolve_run_settings(environ: Mapping[str, str] | None = None) -> McpRunSettings:
    transport = resolve_transport(environ)
    if transport == "stdio":
        return McpRunSettings(transport="stdio")

    bind = resolve_http_bind(environ)
    verify_url = _env(environ, _ENV_VERIFY_URL, None)
    if verify_url:
        internal_token = _env(environ, _ENV_INTERNAL_TOKEN, None)
        if not internal_token:
            raise McpHttpConfigError(
                f"{_ENV_INTERNAL_TOKEN} is required when {_ENV_VERIFY_URL} is set (fail closed)."
            )
        return McpRunSettings(
            transport="http",
            host=bind.host,
            port=bind.port,
            path=bind.path,
            token=None,
            verify_url=verify_url,
            internal_token=internal_token,
            allowed_hosts=DEFAULT_ALLOWED_HOSTS,
            uvicorn_config=DEFAULT_UVICORN_CONFIG,
        )

    token = require_http_token(environ)
    return McpRunSettings(
        transport="http",
        host=bind.host,
        port=bind.port,
        path=bind.path,
        token=token,
        verify_url=None,
        internal_token=None,
        allowed_hosts=DEFAULT_ALLOWED_HOSTS,
        uvicorn_config=DEFAULT_UVICORN_CONFIG,
    )


def run_mcp(mcp, settings: McpRunSettings | None = None, environ: Mapping[str, str] | None = None) -> None:
    """Start FastMCP with stdio (default) or authenticated HTTP."""
    resolved = settings if settings is not None else resolve_run_settings(environ)
    if resolved.transport == "stdio":
        mcp.run(transport="stdio", show_banner=False)
        return

    assert resolved.host is not None
    assert resolved.port is not None
    assert resolved.path is not None

    if resolved.verify_url:
        assert resolved.internal_token is not None
        verifier = build_gateway_token_verifier(
            verify_url=resolved.verify_url,
            internal_token=resolved.internal_token,
        )
        apply_http_auth(mcp, verifier)
    else:
        assert resolved.token is not None
        apply_http_auth(mcp, resolved.token)

    allowed_hosts = list(resolved.allowed_hosts or DEFAULT_ALLOWED_HOSTS)
    uvicorn_config = dict(resolved.uvicorn_config or DEFAULT_UVICORN_CONFIG)

    mcp.run(
        transport="http",
        show_banner=False,
        host=resolved.host,
        port=resolved.port,
        path=resolved.path,
        allowed_hosts=allowed_hosts,
        uvicorn_config=uvicorn_config,
    )
