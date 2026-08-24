"""Tests for GatewayTokenVerifier and loopback verify-mode MCP auth."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
import pytest
from starlette.testclient import TestClient

from grok_search.mcp_transport import (
    DEFAULT_ALLOWED_HOSTS,
    DEFAULT_UVICORN_CONFIG,
    GatewayTokenVerifier,
    McpHttpConfigError,
    apply_http_auth,
    build_gateway_token_verifier,
    resolve_run_settings,
    run_mcp,
)
from grok_search.server import mcp

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using `httpx` with `starlette.testclient` is deprecated"
)

MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
    "MCP-Protocol-Version": "2025-03-26",
}

INITIALIZE_BODY = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "grok-search-tests", "version": "0.1.0"},
    },
}

TOOLS_LIST_BODY = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list",
    "params": {},
}


def _decode_mcp_response(response: httpx.Response) -> dict:
    ctype = response.headers.get("content-type", "")
    if "text/event-stream" in ctype:
        data_lines = [
            line[5:].strip()
            for line in response.text.splitlines()
            if line.startswith("data:")
        ]
        assert data_lines, response.text
        return json.loads(data_lines[-1])
    return response.json()


# --- Auth Mode Resolution Tests ---


def test_verify_url_without_internal_token_fails_closed():
    environ = {
        "GROK_SEARCH_MCP_TRANSPORT": "http",
        "GROK_SEARCH_MCP_VERIFY_URL": "http://127.0.0.1:8080/internal/keys/verify",
    }
    with pytest.raises(McpHttpConfigError, match="GROK_SEARCH_MCP_INTERNAL_TOKEN"):
        resolve_run_settings(environ)


def test_verify_url_with_blank_internal_token_fails_closed():
    environ = {
        "GROK_SEARCH_MCP_TRANSPORT": "http",
        "GROK_SEARCH_MCP_VERIFY_URL": "http://127.0.0.1:8080/internal/keys/verify",
        "GROK_SEARCH_MCP_INTERNAL_TOKEN": "   ",
    }
    with pytest.raises(McpHttpConfigError, match="GROK_SEARCH_MCP_INTERNAL_TOKEN"):
        resolve_run_settings(environ)


def test_verify_mode_success_without_static_token():
    environ = {
        "GROK_SEARCH_MCP_TRANSPORT": "http",
        "GROK_SEARCH_MCP_VERIFY_URL": "http://127.0.0.1:8080/internal/keys/verify",
        "GROK_SEARCH_MCP_INTERNAL_TOKEN": "secret-internal-key",
    }
    settings = resolve_run_settings(environ)
    assert settings.transport == "http"
    assert settings.verify_url == "http://127.0.0.1:8080/internal/keys/verify"
    assert settings.internal_token == "secret-internal-key"
    assert settings.token is None
    assert settings.host == "127.0.0.1"
    assert settings.port == 8800
    assert settings.path == "/mcp"
    assert list(settings.allowed_hosts or []) == list(DEFAULT_ALLOWED_HOSTS)


def test_verify_url_wins_when_both_verify_url_and_token_are_present():
    environ = {
        "GROK_SEARCH_MCP_TRANSPORT": "http",
        "GROK_SEARCH_MCP_VERIFY_URL": "http://127.0.0.1:8080/internal/keys/verify",
        "GROK_SEARCH_MCP_INTERNAL_TOKEN": "secret-internal-key",
        "GROK_SEARCH_MCP_TOKEN": "static-token-ignored",
    }
    settings = resolve_run_settings(environ)
    assert settings.verify_url == "http://127.0.0.1:8080/internal/keys/verify"
    assert settings.internal_token == "secret-internal-key"
    assert settings.token is None


# --- GatewayTokenVerifier Unit Tests ---


@pytest.mark.asyncio
async def test_verifier_200_returns_access_token():
    requests_seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        assert request.headers.get("X-Internal-Token") == "internal-secret"
        payload = json.loads(request.read())
        assert payload == {"token": "gsk_test_12345"}
        return httpx.Response(
            200,
            json={
                "name": "tester-key",
                "prefix": "gsk_test",
                "fingerprint": "fp123",
                "client_id": "client-abc",
                "scopes": ["mcp", "search"],
            },
        )

    transport = httpx.MockTransport(handler)
    verifier = GatewayTokenVerifier(
        verify_url="http://127.0.0.1:8080/internal/keys/verify",
        internal_token="internal-secret",
        transport=transport,
    )

    token_info = await verifier.verify_token("gsk_test_12345")
    assert token_info is not None
    assert token_info.token == "gsk_test_12345"
    assert token_info.client_id == "client-abc"
    assert token_info.scopes == ["mcp", "search"]
    assert token_info.claims["name"] == "tester-key"
    assert len(requests_seen) == 1


@pytest.mark.asyncio
async def test_verifier_200_fallbacks_for_client_id_and_scopes():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"name": "named-key"})

    transport = httpx.MockTransport(handler)
    verifier = GatewayTokenVerifier(
        verify_url="http://127.0.0.1:8080/internal/keys/verify",
        internal_token="internal-secret",
        transport=transport,
    )

    token_info = await verifier.verify_token("gsk_test_name_only")
    assert token_info is not None
    assert token_info.client_id == "named-key"
    assert token_info.scopes == ["mcp"]


@pytest.mark.asyncio
async def test_verifier_required_scopes_subset_check():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"client_id": "client-1", "scopes": ["mcp"]},
        )

    transport = httpx.MockTransport(handler)
    verifier = GatewayTokenVerifier(
        verify_url="http://127.0.0.1:8080/internal/keys/verify",
        internal_token="internal-secret",
        transport=transport,
        required_scopes=["mcp", "admin"],
    )

    token_info = await verifier.verify_token("gsk_insufficient_scopes")
    assert token_info is None


@pytest.mark.asyncio
async def test_verifier_401_negative_cache_and_ttl(monkeypatch):
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(401, json={"error": "invalid_key"})

    transport = httpx.MockTransport(handler)
    fake_now = 1000.0
    monkeypatch.setattr("time.time", lambda: fake_now)

    verifier = GatewayTokenVerifier(
        verify_url="http://127.0.0.1:8080/internal/keys/verify",
        internal_token="internal-secret",
        transport=transport,
        negative_cache_ttl=60.0,
    )

    # First call: hits gateway, gets 401, negative-caches
    res1 = await verifier.verify_token("gsk_bad_token")
    assert res1 is None
    assert call_count == 1

    # Second call within TTL: cached, does NOT hit gateway
    res2 = await verifier.verify_token("gsk_bad_token")
    assert res2 is None
    assert call_count == 1

    # Ensure raw token is never stored in cache dictionary keys
    assert "gsk_bad_token" not in verifier._negative_cache
    assert len(verifier._negative_cache) == 1

    # Advance time past 60s TTL: hits gateway again
    fake_now = 1065.0
    res3 = await verifier.verify_token("gsk_bad_token")
    assert res3 is None
    assert call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_code, response_body",
    [
        (403, '{"error": "forbidden"}'),
        (500, '{"error": "internal_error"}'),
        (502, "Bad Gateway"),
        (200, "not json!"),
    ],
)
async def test_verifier_errors_not_negatively_cached(status_code, response_body):
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(status_code, text=response_body)

    transport = httpx.MockTransport(handler)
    verifier = GatewayTokenVerifier(
        verify_url="http://127.0.0.1:8080/internal/keys/verify",
        internal_token="internal-secret",
        transport=transport,
    )

    res1 = await verifier.verify_token("gsk_test_key")
    assert res1 is None
    assert call_count == 1
    assert len(verifier._negative_cache) == 0

    # Second call hits gateway again because error was NOT cached
    res2 = await verifier.verify_token("gsk_test_key")
    assert res2 is None
    assert call_count == 2


@pytest.mark.asyncio
async def test_verifier_network_timeout_not_negatively_cached():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        raise httpx.ConnectTimeout("connection timed out")

    transport = httpx.MockTransport(handler)
    verifier = GatewayTokenVerifier(
        verify_url="http://127.0.0.1:8080/internal/keys/verify",
        internal_token="internal-secret",
        transport=transport,
    )

    res1 = await verifier.verify_token("gsk_timeout_key")
    assert res1 is None
    assert call_count == 1
    assert len(verifier._negative_cache) == 0

    res2 = await verifier.verify_token("gsk_timeout_key")
    assert res2 is None
    assert call_count == 2


# --- FastMCP HTTP verify-mode Integration Test ---


def test_http_mcp_app_with_gateway_token_verifier():
    def gateway_handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("X-Internal-Token") != "internal-secret":
            return httpx.Response(403, json={"error": "forbidden"})
        payload = json.loads(request.read())
        token = payload.get("token")
        if token == "gsk_valid_live_token":
            return httpx.Response(
                200,
                json={
                    "client_id": "live-client",
                    "scopes": ["mcp"],
                    "name": "live-key",
                },
            )
        return httpx.Response(401, json={"error": "invalid_key"})

    transport = httpx.MockTransport(gateway_handler)
    verifier = build_gateway_token_verifier(
        verify_url="http://127.0.0.1:8080/internal/keys/verify",
        internal_token="internal-secret",
        transport=transport,
    )

    previous = getattr(mcp, "auth", None)
    apply_http_auth(mcp, verifier)
    app = mcp.http_app(path="/mcp", transport="http")

    try:
        with TestClient(app) as client:
            # 1. Missing bearer -> 401
            missing = client.post("/mcp", json=INITIALIZE_BODY, headers=MCP_HEADERS)
            assert missing.status_code == 401

            # 2. Invalid bearer -> 401
            invalid = client.post(
                "/mcp",
                json=INITIALIZE_BODY,
                headers={**MCP_HEADERS, "Authorization": "Bearer gsk_bad_live_token"},
            )
            assert invalid.status_code == 401

            # 3. Valid bearer -> 200 initialize + 200 tools/list
            valid_headers = {
                **MCP_HEADERS,
                "Authorization": "Bearer gsk_valid_live_token",
            }
            init = client.post("/mcp", json=INITIALIZE_BODY, headers=valid_headers)
            assert init.status_code == 200, init.text
            init_payload = _decode_mcp_response(init)
            assert init_payload.get("result", {}).get("serverInfo", {}).get("name") == "grok-search"

            session = init.headers.get("mcp-session-id")
            if session:
                valid_headers["Mcp-Session-Id"] = session

            listed = client.post("/mcp", json=TOOLS_LIST_BODY, headers=valid_headers)
            assert listed.status_code == 200, listed.text
            tools_payload = _decode_mcp_response(listed)
            names = {t["name"] for t in tools_payload["result"]["tools"]}
            assert "web_search" in names
            assert "get_sources" in names
    finally:
        mcp.auth = previous


@pytest.mark.asyncio
async def test_verifier_no_token_leaks_in_cache_or_logs(caplog):
    import logging

    caplog.set_level(logging.DEBUG)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid_key"})

    transport = httpx.MockTransport(handler)
    verifier = GatewayTokenVerifier(
        verify_url="http://127.0.0.1:8080/internal/keys/verify",
        internal_token="internal-super-secret-key-123",
        transport=transport,
    )

    sensitive_token = "gsk_live_secret_token_abcdef123456"
    res = await verifier.verify_token(sensitive_token)
    assert res is None

    # Check cache does not store raw token
    for k in verifier._negative_cache:
        assert sensitive_token not in k
        assert "gsk_" not in k

    # Check logs do not leak raw bearer or internal secret
    all_logs = caplog.text
    assert sensitive_token not in all_logs
    assert "internal-super-secret-key-123" not in all_logs


def test_run_mcp_forwards_allowed_hosts_and_uvicorn_config(monkeypatch):
    mock_mcp = MagicMock()
    mock_mcp.run = MagicMock()

    environ = {
        "GROK_SEARCH_MCP_TRANSPORT": "http",
        "GROK_SEARCH_MCP_TOKEN": "test-token",
    }
    settings = resolve_run_settings(environ)
    run_mcp(mock_mcp, settings=settings)

    mock_mcp.run.assert_called_once_with(
        transport="http",
        show_banner=False,
        host="127.0.0.1",
        port=8800,
        path="/mcp",
        allowed_hosts=list(DEFAULT_ALLOWED_HOSTS),
        uvicorn_config=dict(DEFAULT_UVICORN_CONFIG),
    )
