"""Tests for M2 authenticated REST API endpoints (/api/v1/{search,fetch,map})."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from starlette.testclient import TestClient

from grok_search.cli_auth import CliAuthConfig, CliAuthCoordinator, AuthRunStore
from grok_search.http_app import create_composed_app
from grok_search.mcp_transport import (
    apply_http_auth,
    build_static_token_verifier,
    build_gateway_token_verifier,
)
from grok_search.rest_api import RestApiHandlers
import grok_search.server as server
from grok_search.server import mcp

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using `httpx` with `starlette.testclient` is deprecated"
)

STATIC_TOKEN = "test-secret-token"
AUTH_HEADER = {"Authorization": f"Bearer {STATIC_TOKEN}"}


def _setup_test_app(
    token: str = STATIC_TOKEN,
    search_fn=None,
    fetch_fn=None,
    map_fn=None,
    custom_verifier=None,
):
    verifier = custom_verifier if custom_verifier is not None else build_static_token_verifier(token)
    apply_http_auth(mcp, verifier)
    mcp_app = mcp.http_app(path="/mcp", transport="http")

    config = CliAuthConfig(
        origin="https://search.karldigi.dev",
        oauth_client_id="grok-search-cli",
        oauth_redirect_uri="https://search.karldigi.dev/auth/cli/callback",
        gateway_authorize_url="https://search.karldigi.dev/authorize",
        gateway_token_url="http://127.0.0.1:8080/token",
        run_ttl_seconds=600,
        poll_interval_seconds=2,
    )
    coordinator = CliAuthCoordinator(config=config, store=AuthRunStore())
    handlers = RestApiHandlers(
        verifier=verifier,
        search_fn=search_fn,
        fetch_fn=fetch_fn,
        map_fn=map_fn,
    )
    app = create_composed_app(
        mcp_app,
        coordinator=coordinator,
        verifier=verifier,
        rest_handlers=handlers,
    )
    return app, handlers, verifier


# --- Auth Guard Tests ---

def test_rest_search_without_auth_returns_401():
    mock_search = AsyncMock()
    app, _, _ = _setup_test_app(search_fn=mock_search)

    with TestClient(app) as client:
        res = client.post("/api/v1/search", json={"query": "test"})
        assert res.status_code == 401
        assert res.headers.get("www-authenticate") == "Bearer"
        data = res.json()
        assert data["ok"] is False
        assert data["error"]["code"] == "unauthorized"
        mock_search.assert_not_called()


def test_rest_search_with_wrong_bearer_returns_401():
    mock_search = AsyncMock()
    app, _, _ = _setup_test_app(search_fn=mock_search)

    with TestClient(app) as client:
        res = client.post(
            "/api/v1/search",
            headers={"Authorization": "Bearer wrong-token"},
            json={"query": "test"},
        )
        assert res.status_code == 401
        assert res.headers.get("www-authenticate") == "Bearer"
        data = res.json()
        assert data["ok"] is False
        assert data["error"]["code"] == "unauthorized"
        mock_search.assert_not_called()


def test_rest_fetch_and_map_auth_guard():
    mock_fetch = AsyncMock()
    mock_map = AsyncMock()
    app, _, _ = _setup_test_app(fetch_fn=mock_fetch, map_fn=mock_map)

    with TestClient(app) as client:
        # fetch without auth
        res_fetch = client.post("/api/v1/fetch", json={"url": "https://example.com"})
        assert res_fetch.status_code == 401
        assert res_fetch.headers.get("www-authenticate") == "Bearer"
        mock_fetch.assert_not_called()

        # map without auth
        res_map = client.post("/api/v1/map", json={"url": "https://example.com"})
        assert res_map.status_code == 401
        assert res_map.headers.get("www-authenticate") == "Bearer"
        mock_map.assert_not_called()


# --- HTTP Method Restrictions (405) ---

def test_rest_endpoints_reject_get_with_405():
    mock_search = AsyncMock()
    mock_fetch = AsyncMock()
    mock_map = AsyncMock()
    app, _, _ = _setup_test_app(search_fn=mock_search, fetch_fn=mock_fetch, map_fn=mock_map)

    with TestClient(app) as client:
        for path in ("/api/v1/search", "/api/v1/fetch", "/api/v1/map"):
            res = client.get(path, headers=AUTH_HEADER)
            assert res.status_code == 405
            data = res.json()
            assert data["ok"] is False
            assert data["error"]["code"] == "method_not_allowed"

        mock_search.assert_not_called()
        mock_fetch.assert_not_called()
        mock_map.assert_not_called()


# --- Input Validation Tests ---

def test_search_missing_or_empty_query():
    mock_search = AsyncMock()
    app, _, _ = _setup_test_app(search_fn=mock_search)

    with TestClient(app) as client:
        # missing query
        r1 = client.post("/api/v1/search", headers=AUTH_HEADER, json={})
        assert r1.status_code == 400
        assert r1.json()["ok"] is False
        assert r1.json()["error"]["code"] == "invalid_param"

        # empty query
        r2 = client.post("/api/v1/search", headers=AUTH_HEADER, json={"query": "   "})
        assert r2.status_code == 400
        assert r2.json()["ok"] is False
        assert r2.json()["error"]["code"] == "invalid_param"

        # invalid JSON
        r3 = client.post(
            "/api/v1/search",
            headers={**AUTH_HEADER, "Content-Type": "application/json"},
            content=b"not-json",
        )
        assert r3.status_code == 400
        assert r3.json()["error"]["code"] == "invalid_json"

        mock_search.assert_not_called()


def test_search_payload_too_large():
    mock_search = AsyncMock()
    app, _, _ = _setup_test_app(search_fn=mock_search)

    with TestClient(app) as client:
        large_query = "x" * (65 * 1024)
        res = client.post("/api/v1/search", headers=AUTH_HEADER, json={"query": large_query})
        assert res.status_code == 413
        assert res.json()["error"]["code"] == "payload_too_large"
        mock_search.assert_not_called()


def test_url_validation_rejects_credentials_and_private_targets():
    mock_fetch = AsyncMock()
    mock_map = AsyncMock()
    app, _, _ = _setup_test_app(fetch_fn=mock_fetch, map_fn=mock_map)

    with TestClient(app) as client:
        bad_urls = [
            "ftp://example.com",
            "http://user:pass@example.com",
            "http://localhost/secret",
            "http://127.0.0.1:8800/api",
            "http://10.0.0.1/admin",
            "http://192.168.1.1/",
            "http://172.16.0.1/",
            "http://[::1]/",
            "http://0.0.0.0/",
            "not-a-url",
        ]

        for u in bad_urls:
            rf = client.post("/api/v1/fetch", headers=AUTH_HEADER, json={"url": u})
            assert rf.status_code == 400
            assert rf.json()["ok"] is False
            assert rf.json()["error"]["code"] == "invalid_url"

            rm = client.post("/api/v1/map", headers=AUTH_HEADER, json={"url": u})
            assert rm.status_code == 400
            assert rm.json()["ok"] is False
            assert rm.json()["error"]["code"] == "invalid_url"

        mock_fetch.assert_not_called()
        mock_map.assert_not_called()


def test_map_parameter_ranges():
    mock_map = AsyncMock(return_value='{"results": []}')
    app, _, _ = _setup_test_app(map_fn=mock_map)

    with TestClient(app) as client:
        # max_depth out of range (< 1 or > 5)
        r1 = client.post(
            "/api/v1/map",
            headers=AUTH_HEADER,
            json={"url": "https://example.com", "max_depth": 0},
        )
        assert r1.status_code == 400
        assert r1.json()["error"]["code"] == "invalid_param"

        r2 = client.post(
            "/api/v1/map",
            headers=AUTH_HEADER,
            json={"url": "https://example.com", "max_depth": 6},
        )
        assert r2.status_code == 400
        assert r2.json()["error"]["code"] == "invalid_param"

        # timeout out of range (< 10 or > 150)
        r3 = client.post(
            "/api/v1/map",
            headers=AUTH_HEADER,
            json={"url": "https://example.com", "timeout": 5},
        )
        assert r3.status_code == 400
        assert r3.json()["error"]["code"] == "invalid_param"

        # Valid parameters succeed
        r_ok = client.post(
            "/api/v1/map",
            headers=AUTH_HEADER,
            json={
                "url": "https://example.com",
                "instructions": "docs only",
                "max_depth": 2,
                "max_breadth": 10,
                "limit": 25,
                "timeout": 30,
            },
        )
        assert r_ok.status_code == 200
        assert r_ok.json()["ok"] is True
        mock_map.assert_called_once_with(
            url="https://example.com",
            instructions="docs only",
            max_depth=2,
            max_breadth=10,
            limit=25,
            timeout=30,
        )


# --- Successful Execution and Parity Tests ---

def test_search_success_embeds_sources():
    sources = [
        {"url": "https://example.com/1", "title": "Doc 1", "provider": "grok"},
        {"url": "https://example.com/2", "title": "Doc 2", "provider": "tavily"},
    ]
    mock_search = AsyncMock(return_value=(
        {"session_id": "sess-123", "content": "Search answer text", "sources_count": 2},
        sources,
    ))
    app, _, _ = _setup_test_app(search_fn=mock_search)

    with TestClient(app) as client:
        res = client.post(
            "/api/v1/search",
            headers=AUTH_HEADER,
            json={
                "query": "hello world",
                "platform": "Twitter",
                "model": "grok-beta",
                "extra_sources": 2,
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["ok"] is True
        assert data["data"] == {
            "content": "Search answer text",
            "sources": sources,
            "session_id": "sess-123",
            "sources_count": 2,
        }
        mock_search.assert_called_once_with(
            query="hello world",
            platform="Twitter",
            model="grok-beta",
            extra_sources=2,
        )


def test_fetch_success_returns_text_shape():
    mock_fetch = AsyncMock(return_value="# Page Title\n\nPage markdown content")
    app, _, _ = _setup_test_app(fetch_fn=mock_fetch)

    with TestClient(app) as client:
        res = client.post(
            "/api/v1/fetch",
            headers=AUTH_HEADER,
            json={"url": "https://example.com/docs"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["ok"] is True
        assert data["data"] == {"content": "# Page Title\n\nPage markdown content"}
        mock_fetch.assert_called_once_with("https://example.com/docs")


def test_map_success_returns_text_shape():
    map_text = json.dumps({"base_url": "https://example.com", "results": ["https://example.com/a"]})
    mock_map = AsyncMock(return_value=map_text)
    app, _, _ = _setup_test_app(map_fn=mock_map)

    with TestClient(app) as client:
        res = client.post(
            "/api/v1/map",
            headers=AUTH_HEADER,
            json={"url": "https://example.com"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["ok"] is True
        assert data["data"] == {"content": map_text}
        mock_map.assert_called_once_with(
            url="https://example.com",
            instructions="",
            max_depth=1,
            max_breadth=20,
            limit=50,
            timeout=150,
        )


# --- Error Envelopes and Upstream Failures ---

def test_search_upstream_error_envelope_returns_502():
    mock_search = AsyncMock(return_value=(
        {"session_id": "sess-fail", "content": "upstream_error: ConnectTimeout", "sources_count": 0},
        [],
    ))
    app, _, _ = _setup_test_app(search_fn=mock_search)

    with TestClient(app) as client:
        res = client.post("/api/v1/search", headers=AUTH_HEADER, json={"query": "test"})
        assert res.status_code == 502
        data = res.json()
        assert data["ok"] is False
        assert data["error"]["code"] == "upstream_error"
        assert "upstream_error: ConnectTimeout" in data["error"]["message"]
        assert data["error"]["session_id"] == "sess-fail"


def test_search_upstream_empty_returns_502():
    mock_search = AsyncMock(return_value=(
        {"session_id": "sess-empty", "content": "upstream_empty: grok stream completed with no answer content", "sources_count": 0},
        [],
    ))
    app, _, _ = _setup_test_app(search_fn=mock_search)

    with TestClient(app) as client:
        res = client.post("/api/v1/search", headers=AUTH_HEADER, json={"query": "test"})
        assert res.status_code == 502
        data = res.json()
        assert data["ok"] is False
        assert data["error"]["code"] == "upstream_error"
        assert "upstream_empty:" in data["error"]["message"]


def test_search_config_error_returns_502():
    mock_search = AsyncMock(return_value=(
        {"session_id": "sess-cfg", "content": "配置错误: GROK_API_KEY 未配置", "sources_count": 0},
        [],
    ))
    app, _, _ = _setup_test_app(search_fn=mock_search)

    with TestClient(app) as client:
        res = client.post("/api/v1/search", headers=AUTH_HEADER, json={"query": "test"})
        assert res.status_code == 502
        data = res.json()
        assert data["ok"] is False
        assert data["error"]["code"] == "configuration_error"


def test_search_invalid_model_returns_400():
    mock_search = AsyncMock(return_value=(
        {"session_id": "sess-model", "content": "无效模型: non-existent", "sources_count": 0},
        [],
    ))
    app, _, _ = _setup_test_app(search_fn=mock_search)

    with TestClient(app) as client:
        res = client.post(
            "/api/v1/search",
            headers=AUTH_HEADER,
            json={"query": "test", "model": "non-existent"},
        )
        assert res.status_code == 400
        data = res.json()
        assert data["ok"] is False
        assert data["error"]["code"] == "invalid_model"


def test_fetch_failures_return_502():
    app_fail, _, _ = _setup_test_app(fetch_fn=AsyncMock(return_value="提取失败: 所有提取服务均未能获取内容"))
    with TestClient(app_fail) as client:
        rf = client.post("/api/v1/fetch", headers=AUTH_HEADER, json={"url": "https://example.com"})
        assert rf.status_code == 502
        assert rf.json()["error"]["code"] == "upstream_error"

    app_empty, _, _ = _setup_test_app(fetch_fn=AsyncMock(return_value=""))
    with TestClient(app_empty) as client:
        re_empty = client.post("/api/v1/fetch", headers=AUTH_HEADER, json={"url": "https://example.com"})
        assert re_empty.status_code == 502
        assert re_empty.json()["error"]["code"] == "upstream_empty"


def test_map_timeout_returns_504():
    app, _, _ = _setup_test_app(map_fn=AsyncMock(return_value="映射超时: 请求超过150秒"))
    with TestClient(app) as client:
        rm = client.post("/api/v1/map", headers=AUTH_HEADER, json={"url": "https://example.com"})
        assert rm.status_code == 504
        assert rm.json()["error"]["code"] == "upstream_timeout"


# --- Gateway Token Verifier Compatibility ---

def test_gateway_verifier_auth_guard():
    async def gateway_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        token = body.get("token")
        if token == "gateway-valid-token":
            return httpx.Response(
                200,
                json={"client_id": "test-client", "scopes": ["mcp"]},
            )
        return httpx.Response(401, json={"error": "unauthorized"})

    transport = httpx.MockTransport(gateway_handler)
    verifier = build_gateway_token_verifier(
        verify_url="http://127.0.0.1:8080/internal/keys/verify",
        internal_token="test-internal",
        transport=transport,
    )
    mock_search = AsyncMock(return_value=(
        {"session_id": "sess-gw", "content": "Gateway search ok", "sources_count": 0},
        [],
    ))
    app, _, _ = _setup_test_app(search_fn=mock_search, custom_verifier=verifier)

    with TestClient(app) as client:
        # Invalid token -> 401
        r_bad = client.post(
            "/api/v1/search",
            headers={"Authorization": "Bearer gateway-invalid-token"},
            json={"query": "test"},
        )
        assert r_bad.status_code == 401
        mock_search.assert_not_called()

        # Valid token -> 200
        r_good = client.post(
            "/api/v1/search",
            headers={"Authorization": "Bearer gateway-valid-token"},
            json={"query": "test"},
        )
        assert r_good.status_code == 200
        assert r_good.json()["ok"] is True
        assert r_good.json()["data"]["content"] == "Gateway search ok"
        mock_search.assert_called_once()


# --- MCP Tool Signature and Output Contract Preservation ---

@pytest.mark.asyncio
async def test_mcp_web_search_still_omits_sources_list(monkeypatch):
    class _FakeGrok:
        def __init__(self, *args, **kwargs):
            pass
        async def search(self, query, platform):
            from grok_search.providers.contracts import SearchOutput, NormalizedSource
            return SearchOutput(
                content="MCP answer",
                sources=(NormalizedSource(url="https://example.com/doc", title="Doc"),),
            )

    monkeypatch.setenv("GROK_API_URL", "https://grok.test/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    monkeypatch.setattr(server, "GrokSearchProvider", _FakeGrok)

    # Call MCP tool directly
    mcp_result = await server.web_search("test query")
    assert set(mcp_result.keys()) == {"session_id", "content", "sources_count"}
    assert "sources" not in mcp_result
    assert mcp_result["content"] == "MCP answer"
    assert mcp_result["sources_count"] == 1

    # Sources can be fetched via get_sources
    source_result = await server.get_sources(mcp_result["session_id"])
    assert source_result["sources_count"] == 1
    assert source_result["sources"][0]["url"] == "https://example.com/doc"
