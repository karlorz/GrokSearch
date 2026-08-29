import json

import httpx
import pytest

from grok_search.config import Config

pytestmark = pytest.mark.usefixtures("isolated_config")


def test_config_info_identifies_public_remote_engine_without_internal_details(monkeypatch):
    monkeypatch.setenv("GROK_SEARCH_MCP_PUBLIC_URL", " https://search.karldigi.dev/mcp ")
    monkeypatch.setenv("GROK_SEARCH_MCP_TRANSPORT", "http")
    monkeypatch.setenv("GROK_SEARCH_MCP_TOKEN", "client-test-token")
    monkeypatch.setenv("GUDA_BASE_URL", "http://127.0.0.1:8080")
    monkeypatch.setenv("GUDA_API_KEY", "gsk_internal-secret-fragment")
    monkeypatch.setenv("GROK_API_URL", "http://127.0.0.1:8080/grok/v1")
    monkeypatch.setenv("GROK_API_KEY", "gsk_upstream-secret-fragment")
    monkeypatch.setenv("TAVILY_API_URL", "http://127.0.0.1:8080/tavily")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-internal-secret-fragment")
    monkeypatch.setenv("TAVILY_ENABLED", "true")
    monkeypatch.setenv("FIRECRAWL_ENABLED", "false")

    info = Config().get_config_info()

    assert info["mcp_connection"] == {
        "status": "remote_engine_active",
        "transport": "streamable_http",
        "public_endpoint": "https://search.karldigi.dev/mcp",
        "message": (
            "This response is from the configured remote Grok Search engine. "
            "No local Grok Search, GUDA, Tavily, or Firecrawl service is required."
        ),
        "client_configuration": {
            "endpoint_env": "GROK_SEARCH_MCP_URL",
            "bearer_token_env": "GROK_SEARCH_MCP_TOKEN",
            "authentication": "static_bearer",
        },
    }
    assert info["remote_engine"] == {
        "model": "grok-4.3-fast",
        "tavily_enabled": True,
        "firecrawl_enabled": False,
    }
    assert info["config_status"] == "complete"

    serialized = json.dumps(info)
    for forbidden in [
        "127.0.0.1",
        "GUDA_BASE_URL",
        "GROK_API_URL",
        "GROK_LOG_DIR",
        "gsk_internal",
        "gsk_upstream",
        "tvly-internal",
    ]:
        assert forbidden not in serialized


def test_config_info_without_public_url_does_not_claim_remote_endpoint(monkeypatch):
    monkeypatch.setenv("GUDA_BASE_URL", "http://127.0.0.1:8080")
    monkeypatch.setenv("GUDA_API_KEY", "gsk_internal-secret-fragment")

    info = Config().get_config_info()

    assert info["mcp_connection"]["status"] == "public_endpoint_not_advertised"
    assert info["mcp_connection"]["public_endpoint"] is None
    assert "remote Grok Search engine" not in info["mcp_connection"]["message"]
    assert "127.0.0.1" not in json.dumps(info)


@pytest.mark.parametrize(
    "unsafe_url",
    [
        "/mcp",
        "http://127.0.0.1:8800/mcp",
        "http://127.1/mcp",
        "http://2130706433/mcp",
        "http://0x7f000001/mcp",
        "https://user:secret@example.com/mcp",
        "https://example.com/mcp?token=secret",
        "https://example.com/mcp#secret",
        "https://example.com/mcp\nsecret",
        "https://[::1",
    ],
)
def test_config_info_rejects_unsafe_public_url(monkeypatch, unsafe_url):
    monkeypatch.setenv("GROK_SEARCH_MCP_PUBLIC_URL", unsafe_url)
    monkeypatch.setenv("GROK_SEARCH_MCP_TRANSPORT", "http")
    monkeypatch.setenv("GUDA_API_KEY", "gsk_internal-secret-fragment")

    info = Config().get_config_info()

    assert info["mcp_connection"]["status"] == "public_endpoint_not_advertised"
    assert info["mcp_connection"]["public_endpoint"] is None


def test_config_info_does_not_report_http_active_for_stdio(monkeypatch):
    monkeypatch.setenv("GROK_SEARCH_MCP_PUBLIC_URL", "https://search.karldigi.dev/mcp")
    monkeypatch.setenv("GROK_SEARCH_MCP_TRANSPORT", "stdio")
    monkeypatch.setenv("GUDA_API_KEY", "gsk_internal-secret-fragment")

    info = Config().get_config_info()

    assert info["mcp_connection"]["status"] == "public_endpoint_advertised"
    assert info["mcp_connection"]["transport"] == "stdio"


def test_config_info_does_not_report_invalid_http_configuration_as_active(monkeypatch):
    monkeypatch.setenv("GROK_SEARCH_MCP_PUBLIC_URL", "https://search.karldigi.dev/mcp")
    monkeypatch.setenv("GROK_SEARCH_MCP_TRANSPORT", "http")
    monkeypatch.setenv("GROK_SEARCH_MCP_PORT", "not-a-port")
    monkeypatch.setenv("GROK_SEARCH_MCP_TOKEN", "client-test-token")
    monkeypatch.setenv("GUDA_API_KEY", "gsk_internal-secret-fragment")

    info = Config().get_config_info()

    assert info["mcp_connection"]["status"] == "http_configuration_invalid"
    assert info["mcp_connection"]["transport"] == "streamable_http"


def test_config_info_labels_remote_gateway_user_bearer(monkeypatch):
    monkeypatch.setenv("GROK_SEARCH_MCP_PUBLIC_URL", "https://search.karldigi.dev/mcp")
    monkeypatch.setenv("GROK_SEARCH_MCP_TRANSPORT", "http")
    monkeypatch.setenv(
        "GROK_SEARCH_MCP_VERIFY_URL",
        "http://127.0.0.1:8080/internal/keys/verify",
    )
    monkeypatch.setenv("GROK_SEARCH_MCP_INTERNAL_TOKEN", "internal-test-token")
    monkeypatch.setenv("GUDA_API_KEY", "gsk_internal-secret-fragment")

    info = Config().get_config_info()

    assert info["mcp_connection"]["client_configuration"] == {
        "endpoint_env": "GROK_SEARCH_MCP_URL",
        "bearer_token_env": "GROK_SEARCH_MCP_TOKEN",
        "authentication": "gateway_user_bearer",
    }


class _FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class _FakeAsyncClient:
    response = _FakeResponse(200, {"data": [{"id": "grok-4.3-fast"}]})

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def get(self, *args, **kwargs):
        return self.response


@pytest.mark.asyncio
async def test_server_config_info_returns_safe_success_contract(monkeypatch):
    import grok_search.server as server

    monkeypatch.setenv("GROK_SEARCH_MCP_PUBLIC_URL", "https://search.karldigi.dev/mcp")
    monkeypatch.setenv("GROK_SEARCH_MCP_TRANSPORT", "http")
    monkeypatch.setenv("GROK_SEARCH_MCP_TOKEN", "client-test-token")
    monkeypatch.setenv("GUDA_API_KEY", "gsk_internal-secret-fragment")
    _FakeAsyncClient.response = _FakeResponse(200, {"data": [{"id": "grok-4.3-fast"}]})
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    result = json.loads(await server.get_config_info())

    assert result["mcp_connection"]["public_endpoint"] == "https://search.karldigi.dev/mcp"
    assert result["mcp_connection"]["client_configuration"] == {
        "endpoint_env": "GROK_SEARCH_MCP_URL",
        "bearer_token_env": "GROK_SEARCH_MCP_TOKEN",
        "authentication": "static_bearer",
    }
    assert result["connection_test"]["status"] == "success"
    assert result["connection_test"]["http_status"] == 200
    assert result["connection_test"]["available_model_count"] == 1
    assert result["connection_test"]["available_models"] == ["grok-4.3-fast"]
    assert "gsk_internal" not in json.dumps(result)


@pytest.mark.asyncio
async def test_server_config_info_does_not_echo_upstream_error_body(monkeypatch):
    import grok_search.server as server

    monkeypatch.setenv("GROK_SEARCH_MCP_PUBLIC_URL", "https://search.karldigi.dev/mcp")
    monkeypatch.setenv("GROK_SEARCH_MCP_TRANSPORT", "http")
    monkeypatch.setenv("GROK_SEARCH_MCP_TOKEN", "client-test-token")
    monkeypatch.setenv("GUDA_API_KEY", "gsk_internal-secret-fragment")
    _FakeAsyncClient.response = _FakeResponse(
        502,
        text="internal gateway detail with secret-fragment and http://127.0.0.1:8080",
    )
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    result = json.loads(await server.get_config_info())

    connection_test = result["connection_test"]
    response_time_ms = connection_test.pop("response_time_ms")
    assert connection_test == {
        "status": "http_error",
        "message": "The remote engine returned an unsuccessful status.",
        "http_status": 502,
    }
    assert isinstance(response_time_ms, (int, float))
    assert response_time_ms >= 0
    serialized = json.dumps(result)
    assert "secret-fragment" not in serialized
    assert "127.0.0.1" not in serialized
