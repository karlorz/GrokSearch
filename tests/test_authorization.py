"""Exercise permissions through real HTTP MCP dispatch, before side effects."""

import json

import httpx
import pytest
from starlette.testclient import TestClient

from grok_search.mcp_transport import build_gateway_token_verifier, build_static_token_verifier
from grok_search.server import mcp


def decode(response):
    if "text/event-stream" in response.headers.get("content-type", ""):
        return json.loads(next(line[5:] for line in response.text.splitlines() if line.startswith("data:")))
    return response.json()


@pytest.mark.parametrize("credential", ["oauth-read", "oauth-config", "host", "static"])
def test_configuration_tools_enforce_http_permission_before_writes(credential, monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    from grok_search.config import config
    changes = []
    monkeypatch.setattr(config, "set_model", lambda model: changes.append(model))
    resource = "https://search.example/mcp"

    def verify(request):
        assert json.loads(request.read())["resource"] == resource
        return httpx.Response(200, json={
            "client_id": credential,
            "scopes": ["mcp", "config:write"] if credential == "oauth-config" else ["mcp"],
            "oauth_owned": credential != "host",
            "resource": resource,
        })

    verifier = build_static_token_verifier("test-only") if credential == "static" else build_gateway_token_verifier(
        "http://gateway/internal/keys/verify", "internal-test", expected_resource=resource,
        transport=httpx.MockTransport(verify),
    )
    monkeypatch.setattr(mcp, "auth", verifier)
    with TestClient(mcp.http_app(path="/mcp", transport="http")) as client:
        headers = {"Accept": "application/json, text/event-stream", "Authorization": "Bearer test-only"}
        response = client.post("/mcp", headers=headers, json={"jsonrpc":"2.0", "id":1, "method":"initialize", "params":{
            "protocolVersion":"2025-03-26", "capabilities":{}, "clientInfo":{"name":"permissions-test", "version":"1"},
        }})
        assert response.status_code == 200
        if response.headers.get("mcp-session-id"):
            headers["Mcp-Session-Id"] = response.headers["mcp-session-id"]
        for name, arguments in [("switch_model", {"model":"grok-4.3-fast"}), ("toggle_builtin_tools", {"action":"on"})]:
            result = decode(client.post("/mcp", headers=headers, json={"jsonrpc":"2.0", "id":2, "method":"tools/call", "params":{"name":name, "arguments":arguments}}))["result"]
            if credential == "oauth-read":
                assert result.get("isError") is True
                assert "config:write" in result["content"][0]["text"]
                assert not changes
                assert not (tmp_path / ".claude/settings.json").exists()
            else:
                assert result.get("isError") is not True, result
        if credential != "oauth-read":
            assert changes == ["grok-4.3-fast"]
            assert (tmp_path / ".claude/settings.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("returned_resource", [None, "https://other.example/mcp"])
async def test_gateway_rejects_oauth_token_for_wrong_or_missing_resource(returned_resource):
    verifier = build_gateway_token_verifier("http://gateway/verify", "test-internal", expected_resource="https://search.example/mcp",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"client_id":"test", "oauth_owned":True, "scopes":["mcp"], "resource":returned_resource})))
    assert await verifier.verify_token("test-only") is None


@pytest.mark.asyncio
async def test_gateway_rejects_credential_without_mcp_scope():
    verifier = build_gateway_token_verifier("http://gateway/verify", "test-internal",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"client_id":"test", "scopes":[]})))
    assert await verifier.verify_token("test-only") is None
