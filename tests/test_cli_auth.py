"""Tests for M1 hosted CLI auth coordinator and composed ASGI app."""

from __future__ import annotations

import json
import re
import urllib.parse
from typing import Any

import httpx
import pytest
from starlette.testclient import TestClient

from grok_search.cli_auth import (
    AuthRunState,
    AuthRunStore,
    CliAuthConfig,
    CliAuthCoordinator,
    _hash_secret,
    resolve_cli_auth_config,
)
from grok_search.http_app import create_composed_app
from grok_search.mcp_transport import apply_http_auth
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


def _extract_csrf(html: str) -> str:
    match = re.search(r'name="csrf_token"\s+value="([^"]+)"', html)
    assert match, f"CSRF token not found in HTML: {html[:200]}"
    return match.group(1)


def _setup_test_app(
    mcp_token: str = "test-mcp-token",
    mock_gateway_handler: Any = None,
    issuer: str | None = "https://search.karldigi.dev",
):
    apply_http_auth(mcp, mcp_token, resource_base_url=issuer)
    mcp_app = mcp.http_app(path="/mcp", transport="http")

    gateway_transport = httpx.MockTransport(mock_gateway_handler) if mock_gateway_handler else None
    mock_http_client = httpx.AsyncClient(transport=gateway_transport, timeout=5.0)

    config = CliAuthConfig(
        origin="https://search.karldigi.dev",
        oauth_client_id="grok-search-cli",
        oauth_redirect_uri="https://search.karldigi.dev/auth/cli/callback",
        gateway_authorize_url="https://search.karldigi.dev/authorize",
        gateway_token_url="http://127.0.0.1:8080/token",
        run_ttl_seconds=600,
        poll_interval_seconds=2,
    )
    store = AuthRunStore()
    coordinator = CliAuthCoordinator(config=config, store=store, http_client=mock_http_client)
    app = create_composed_app(mcp_app, coordinator=coordinator)
    return app, coordinator, store


# --- Unit & Route Tests ---


def test_start_returns_https_url_and_poll_secret_without_secret_in_url():
    app, coordinator, store = _setup_test_app()
    with TestClient(app) as client:
        resp = client.post("/auth/cli/start")
        assert resp.status_code == 201
        data = resp.json()

        assert "approveUrl" in data
        assert "authRunId" in data
        assert "pairingCode" in data
        assert "pollSecret" in data
        assert "expiresAt" in data
        assert "intervalSeconds" in data

        approve_url = data["approveUrl"]
        poll_secret = data["pollSecret"]
        run_id = data["authRunId"]

        # approveUrl must start with https:// and MUST NOT contain pollSecret
        assert approve_url.startswith("https://")
        assert poll_secret not in approve_url
        assert "ref=" in approve_url

        # Check security headers
        assert resp.headers["cache-control"] == "no-store"
        assert resp.headers["referrer-policy"] == "no-referrer"
        assert "DENY" in resp.headers["x-frame-options"]

        # Verify run state in store
        run = store.get_by_id(run_id)
        assert run is not None
        assert run.state == AuthRunState.PENDING
        assert run.poll_secret_hash == _hash_secret(poll_secret)


def test_start_rate_limiting():
    app, coordinator, store = _setup_test_app()
    coordinator.config.rate_limit_per_min = 2
    with TestClient(app) as client:
        r1 = client.post("/auth/cli/start")
        assert r1.status_code == 201
        r2 = client.post("/auth/cli/start")
        assert r2.status_code == 201
        r3 = client.post("/auth/cli/start")
        assert r3.status_code == 429
        assert r3.json()["error"] == "rate_limited"


def test_approve_get_renders_page_and_sets_csrf_cookie():
    app, coordinator, store = _setup_test_app()
    with TestClient(app) as client:
        start_res = client.post("/auth/cli/start").json()
        ref = urllib.parse.parse_qs(urllib.parse.urlparse(start_res["approveUrl"]).query)["ref"][0]

        # GET approve page
        get_res = client.get(f"/auth/cli/approve?ref={ref}")
        assert get_res.status_code == 200
        assert "text/html" in get_res.headers["content-type"]
        assert get_res.headers["cache-control"] == "no-store"
        assert get_res.headers["referrer-policy"] == "no-referrer"

        # Check pairing code in HTML
        run = store.get_by_id(start_res["authRunId"])
        assert run.pairing_code in get_res.text

        # Never leak pollSecret in HTML
        assert start_res["pollSecret"] not in get_res.text

        # Verify cookie is set
        cookies = get_res.cookies
        cookie_val = cookies.get("grok_search_cli_auth")
        assert cookie_val is not None


def test_approve_post_deny_marks_cancelled():
    app, coordinator, store = _setup_test_app()
    with TestClient(app) as client:
        start_res = client.post("/auth/cli/start").json()
        ref = urllib.parse.parse_qs(urllib.parse.urlparse(start_res["approveUrl"]).query)["ref"][0]

        get_res = client.get(f"/auth/cli/approve?ref={ref}")
        csrf_token = _extract_csrf(get_res.text)

        deny_res = client.post(
            "/auth/cli/approve",
            data={"ref": ref, "csrf_token": csrf_token, "action": "deny"},
        )
        assert deny_res.status_code == 200
        assert "Authorization Denied" in deny_res.text

        run = store.get_by_id(start_res["authRunId"])
        assert run.state == AuthRunState.CANCELLED

        # Check endpoint poll returns cancelled
        poll_res = client.get(
            f"/auth/cli/check?authRunId={run.run_id}",
            headers={"X-CLI-Poll-Secret": start_res["pollSecret"]},
        )
        assert poll_res.status_code == 200
        assert poll_res.json()["status"] == "cancelled"


def test_approve_post_approve_redirects_to_gateway_authorize_with_pkce():
    app, coordinator, store = _setup_test_app()
    with TestClient(app) as client:
        start_res = client.post("/auth/cli/start").json()
        ref = urllib.parse.parse_qs(urllib.parse.urlparse(start_res["approveUrl"]).query)["ref"][0]

        get_res = client.get(f"/auth/cli/approve?ref={ref}")
        csrf_token = _extract_csrf(get_res.text)

        post_res = client.post(
            "/auth/cli/approve",
            data={"ref": ref, "csrf_token": csrf_token, "action": "approve"},
            follow_redirects=False,
        )
        assert post_res.status_code == 302
        location = post_res.headers["location"]
        assert location.startswith("https://search.karldigi.dev/authorize")

        parsed = urllib.parse.urlparse(location)
        q = urllib.parse.parse_qs(parsed.query)
        assert q["client_id"][0] == "grok-search-cli"
        assert q["redirect_uri"][0] == "https://search.karldigi.dev/auth/cli/callback"
        assert q["response_type"][0] == "code"
        assert q["code_challenge_method"][0] == "S256"
        assert "code_challenge" in q
        assert "state" in q

        run = store.get_by_id(start_res["authRunId"])
        assert run.state == AuthRunState.AWAITING_GATEWAY
        assert run.pkce_verifier is not None
        assert run.oauth_state == q["state"][0]


def test_callback_binds_code_and_prevents_replay():
    app, coordinator, store = _setup_test_app()
    with TestClient(app) as client:
        start_res = client.post("/auth/cli/start").json()
        ref = urllib.parse.parse_qs(urllib.parse.urlparse(start_res["approveUrl"]).query)["ref"][0]

        get_res = client.get(f"/auth/cli/approve?ref={ref}")
        csrf_token = _extract_csrf(get_res.text)

        post_res = client.post(
            "/auth/cli/approve",
            data={"ref": ref, "csrf_token": csrf_token, "action": "approve"},
            follow_redirects=False,
        )
        parsed = urllib.parse.urlparse(post_res.headers["location"])
        oauth_state = urllib.parse.parse_qs(parsed.query)["state"][0]

        # Callback GET
        cb_res = client.get(f"/auth/cli/callback?code=mock_gateway_auth_code_123&state={oauth_state}")
        assert cb_res.status_code == 200
        assert "Authorization Successful" in cb_res.text

        run = store.get_by_id(start_res["authRunId"])
        assert run.state == AuthRunState.READY
        assert run.gateway_code == "mock_gateway_auth_code_123"

        # Replay callback with same state fails
        cb_replay = client.get(f"/auth/cli/callback?code=mock_gateway_auth_code_123&state={oauth_state}")
        assert cb_replay.status_code == 400


def test_check_without_secret_or_wrong_secret_does_not_leak_state():
    app, coordinator, store = _setup_test_app()
    with TestClient(app) as client:
        start_res = client.post("/auth/cli/start").json()
        run_id = start_res["authRunId"]

        # Missing secret header -> 400
        r_missing = client.get(f"/auth/cli/check?authRunId={run_id}")
        assert r_missing.status_code == 400

        r_wrong = client.get(
            f"/auth/cli/check?authRunId={run_id}",
            headers={"X-CLI-Poll-Secret": "wrong-secret-123"},
        )
        r_nonexistent = client.get(
            "/auth/cli/check?authRunId=unknown99999999",
            headers={"X-CLI-Poll-Secret": start_res["pollSecret"]},
        )
        assert r_wrong.status_code == 404
        assert r_nonexistent.status_code == 404
        assert r_wrong.json() == r_nonexistent.json()


def test_successful_check_exchanges_pkce_and_second_check_consumed():
    def gateway_token_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/token"
        form = urllib.parse.parse_qs(request.read().decode("utf-8"))
        assert form["grant_type"][0] == "authorization_code"
        assert form["code"][0] == "mock_code_abc"
        assert form["client_id"][0] == "grok-search-cli"
        assert form["redirect_uri"][0] == "https://search.karldigi.dev/auth/cli/callback"
        assert "code_verifier" in form

        return httpx.Response(
            200,
            json={
                "access_token": "gsk_issued_cli_token_999",
                "token_type": "Bearer",
                "expires_in": 7200,
                "refresh_token": "discard_this_refresh_token",
            },
        )

    app, coordinator, store = _setup_test_app(mock_gateway_handler=gateway_token_handler)

    with TestClient(app) as client:
        start_res = client.post("/auth/cli/start").json()
        run_id = start_res["authRunId"]
        poll_secret = start_res["pollSecret"]
        ref = urllib.parse.parse_qs(urllib.parse.urlparse(start_res["approveUrl"]).query)["ref"][0]

        # Poll while pending
        pending_poll = client.get(f"/auth/cli/check?authRunId={run_id}", headers={"X-CLI-Poll-Secret": poll_secret})
        assert pending_poll.status_code == 200
        assert pending_poll.json()["status"] == "pending"

        # Approve
        get_res = client.get(f"/auth/cli/approve?ref={ref}")
        csrf_token = _extract_csrf(get_res.text)
        post_res = client.post(
            "/auth/cli/approve",
            data={"ref": ref, "csrf_token": csrf_token, "action": "approve"},
            follow_redirects=False,
        )
        oauth_state = urllib.parse.parse_qs(urllib.parse.urlparse(post_res.headers["location"]).query)["state"][0]

        # Callback
        client.get(f"/auth/cli/callback?code=mock_code_abc&state={oauth_state}")

        # Check returns token
        claim_res = client.get(f"/auth/cli/check?authRunId={run_id}", headers={"X-CLI-Poll-Secret": poll_secret})
        assert claim_res.status_code == 200
        claim_data = claim_res.json()
        assert claim_data["status"] == "success"
        assert claim_data["token"] == "gsk_issued_cli_token_999"
        assert claim_data["token_type"] == "Bearer"
        assert claim_data["expires_in"] == 7200
        assert "refresh_token" not in claim_data

        # Store does NOT persist raw bearer
        run = store.get_by_id(run_id)
        assert run.state == AuthRunState.CONSUMED
        assert not hasattr(run, "access_token") or run.access_token is None
        assert run.gateway_code is None

        # Second poll returns consumed
        second_poll = client.get(f"/auth/cli/check?authRunId={run_id}", headers={"X-CLI-Poll-Secret": poll_secret})
        assert second_poll.status_code == 200
        assert second_poll.json()["status"] == "consumed"


def test_expiry_and_restart_invalidation():
    app, coordinator, store = _setup_test_app()
    with TestClient(app) as client:
        start_res = client.post("/auth/cli/start").json()
        run_id = start_res["authRunId"]
        poll_secret = start_res["pollSecret"]
        run = store.get_by_id(run_id)

        # Simulate expiration
        run.expires_at = 100.0

        poll_res = client.get(f"/auth/cli/check?authRunId={run_id}", headers={"X-CLI-Poll-Secret": poll_secret})
        assert poll_res.status_code == 200
        assert poll_res.json()["status"] == "expired"

    # Simulate server restart (empty store)
    restarted_app, restarted_coord, restarted_store = _setup_test_app()
    with TestClient(restarted_app) as client:
        poll_restart = client.get(f"/auth/cli/check?authRunId={run_id}", headers={"X-CLI-Poll-Secret": poll_secret})
        assert poll_restart.status_code == 404
        assert poll_restart.json()["error"] == "not_found"


# --- Composed App MCP Tests (FastMCP Lifespan and Bearer Regression) ---


def test_composed_app_mcp_initialize_and_tools_with_bearer():
    token = "test-composed-token"
    app, coordinator, store = _setup_test_app(mcp_token=token)

    with TestClient(app) as client:
        # CLI route works
        r_start = client.post("/auth/cli/start")
        assert r_start.status_code == 201

        # MCP unauthenticated -> 401 with resource_metadata
        r_unauth = client.post("/mcp", json=INITIALIZE_BODY, headers=MCP_HEADERS)
        assert r_unauth.status_code == 401
        assert "Bearer" in r_unauth.headers.get("www-authenticate", "")
        assert 'resource_metadata="https://search.karldigi.dev/.well-known/oauth-protected-resource/mcp"' in r_unauth.headers.get("www-authenticate", "")

        # MCP authenticated -> 200 initialize
        auth_headers = {**MCP_HEADERS, "Authorization": f"Bearer {token}"}
        r_init = client.post("/mcp", json=INITIALIZE_BODY, headers=auth_headers)
        assert r_init.status_code == 200
        init_data = _decode_mcp_response(r_init)
        assert init_data.get("result", {}).get("serverInfo", {}).get("name") == "grok-search"

        # Tools list -> 200
        session = r_init.headers.get("mcp-session-id")
        if session:
            auth_headers["Mcp-Session-Id"] = session
        r_tools = client.post("/mcp", json=TOOLS_LIST_BODY, headers=auth_headers)
        assert r_tools.status_code == 200
        tools_data = _decode_mcp_response(r_tools)
        tool_names = {t["name"] for t in tools_data["result"]["tools"]}
        assert "web_search" in tool_names
        assert "get_sources" in tool_names


def test_start_with_agent_id_and_invite_code():
    app, coordinator, store = _setup_test_app()
    with TestClient(app) as client:
        resp = client.post(
            "/auth/cli/start",
            json={
                "client_name": "GrokSearch CLI",
                "agent_id": "muse",
                "invite_code": "secret-invite-123",
            },
        )
        assert resp.status_code == 201
        data = resp.json()

        approve_url = data["approveUrl"]
        poll_secret = data["pollSecret"]
        run_id = data["authRunId"]

        # approveUrl contains invite= and ref=, but NOT pollSecret
        assert poll_secret not in approve_url
        assert "ref=" in approve_url
        assert "invite=secret-invite-123" in approve_url

        parsed_url = urllib.parse.urlparse(approve_url)
        q = urllib.parse.parse_qs(parsed_url.query)
        assert "ref" in q
        assert q["invite"] == ["secret-invite-123"]
        assert "pollSecret" not in q

        # Verify run attributes in store
        run = store.get_by_id(run_id)
        assert run is not None
        assert run.agent_id == "muse"
        assert run.invite_code == "secret-invite-123"

        # Verify approve post passes invite and agent_label to gateway authorize
        ref = q["ref"][0]
        get_res = client.get(f"/auth/cli/approve?ref={ref}")
        csrf_token = _extract_csrf(get_res.text)

        post_res = client.post(
            "/auth/cli/approve",
            data={"ref": ref, "csrf_token": csrf_token, "action": "approve"},
            follow_redirects=False,
        )
        assert post_res.status_code == 302
        location = post_res.headers["location"]
        auth_q = urllib.parse.parse_qs(urllib.parse.urlparse(location).query)
        assert auth_q["agent_label"] == ["muse"]
        assert auth_q["invite"] == ["secret-invite-123"]


def test_cli_routes_not_blocked_by_mcp_auth():
    token = "test-composed-token"
    app, coordinator, store = _setup_test_app(mcp_token=token)

    with TestClient(app) as client:
        # Calls to /auth/cli/* without any Bearer token are not 401'd by MCP auth
        r = client.post("/auth/cli/start")
        assert r.status_code == 201

        # Calls with an invalid Bearer token to /auth/cli/start are still not rejected with 401
        r_with_bearer = client.post("/auth/cli/start", headers={"Authorization": "Bearer invalid-token"})
        assert r_with_bearer.status_code == 201
