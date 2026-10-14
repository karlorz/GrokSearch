"""CLI configuration cannot disable an independently valid MCP HTTP server."""
from contextlib import asynccontextmanager

import httpx
import pytest
from starlette.testclient import TestClient

from grok_search.cli_auth import CliAuthConfig, resolve_cli_auth_config
from grok_search.http_app import create_composed_app
from grok_search.mcp_transport import apply_http_auth, resolve_run_settings, run_mcp
from grok_search.server import mcp

INITIALIZE = {
    'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
    'params': {'protocolVersion': '2025-03-26', 'capabilities': {},
               'clientInfo': {'name': 'dummy-cli-disabled-fixture', 'version': '1'}},
}
HEADERS = {'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream'}


@pytest.mark.parametrize('issuer', ['https://connector.example.invalid', 'http://127.0.0.1', 'http://localhost', 'http://[::1]'])
@pytest.mark.parametrize('entrypoint', [False, True], ids=['composed', 'run_mcp'])
def test_unconfigured_cli_preserves_custom_mcp_and_lifespan(monkeypatch, issuer, entrypoint):
    env = {
        'GROK_SEARCH_MCP_TRANSPORT': 'http', 'GROK_SEARCH_MCP_TOKEN': 'dummy-mcp-static',
        'GROK_SEARCH_MCP_OAUTH_ISSUER': issuer,
        'GROK_SEARCH_PUBLIC_ORIGIN': 'https://custom-cli.example.invalid',
        'GROK_SEARCH_CLI_OAUTH_REDIRECT_URI': 'https://custom-cli.example.invalid/callback',
        'GUDA_BASE_URL': 'https://custom-gateway.example.invalid',
        'GROK_SEARCH_CLI_OAUTH_CLIENT_ID': '   ',
    }
    settings = resolve_run_settings(env)
    assert settings.issuer == issuer
    assert resolve_cli_auth_config(env) == CliAuthConfig()
    # A configured process environment must not override explicit entrypoint env.
    monkeypatch.setenv('GROK_SEARCH_CLI_OAUTH_CLIENT_ID', 'should-not-enable-explicit-env')
    monkeypatch.setenv('GROK_SEARCH_MCP_OAUTH_ISSUER', 'https://attacker.invalid')
    async def forbidden_request(*args, **kwargs):
        pytest.fail('disabled CLI attempted a token exchange or custom-origin request')

    monkeypatch.setattr(httpx.AsyncClient, 'request', forbidden_request)
    lifespan_events = []
    closed = []
    original_http_app = mcp.http_app

    def observed_http_app(*args, **kwargs):
        app = original_http_app(*args, **kwargs)
        original_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def observed_lifespan(app):
            async with original_lifespan(app):
                lifespan_events.append('startup')
                yield
            lifespan_events.append('shutdown')

        app.router.lifespan_context = observed_lifespan
        return app

    monkeypatch.setattr(mcp, 'http_app', observed_http_app)

    def exercise(app):
        coordinator = app.state.cli_coordinator
        original_close = coordinator.aclose

        async def observe_close():
            await original_close()
            closed.append(True)

        monkeypatch.setattr(coordinator, 'aclose', observe_close)
        # Starlette's current TestClient cannot parse bracketed IPv6 netlocs;
        # the service's advertised issuer remains the separately validated value.
        with TestClient(app, base_url='http://127.0.0.1') as client:
            assert lifespan_events == ['startup']
            response = client.post('/mcp', json=INITIALIZE, headers=HEADERS)
            assert response.status_code == 401
            assert issuer in response.headers['www-authenticate']
            response = client.post('/mcp', json=INITIALIZE, headers={**HEADERS, 'Authorization': 'Bearer dummy-mcp-static'})
            assert response.status_code == 200
            for method, path in [('POST', '/auth/cli/start'), ('GET', '/auth/cli/approve?ref=dummy'),
                                 ('POST', '/auth/cli/approve'), ('GET', '/auth/cli/callback?code=dummy&state=dummy'),
                                 ('GET', '/auth/cli/check?authRunId=dummy')]:
                response = client.request(method, path, follow_redirects=False)
                assert response.status_code in {400, 404, 503}
                assert response.headers['cache-control'] == 'no-store'
                assert 'location' not in response.headers
                assert 'dummy-mcp-static' not in response.text
                assert 'custom-cli.example.invalid' not in response.text
                assert 'custom-gateway.example.invalid' not in response.text
            assert not coordinator.store._runs_by_id
        assert lifespan_events == ['startup', 'shutdown']
        assert closed == [True]

    original_auth = mcp.auth
    try:
        if entrypoint:
            import grok_search.mcp_transport as transport

            def serve(server):
                assert server.config.lifespan == 'on'
                assert server.config.access_log is False
                exercise(server.config.app)

            monkeypatch.setattr(transport.uvicorn.Server, 'run', serve)
            run_mcp(mcp, environ=env)
        else:
            apply_http_auth(mcp, 'dummy-mcp-static', resource_base_url=settings.issuer)
            exercise(create_composed_app(mcp.http_app(path='/mcp', transport='http'), cli_auth_environ=env))
    finally:
        mcp.auth = original_auth


def test_enabled_cli_configuration_stays_pinned():
    config = resolve_cli_auth_config({'GROK_SEARCH_CLI_OAUTH_CLIENT_ID': 'fixed-client'})
    assert config.oauth_client_id == 'fixed-client'
    assert config.origin == 'https://search.karldigi.dev'
    assert config.gateway_token_url == 'http://127.0.0.1:8080/token'


def test_disabled_cli_process_environment_preserves_mcp(monkeypatch):
    monkeypatch.delenv('GROK_SEARCH_CLI_OAUTH_CLIENT_ID', raising=False)
    monkeypatch.setenv('GROK_SEARCH_MCP_OAUTH_ISSUER', 'https://connector.example.invalid')
    monkeypatch.setenv('GUDA_BASE_URL', 'https://custom-gateway.example.invalid')
    assert resolve_cli_auth_config() == CliAuthConfig()
    original_auth = mcp.auth
    try:
        apply_http_auth(mcp, 'dummy-mcp-static', resource_base_url='https://connector.example.invalid')
        app = create_composed_app(mcp.http_app(path='/mcp', transport='http'))
        with TestClient(app) as client:
            assert client.post('/auth/cli/start').status_code == 503
            assert client.post('/mcp',json=INITIALIZE,headers={**HEADERS,'Authorization':'Bearer dummy-mcp-static'}).status_code == 200
    finally:
        mcp.auth = original_auth


def test_real_sdk_missing_session_is_pre_dispatch_rejection():
    from fastmcp import FastMCP

    dummy = FastMCP('dummy-session-fixture')
    calls = []

    @dummy.tool
    def web_search(query: str) -> str:
        calls.append(query)
        return 'dummy-answer'

    with TestClient(dummy.http_app(path='/mcp', transport='http')) as client:
        response = client.post('/mcp', json={'jsonrpc':'2.0','id':3,'method':'tools/call',
                                             'params':{'name':'web_search','arguments':{'query':'dummy'}}},
                               headers={**HEADERS,'Mcp-Session-Id':'unknown-dummy-session','MCP-Protocol-Version':'2025-03-26'})
        assert response.status_code == 404
        assert response.headers['content-type'] == 'application/json'
        assert response.json() == {'jsonrpc':'2.0','id':'server-error',
                                   'error':{'code':-32600,'message':'Session not found'}}
    assert not calls


@pytest.mark.asyncio
@pytest.mark.parametrize('terminated', [False, True])
async def test_real_sdk_transport_session_rejection_envelopes(terminated):
    from mcp.server.streamable_http import StreamableHTTPServerTransport

    transport = StreamableHTTPServerTransport(mcp_session_id='real-dummy-session', is_json_response_enabled=True)
    async with transport.connect(), httpx.AsyncClient(transport=httpx.ASGITransport(app=transport.handle_request), base_url='http://127.0.0.1') as client:
        if terminated:
            await transport.terminate()
        response = await client.post('/mcp',json={'jsonrpc':'2.0','id':3,'method':'tools/call',
                                                  'params':{'name':'web_search','arguments':{'query':'dummy'}}},
                                     headers={**HEADERS,'Mcp-Session-Id':'invalid-dummy-session','MCP-Protocol-Version':'2025-03-26'})
        assert response.status_code == 404
        assert response.headers['content-type'] == 'application/json'
        assert response.headers['mcp-session-id'] == 'real-dummy-session'
        assert response.json() == {'jsonrpc':'2.0','id':'server-error','error':{'code':-32600,
                                   'message': 'Not Found: Session has been terminated' if terminated else 'Not Found: Invalid or expired session ID'}}
