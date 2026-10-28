"""Host smoke of candidate engine against its configured upstream (no secrets printed).

Run with PYTHONPATH=<candidate>/src and the service's existing environment.
REST uses an in-process ASGI app with an ephemeral test bearer. MCP uses the
real SDK in-memory transport. Neither changes the production listener or auth.
"""
import asyncio
import json
import secrets

import httpx
from fastmcp import Client
from starlette.applications import Starlette

from grok_search import server
from grok_search.mcp_transport import build_static_token_verifier
from grok_search.providers.contracts import has_tool_trace, envelope_kind
from grok_search.rest_api import RestApiHandlers, create_rest_routes

QUERY = 'US stocks highest dollar volume October 28 2026'


def check_answer(content, sources):
    assert content.strip() and not has_tool_trace(content)
    assert envelope_kind(content) == 'ok'
    assert sources, 'successful search must have sources'


async def main():
    bearer = secrets.token_urlsafe(32)
    app = Starlette(routes=create_rest_routes(RestApiHandlers(
        verifier=build_static_token_verifier(bearer),
    )))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://localhost') as client:
        for label, query in [('original', QUERY), ('compact', QUERY + '. Compact top five with dollar volume in USD and sources.')]:
            response = await client.post('/api/v1/search', headers={'Authorization': 'Bearer ' + bearer}, json={'query': query})
            body = response.json()
            if response.status_code == 200:
                data = body['data']
                check_answer(data['content'], data['sources'])
                report = {'content_bytes': len(data['content'].encode()), 'sources': len(data['sources'])}
            else:
                assert response.status_code == 422, f'Unexpected REST status: {response.status_code}'
                assert body['ok'] is False and body['error']['code'] == 'search_incomplete'
                assert not has_tool_trace(body['error']['message'])
                report = {'code': body['error']['code']}
            print(json.dumps({'transport': 'REST-ASGI', 'query': label, 'status': response.status_code, **report}), flush=True)

    async with Client(server.mcp, timeout=600) as client:
        result = await client.call_tool('web_search', {'query': QUERY})
        data = json.loads(result.content[0].text)
        sources_result = await client.call_tool('get_sources', {'session_id': data['session_id']})
        sources = json.loads(sources_result.content[0].text)['sources']
        kind = envelope_kind(data['content'])
        assert not has_tool_trace(data['content'])
        if kind == 'ok':
            check_answer(data['content'], sources)
        else:
            assert kind == 'search_incomplete' and not sources
        print(json.dumps({'transport': 'MCP-SDK', 'kind': kind, 'sources': len(sources)}), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
