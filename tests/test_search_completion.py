"""Replay the production reasoning-only stream through provider, REST and MCP."""
import json

import httpx
import pytest
from starlette.testclient import TestClient

import grok_search.providers.grok as grok
import grok_search.server as server
from grok_search.providers.contracts import SearchOutput, NormalizedSource, envelope_from_grok_result
from test_rest_api import _setup_test_app, AUTH_HEADER

QUERY = 'US stocks highest dollar volume October 28 2026'
TRACE = '🔍 web_search: ' + QUERY
ANSWER = '| Rank | Ticker | Price USD | Shares | Dollar volume USD |\n|1|AAA|100|200|20,000|\nhttps://example.test/market'


def install_upstream(monkeypatch, outputs):
    requests = []
    real_client = httpx.AsyncClient

    def handle(request):
        requests.append(json.loads(request.content))
        value = outputs[min(len(requests) - 1, len(outputs) - 1)]
        if isinstance(value, int):
            return httpx.Response(value)
        events = [f'data: {json.dumps(event)}\n\n' for event in value]
        return httpx.Response(200, text=''.join(events) + 'data: [DONE]\n\n')

    monkeypatch.setattr(grok.httpx, 'AsyncClient', lambda **kw: real_client(**kw, transport=httpx.MockTransport(handle)))
    monkeypatch.setenv('GROK_API_URL', 'https://grok.test/v1')
    monkeypatch.setenv('GROK_API_KEY', 'test-key')
    monkeypatch.setenv('GROK_RETRY_MULTIPLIER', '0')
    monkeypatch.setenv('GROK_RETRY_MAX_WAIT', '0')
    return requests


def event(**delta):
    return {'choices': [{'delta': delta, 'finish_reason': 'stop'}]}


@pytest.mark.asyncio
async def test_reasoning_only_retries_once_then_returns_clean_error(monkeypatch):
    calls = install_upstream(monkeypatch, [[event(reasoning_content=TRACE)]])
    result = await grok.GrokSearchProvider('https://grok.test/v1', 'test-key').search(QUERY)
    assert result.content.startswith('search_incomplete:')
    assert TRACE not in result.content
    assert result.sources == ()
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_recovery_returns_only_final_answer_and_original_query(monkeypatch):
    calls = install_upstream(monkeypatch, [[event(reasoning_content=TRACE)], [event(content=ANSWER)]])
    result = await grok.GrokSearchProvider('https://grok.test/v1', 'test-key').search(QUERY)
    assert result.content == ANSWER
    assert len(calls) == 2
    assert QUERY in calls[1]['messages'][1]['content']
    assert 'price' in calls[1]['messages'][0]['content'].lower()
    assert 'share volume' in calls[1]['messages'][0]['content'].lower()


@pytest.mark.parametrize('content', [TRACE, TRACE + '\nHere is a ranking https://example.test/a'])
def test_shared_envelope_rejects_traces_even_with_sources(content):
    result = envelope_from_grok_result(SearchOutput(content=content, sources=(NormalizedSource('https://example.test/a'),)))
    assert result.content.startswith('search_incomplete:')
    assert 'web_search:' not in result.content
    assert not result.sources


@pytest.mark.asyncio
async def test_tool_role_is_never_answer(monkeypatch):
    calls = install_upstream(monkeypatch, [[event(role='tool', content=TRACE), event(role='assistant', content=ANSWER)]])
    result = await grok.GrokSearchProvider('https://grok.test/v1', 'test-key').search(QUERY)
    assert result.content == ANSWER
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_transient_502_and_second_compact_query_recover(monkeypatch):
    calls = install_upstream(monkeypatch, [502, [event(content=ANSWER)], 502, [event(content=ANSWER)]])
    provider = grok.GrokSearchProvider('https://grok.test/v1', 'test-key')
    assert (await provider.search(QUERY)).content == ANSWER
    assert (await provider.search(QUERY + ' Compact top five.')).content == ANSWER
    assert len(calls) == 4


@pytest.mark.asyncio
async def test_mcp_incomplete_content_matches_rest_error(monkeypatch):
    install_upstream(monkeypatch, [[event(reasoning_content=TRACE)]])
    result = await server.web_search(QUERY)
    assert result['content'].startswith('search_incomplete:')
    assert result['sources_count'] == 0
    assert (await server.get_sources(result['session_id']))['sources'] == []


def test_rest_consecutive_incomplete_queries_are_explicit_422_not_502(monkeypatch):
    calls = install_upstream(monkeypatch, [[event(reasoning_content=TRACE)]])
    app, _, _ = _setup_test_app()
    with TestClient(app) as client:
        for query in (QUERY, QUERY + ' Compact top five.'):
            response = client.post('/api/v1/search', headers=AUTH_HEADER, json={'query': query})
            assert response.status_code == 422
            body = response.json()
            assert body['ok'] is False
            assert body['error']['code'] == 'search_incomplete'
            assert 'web_search:' not in response.text
    assert len(calls) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize('output', [
    [event(content='An unsupported ranking')],
    [event(content='SEARCH_INCOMPLETE')],
    [event(content='No verified ranking https://example.test/data. Therefore SEARCH_INCOMPLETE.')],
    [{'choices': [{'delta': {'content': ANSWER}, 'finish_reason': 'length'}]}],
    [event(annotations=[{'url': 'https://example.test/source'}])],
])
async def test_unsourced_declined_truncated_or_sources_only_search_is_incomplete(monkeypatch, output):
    calls = install_upstream(monkeypatch, [output])
    result = await grok.GrokSearchProvider('https://grok.test/v1', 'test-key').search(QUERY)
    assert result.content.startswith('search_incomplete:')
    assert not result.sources
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_fragmented_tool_role_content_is_ignored(monkeypatch):
    install_upstream(monkeypatch, [[event(role='tool'), event(content=TRACE), event(role='assistant', content=ANSWER)]])
    result = await grok.GrokSearchProvider('https://grok.test/v1', 'test-key').search(QUERY)
    assert result.content == ANSWER


def test_rest_recovered_answer_embeds_sources(monkeypatch):
    install_upstream(monkeypatch, [[event(reasoning_content=TRACE)], [event(content=ANSWER)]])
    app, _, _ = _setup_test_app()
    with TestClient(app) as client:
        response = client.post('/api/v1/search', headers=AUTH_HEADER, json={'query': QUERY})
        assert response.status_code == 200
        body = response.json()
        assert body['ok'] is True
        assert '|1|AAA|100|200|20,000|' in body['data']['content']
        assert body['data']['sources'][0]['url'] == 'https://example.test/market'
