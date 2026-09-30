import httpx
import pytest

import grok_search.server as server
from grok_search.providers.contracts import NormalizedSource, SearchOutput


class _EmptyGrokProvider:
    def __init__(self, api_url: str, api_key: str, model: str) -> None:
        pass

    async def search(self, query: str, platform: str) -> SearchOutput:
        return SearchOutput()


@pytest.mark.asyncio
async def test_web_search_merges_structured_fallback_and_extra_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeGrokProvider:
        def __init__(self, api_url: str, api_key: str, model: str) -> None:
            pass

        async def search(self, query: str, platform: str) -> SearchOutput:
            return SearchOutput(
                content=(
                    "Answer body\n\n"
                    "## Sources\n"
                    "- [Fallback A](https://example.test/a)\n"
                    "- [Fallback C](https://example.test/c)"
                ),
                sources=(
                    NormalizedSource(
                        url="https://example.test/a",
                        title="Structured A",
                        provider="grok",
                    ),
                    NormalizedSource(
                        url="https://example.test/b",
                        provider="grok",
                    ),
                ),
            )

    async def _fake_tavily(query: str, max_results: int) -> list[dict]:
        return [
            {
                "url": "https://example.test/d",
                "title": "Tavily D",
                "content": "D description",
            }
        ]

    async def _fake_firecrawl(query: str, limit: int) -> list[dict]:
        return [
            {
                "url": "https://example.test/b",
                "title": "Firecrawl B",
                "description": "B description",
            }
        ]

    monkeypatch.setenv("GROK_API_URL", "https://grok.test/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-grok-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-firecrawl-key")
    monkeypatch.setattr(server, "GrokSearchProvider", _FakeGrokProvider)
    monkeypatch.setattr(server, "_call_tavily_search", _fake_tavily)
    monkeypatch.setattr(server, "_call_firecrawl_search", _fake_firecrawl)

    search_response = await server.web_search("query", extra_sources=2)
    source_response = await server.get_sources(search_response["session_id"])

    assert set(search_response) == {"session_id", "content", "sources_count"}
    assert search_response["content"] == "Answer body"
    assert search_response["sources_count"] == 4
    assert source_response == {
        "session_id": search_response["session_id"],
        "sources": [
            {
                "url": "https://example.test/a",
                "title": "Structured A",
                "provider": "grok",
            },
            {
                "url": "https://example.test/b",
                "provider": "grok",
                "title": "Firecrawl B",
                "description": "B description",
            },
            {"url": "https://example.test/c", "title": "Fallback C"},
            {
                "url": "https://example.test/d",
                "provider": "tavily",
                "title": "Tavily D",
                "description": "D description",
            },
        ],
        "sources_count": 4,
    }


@pytest.mark.asyncio
async def test_web_search_keeps_structured_precedence_over_inline_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answer_text = (
        "Read [Inline B](https://example.test/b) before "
        "[Fallback A](https://example.test/a)."
    )

    class _FakeGrokProvider:
        def __init__(self, api_url: str, api_key: str, model: str) -> None:
            pass

        async def search(self, query: str, platform: str) -> SearchOutput:
            return SearchOutput(
                content=answer_text,
                sources=(
                    NormalizedSource(
                        url="https://example.test/a",
                        title="Structured A",
                        provider="grok",
                    ),
                ),
            )

    monkeypatch.setenv("GROK_API_URL", "https://grok.test/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-grok-key")
    monkeypatch.setattr(server, "GrokSearchProvider", _FakeGrokProvider)

    search_response = await server.web_search("query")
    source_response = await server.get_sources(search_response["session_id"])

    assert search_response == {
        "session_id": search_response["session_id"],
        "content": answer_text,
        "sources_count": 2,
    }
    assert source_response == {
        "session_id": search_response["session_id"],
        "sources": [
            {
                "url": "https://example.test/a",
                "title": "Structured A",
                "provider": "grok",
            },
            {"url": "https://example.test/b", "title": "Inline B"},
        ],
        "sources_count": 2,
    }


@pytest.mark.asyncio
async def test_web_search_surfaces_provider_exception_instead_of_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BoomGrokProvider:
        def __init__(self, api_url: str, api_key: str, model: str) -> None:
            pass

        async def search(self, query: str, platform: str) -> SearchOutput:
            raise TimeoutError("read timeout")

    monkeypatch.setenv("GROK_API_URL", "https://grok.test/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-grok-key")
    monkeypatch.setattr(server, "GrokSearchProvider", _BoomGrokProvider)

    search_response = await server.web_search("query")

    assert search_response["content"].startswith("upstream_error:")
    assert "TimeoutError" in search_response["content"]
    assert search_response["sources_count"] == 0


@pytest.mark.asyncio
async def test_web_search_surfaces_empty_stream_instead_of_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROK_API_URL", "https://grok.test/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-grok-key")
    monkeypatch.setattr(server, "GrokSearchProvider", _EmptyGrokProvider)

    search_response = await server.web_search("query")

    assert search_response["content"].startswith("upstream_empty:")
    assert search_response["sources_count"] == 0


@pytest.mark.asyncio
async def test_web_search_http_status_error_does_not_cache_upstream_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StatusGrokProvider:
        def __init__(self, api_url: str, api_key: str, model: str) -> None:
            pass

        async def search(self, query: str, platform: str) -> SearchOutput:
            request = httpx.Request(
                "POST", "http://127.0.0.1:8080/grok/v1/chat/completions"
            )
            response = httpx.Response(502, request=request)
            raise httpx.HTTPStatusError(
                "Server error '502 Bad Gateway' for url "
                "'http://127.0.0.1:8080/grok/v1/chat/completions' "
                "For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/502",
                request=request,
                response=response,
            )

    monkeypatch.setenv("GROK_API_URL", "https://grok.test/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-grok-key")
    monkeypatch.setattr(server, "GrokSearchProvider", _StatusGrokProvider)

    search_response = await server.web_search("query")
    source_response = await server.get_sources(search_response["session_id"])

    assert search_response["content"].startswith("upstream_error:")
    assert "HTTPStatusError" in search_response["content"]
    assert "http://" not in search_response["content"]
    assert "https://" not in search_response["content"]
    assert search_response["sources_count"] == 0
    assert source_response["sources"] == []
    assert source_response["sources_count"] == 0


@pytest.mark.asyncio
async def test_web_search_envelope_log_goes_to_stderr_not_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("GROK_API_URL", "https://grok.test/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-grok-key")
    monkeypatch.setattr(server, "GrokSearchProvider", _EmptyGrokProvider)

    await server.web_search("query")
    captured = capsys.readouterr()

    assert "web_search envelope" not in captured.out
    assert "web_search envelope" in captured.err
    assert "kind=upstream_empty" in captured.err
