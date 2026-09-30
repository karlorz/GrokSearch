import pytest

from grok_search.config import Config, config


@pytest.fixture
def isolated_config(monkeypatch, tmp_path):
    """Isolate Config environment and singleton caches for configuration tests."""
    monkeypatch.setenv("HOME", str(tmp_path))
    for env_var in [
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
        "FIRECRAWL_ENABLED",
        "GROK_API_KEY",
        "GROK_API_URL",
        "GROK_MODEL",
        "GROK_SEARCH_MCP_HOST",
        "GROK_SEARCH_MCP_INTERNAL_TOKEN",
        "GROK_SEARCH_MCP_PATH",
        "GROK_SEARCH_MCP_PORT",
        "GROK_SEARCH_MCP_PUBLIC_URL",
        "GROK_SEARCH_MCP_TOKEN",
        "GROK_SEARCH_MCP_TRANSPORT",
        "GROK_SEARCH_MCP_VERIFY_URL",
        "GUDA_API_KEY",
        "GUDA_BASE_URL",
        "TAVILY_API_KEY",
        "TAVILY_API_URL",
        "TAVILY_ENABLED",
    ]:
        monkeypatch.delenv(env_var, raising=False)

    Config._instance = None
    config._config_file = None
    config._cached_model = None
    yield
    Config._instance = None
    config._config_file = None
    config._cached_model = None
