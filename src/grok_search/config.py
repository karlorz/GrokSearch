import os
import json
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlsplit


class Config:
    _instance = None
    _SETUP_COMMAND = (
        'claude mcp add-json grok-search --scope user '
        '\'{"type":"stdio","command":"uvx","args":["--from",'
        '"git+https://github.com/GuDaStudio/GrokSearch","grok-search"],'
        '"env":{"GUDA_API_KEY":"your-guda-api-key"}}\''
    )
    _DEFAULT_MODEL = "grok-4.3-fast"
    _DEPRECATED_MODELS = {
        "grok-4.20-beta": "grok-4.3-fast",
    }
    _DEFAULT_GUDA_BASE_URL = "https://code.guda.studio"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config_file = None
            cls._instance._cached_model = None
        return cls._instance

    @property
    def config_file(self) -> Path:
        if self._config_file is None:
            config_dir = Path.home() / ".config" / "grok-search"
            try:
                config_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                config_dir = Path.cwd() / ".grok-search"
                config_dir.mkdir(parents=True, exist_ok=True)
            self._config_file = config_dir / "config.json"
        return self._config_file

    def _load_config_file(self) -> dict:
        if not self.config_file.exists():
            return {}
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    return {}
                if "model" in data and self.is_deprecated_model(data["model"]):
                    data["model"] = self.canonicalize_model(data["model"])
                    try:
                        self._save_config_file(data)
                    except Exception:
                        pass
                return data
        except (json.JSONDecodeError, IOError):
            return {}

    def _save_config_file(self, config_data: dict) -> None:
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            raise ValueError(f"无法保存配置文件: {str(e)}")

    @property
    def debug_enabled(self) -> bool:
        return os.getenv("GROK_DEBUG", "false").lower() in ("true", "1", "yes")

    @property
    def retry_max_attempts(self) -> int:
        return int(os.getenv("GROK_RETRY_MAX_ATTEMPTS", "3"))

    @property
    def retry_multiplier(self) -> float:
        return float(os.getenv("GROK_RETRY_MULTIPLIER", "1"))

    @property
    def retry_max_wait(self) -> int:
        return int(os.getenv("GROK_RETRY_MAX_WAIT", "10"))

    @property
    def guda_base_url(self) -> str:
        return os.getenv("GUDA_BASE_URL", self._DEFAULT_GUDA_BASE_URL)

    @property
    def guda_api_key(self) -> str | None:
        return os.getenv("GUDA_API_KEY")

    @property
    def mcp_public_url(self) -> str | None:
        """Return the deployment's public MCP endpoint when it is advertised."""
        public_url = os.getenv("GROK_SEARCH_MCP_PUBLIC_URL", "").strip()
        if not public_url:
            return None
        if any(ord(character) <= 32 or ord(character) == 127 for character in public_url):
            return None
        try:
            parsed_url = urlsplit(public_url)
            hostname = parsed_url.hostname
            _ = parsed_url.port
        except ValueError:
            return None
        if (
            parsed_url.scheme not in {"http", "https"}
            or not hostname
            or parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.query
            or parsed_url.fragment
        ):
            return None
        normalized_hostname = hostname.lower().rstrip(".")
        if normalized_hostname == "localhost" or normalized_hostname.endswith(
            (".localhost", ".local", ".internal")
        ):
            return None
        try:
            address = ip_address(normalized_hostname)
        except ValueError:
            is_ambiguous_ipv4 = all(
                character in "0123456789." for character in normalized_hostname
            ) or (
                normalized_hostname.startswith("0x")
                and all(
                    character in "0123456789abcdef"
                    for character in normalized_hostname[2:]
                )
            )
            if is_ambiguous_ipv4:
                return None
        else:
            if (
                address.is_private
                or address.is_loopback
                or address.is_link_local
                or address.is_reserved
                or address.is_unspecified
            ):
                return None
        return public_url

    @property
    def grok_api_url(self) -> str:
        url = os.getenv("GROK_API_URL")
        if not url:
            if self.guda_api_key:
                return f"{self.guda_base_url}/grok/v1"
            raise ValueError(
                f"Grok API URL 未配置！\n"
                f"请使用以下命令配置 MCP 服务器：\n{self._SETUP_COMMAND}"
            )
        return url

    @property
    def grok_api_key(self) -> str:
        key = os.getenv("GROK_API_KEY") or self.guda_api_key
        if not key:
            raise ValueError(
                f"Grok API Key 未配置！\n"
                f"请使用以下命令配置 MCP 服务器：\n{self._SETUP_COMMAND}"
            )
        return key

    @property
    def tavily_enabled(self) -> bool:
        return os.getenv("TAVILY_ENABLED", "true").lower() in ("true", "1", "yes")

    @property
    def firecrawl_enabled(self) -> bool:
        return os.getenv("FIRECRAWL_ENABLED", "true").lower() in ("true", "1", "yes")

    @property
    def tavily_api_url(self) -> str:
        url = os.getenv("TAVILY_API_URL")
        if not url and self.guda_api_key:
            return f"{self.guda_base_url}/tavily"
        return url or "https://api.tavily.com"

    @property
    def tavily_api_key(self) -> str | None:
        return os.getenv("TAVILY_API_KEY") or self.guda_api_key

    @property
    def firecrawl_api_url(self) -> str:
        url = os.getenv("FIRECRAWL_API_URL")
        if not url and self.guda_api_key:
            return f"{self.guda_base_url}/firecrawl"
        return url or "https://api.firecrawl.dev/v2"

    @property
    def firecrawl_api_key(self) -> str | None:
        return os.getenv("FIRECRAWL_API_KEY") or self.guda_api_key

    @property
    def log_level(self) -> str:
        return os.getenv("GROK_LOG_LEVEL", "INFO").upper()

    @property
    def log_dir(self) -> Path:
        log_dir_str = os.getenv("GROK_LOG_DIR", "logs")
        log_dir = Path(log_dir_str)
        if log_dir.is_absolute():
            return log_dir

        home_log_dir = Path.home() / ".config" / "grok-search" / log_dir_str
        try:
            home_log_dir.mkdir(parents=True, exist_ok=True)
            return home_log_dir
        except OSError:
            pass

        cwd_log_dir = Path.cwd() / log_dir_str
        try:
            cwd_log_dir.mkdir(parents=True, exist_ok=True)
            return cwd_log_dir
        except OSError:
            pass

        tmp_log_dir = Path("/tmp") / "grok-search" / log_dir_str
        tmp_log_dir.mkdir(parents=True, exist_ok=True)
        return tmp_log_dir

    @classmethod
    def canonicalize_model(cls, model: str | None) -> str | None:
        """Canonicalize deprecated model aliases to their current equivalents."""
        if not model:
            return model
        return cls._DEPRECATED_MODELS.get(model, model)

    @classmethod
    def is_deprecated_model(cls, model: str | None) -> bool:
        """Check whether the given model name is a deprecated alias."""
        if not model:
            return False
        return model in cls._DEPRECATED_MODELS

    def _apply_model_suffix(self, model: str) -> str:
        try:
            url = self.grok_api_url
        except ValueError:
            return model
        if "openrouter" in url and ":online" not in model:
            return f"{model}:online"
        return model

    @property
    def grok_model(self) -> str:
        if self._cached_model is not None:
            return self._cached_model

        raw_model = (
            os.getenv("GROK_MODEL")
            or self._load_config_file().get("model")
            or self._DEFAULT_MODEL
        )
        canonical = self.canonicalize_model(raw_model) or self._DEFAULT_MODEL
        self._cached_model = self._apply_model_suffix(canonical)
        return self._cached_model

    def set_model(self, model: str) -> None:
        canonical = self.canonicalize_model(model) or model
        config_data = self._load_config_file()
        config_data["model"] = canonical
        self._save_config_file(config_data)
        self._cached_model = self._apply_model_suffix(canonical)

    def get_config_info(self) -> dict:
        """Return a client-safe summary without secrets or internal routing details."""
        try:
            self.grok_api_url
            self.grok_api_key
            config_status = "complete"
        except ValueError:
            config_status = "incomplete"

        from .mcp_transport import (
            McpHttpConfigError,
            resolve_run_settings,
            resolve_transport,
        )

        public_url = self.mcp_public_url
        try:
            run_settings = resolve_run_settings()
            configured_transport = run_settings.transport
        except McpHttpConfigError:
            run_settings = None
            try:
                configured_transport = resolve_transport()
            except McpHttpConfigError:
                configured_transport = "invalid"

        if public_url and run_settings is not None and run_settings.transport == "http":
            mcp_connection = {
                "status": "remote_engine_active",
                "transport": "streamable_http",
                "message": (
                    "This response is from the configured remote Grok Search engine. "
                    "No local Grok Search, GUDA, Tavily, or Firecrawl service is required."
                ),
            }
        elif public_url and configured_transport == "http":
            mcp_connection = {
                "status": "http_configuration_invalid",
                "transport": "streamable_http",
                "message": (
                    "A public MCP endpoint is advertised, but this process has an "
                    "invalid HTTP transport configuration."
                ),
            }
        elif public_url:
            mcp_connection = {
                "status": "public_endpoint_advertised",
                "transport": configured_transport,
                "message": (
                    "A public MCP endpoint is advertised, but this process is not "
                    "configured for HTTP transport."
                ),
            }
        else:
            mcp_connection = {
                "status": "public_endpoint_not_advertised",
                "transport": (
                    "streamable_http"
                    if configured_transport == "http"
                    else configured_transport
                ),
                "message": "This deployment has not advertised a public MCP endpoint.",
            }

        if run_settings is not None and run_settings.transport == "http":
            authentication = (
                "gateway_user_bearer"
                if run_settings.verify_url
                else "static_bearer"
            )
        elif configured_transport == "stdio":
            authentication = "not_applicable"
        else:
            authentication = "configuration_invalid"

        mcp_connection.update(
            {
                "public_endpoint": public_url,
                "client_configuration": {
                    "endpoint_env": "GROK_SEARCH_MCP_URL",
                    "bearer_token_env": "GROK_SEARCH_MCP_TOKEN",
                    "authentication": authentication,
                },
            }
        )

        raw_env_model = os.getenv("GROK_MODEL")
        deprecation_note = None
        if self.is_deprecated_model(raw_env_model):
            deprecation_note = f"'{raw_env_model}' 已废弃，已自动规范化为 '{self.canonicalize_model(raw_env_model)}'"

        info = {
            "mcp_connection": mcp_connection,
            "remote_engine": {
                "model": self.grok_model,
                "tavily_enabled": self.tavily_enabled,
                "firecrawl_enabled": self.firecrawl_enabled,
            },
            "config_status": config_status,
        }
        if deprecation_note:
            info["model_deprecation"] = deprecation_note
        return info

config = Config()
