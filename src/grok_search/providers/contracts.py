from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit


@dataclass(frozen=True, slots=True)
class NormalizedSource:
    """Provider-independent citation data retained from an upstream response."""

    url: str
    title: str | None = None
    description: str | None = None
    provider: str | None = None

    def __post_init__(self) -> None:
        url = self.url.strip()
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("source URL must be an absolute HTTP(S) URL")
        object.__setattr__(self, "url", url)
        for key in ("title", "description", "provider"):
            value = getattr(self, key)
            object.__setattr__(
                self,
                key,
                value.strip() if isinstance(value, str) and value.strip() else None,
            )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        provider: str | None = None,
    ) -> "NormalizedSource | None":
        nested = payload.get("url_citation")
        if isinstance(nested, Mapping):
            payload = nested

        url = payload.get("url") or payload.get("href") or payload.get("link")
        if not isinstance(url, str):
            return None
        url = url.strip()
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return None

        def _first_text(*keys: str) -> str | None:
            for key in keys:
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return None

        return cls(
            url=url,
            title=_first_text("title", "name", "label"),
            description=_first_text("description", "snippet", "content"),
            provider=provider or _first_text("provider"),
        )

    def to_dict(self) -> dict[str, str]:
        result = {"url": self.url}
        for key in ("title", "description", "provider"):
            value = getattr(self, key)
            if value:
                result[key] = value
        return result

    def enriched_with(self, other: "NormalizedSource") -> "NormalizedSource":
        """Keep first-seen values while filling metadata absent from this source."""

        if self.url != other.url:
            raise ValueError("sources with different URLs cannot be merged")
        return NormalizedSource(
            url=self.url,
            title=self.title or other.title,
            description=self.description or other.description,
            provider=self.provider or other.provider,
        )


def merge_normalized_sources(
    *source_lists: Iterable[NormalizedSource],
) -> tuple[NormalizedSource, ...]:
    """Merge exact URLs in first-seen order and fill only missing metadata."""

    merged: dict[str, NormalizedSource] = {}
    for sources in source_lists:
        for source in sources:
            existing = merged.get(source.url)
            merged[source.url] = (
                existing.enriched_with(source) if existing is not None else source
            )
    return tuple(merged.values())


@dataclass(frozen=True, slots=True)
class SearchOutput:
    """Internal Grok search result; the public MCP response remains unchanged."""

    content: str = ""
    sources: tuple[NormalizedSource, ...] = ()


UPSTREAM_ERROR_PREFIX = "upstream_error:"
UPSTREAM_EMPTY_PREFIX = "upstream_empty:"
_URL_IN_TEXT = re.compile(r"https?://\S+", re.IGNORECASE)


def envelope_kind(content: str) -> str:
    if content.startswith(UPSTREAM_ERROR_PREFIX):
        return "upstream_error"
    if content.startswith(UPSTREAM_EMPTY_PREFIX):
        return "upstream_empty"
    return "ok"


def is_failure_envelope(content: str) -> bool:
    return envelope_kind(content) != "ok"


def envelope_from_grok_result(
    result: SearchOutput | None = None,
    error: BaseException | None = None,
) -> SearchOutput:
    """Turn a Grok provider result or exception into a non-blank MCP content envelope."""
    if error is not None:
        detail = f"{type(error).__name__}: {error}"
        detail = _URL_IN_TEXT.sub("[url]", " ".join(detail.split()))[:240]
        return SearchOutput(content=f"{UPSTREAM_ERROR_PREFIX} {detail}")
    result = result or SearchOutput()
    if (result.content or "").strip():
        return result
    if result.sources:
        return result
    return SearchOutput(
        content=f"{UPSTREAM_EMPTY_PREFIX} grok stream completed with no answer content"
    )
