"""M2 authenticated REST API handlers and routing.

Provides POST /api/v1/{search,fetch,map} with Bearer token authentication,
strict JSON input limits, URL validation against private/loopback targets,
and uniform {ok: true, data: ...} / {ok: false, error: ...} envelopes.
"""

from __future__ import annotations

import json
from ipaddress import ip_address
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

from fastmcp.server.auth import TokenVerifier
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from grok_search.providers.contracts import is_failure_envelope

MAX_REQUEST_BODY_BYTES = 64 * 1024  # 64 KB max body


def error_response(
    code: str,
    message: str,
    status_code: int = 400,
    headers: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> JSONResponse:
    err: dict[str, Any] = {"code": code, "message": message}
    if extra:
        err.update(extra)
    return JSONResponse(
        {"ok": False, "error": err},
        status_code=status_code,
        headers=headers,
    )


def success_response(data: Any, status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        {"ok": True, "data": data},
        status_code=status_code,
    )


def validate_user_url(raw_url: Any) -> str:
    """Validate that raw_url is a safe, absolute HTTP(S) URL.

    Rejects credentials, loopback/private/unspecified/multicast/link-local/reserved IPs,
    and missing hosts.
    """
    if not isinstance(raw_url, str):
        raise ValueError("url must be a string")
    url = raw_url.strip()
    if not url:
        raise ValueError("url must not be empty")

    parsed = urlsplit(url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError(f"Invalid URL scheme {parsed.scheme!r}. Only HTTP and HTTPS are allowed.")

    if parsed.username or parsed.password:
        raise ValueError("URL must not contain user credentials")

    hostname = parsed.hostname
    if not hostname:
        if parsed.netloc == "::1" or parsed.netloc.startswith("::1:"):
            hostname = "::1"
        else:
            raise ValueError("URL must contain a valid host")

    lower_host = hostname.lower()
    if lower_host in ("localhost", "ip6-localhost", "ip6-loopback"):
        raise ValueError("Localhost targets are not allowed")

    try:
        ip = ip_address(lower_host)
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ValueError(f"Private, loopback, or reserved IP address is not allowed: {hostname}")
    except ValueError as e:
        if "Private, loopback, or reserved IP" in str(e):
            raise

    return url


class RestApiHandlers:
    """REST handlers bound to a TokenVerifier and service functions."""

    def __init__(
        self,
        verifier: TokenVerifier | None = None,
        search_fn: Callable[..., Awaitable[tuple[dict, list[dict]]]] | None = None,
        fetch_fn: Callable[..., Awaitable[str]] | None = None,
        map_fn: Callable[..., Awaitable[str]] | None = None,
    ) -> None:
        self.verifier = verifier
        self._search_fn = search_fn
        self._fetch_fn = fetch_fn
        self._map_fn = map_fn

    async def _get_search_fn(self):
        if self._search_fn is not None:
            return self._search_fn
        from grok_search.server import execute_search
        return execute_search

    async def _get_fetch_fn(self):
        if self._fetch_fn is not None:
            return self._fetch_fn
        from grok_search.server import execute_fetch
        return execute_fetch

    async def _get_map_fn(self):
        if self._map_fn is not None:
            return self._map_fn
        from grok_search.server import execute_map
        return execute_map

    async def _authenticate(self, request: Request) -> JSONResponse | None:
        """Enforce Bearer authentication. Returns 401 error response on failure, None on success."""
        if self.verifier is None:
            return error_response(
                code="service_unavailable",
                message="Authentication verifier not configured",
                status_code=503,
            )

        auth_header = request.headers.get("authorization", "")
        if not auth_header:
            return error_response(
                code="unauthorized",
                message="Missing Authorization header",
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        parts = auth_header.split(None, 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return error_response(
                code="unauthorized",
                message="Authorization header must use Bearer scheme",
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = parts[1].strip()
        if not token:
            return error_response(
                code="unauthorized",
                message="Empty bearer token",
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        try:
            token_obj = await self.verifier.verify_token(token)
        except Exception:
            token_obj = None

        if token_obj is None:
            return error_response(
                code="unauthorized",
                message="Invalid or expired bearer token",
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        return None

    async def _read_json_body(self, request: Request) -> tuple[dict | None, JSONResponse | None]:
        """Read and validate bounded JSON request body."""
        content_length_str = request.headers.get("content-length")
        if content_length_str:
            try:
                if int(content_length_str) > MAX_REQUEST_BODY_BYTES:
                    return None, error_response(
                        code="payload_too_large",
                        message=f"Request body exceeds {MAX_REQUEST_BODY_BYTES} bytes limit",
                        status_code=413,
                    )
            except ValueError:
                return None, error_response(code="bad_request", message="Invalid Content-Length header", status_code=400)

        body = await request.body()
        if len(body) > MAX_REQUEST_BODY_BYTES:
            return None, error_response(
                code="payload_too_large",
                message=f"Request body exceeds {MAX_REQUEST_BODY_BYTES} bytes limit",
                status_code=413,
            )

        if not body:
            return None, error_response(code="bad_request", message="Request body must not be empty", status_code=400)

        try:
            data = json.loads(body.decode("utf-8"))
        except Exception as e:
            return None, error_response(code="invalid_json", message=f"Invalid JSON: {str(e)}", status_code=400)

        if not isinstance(data, dict):
            return None, error_response(code="invalid_json", message="JSON body must be an object", status_code=400)

        return data, None

    async def _authenticated_json(self, request: Request) -> tuple[dict | None, JSONResponse | None]:
        auth_err = await self._authenticate(request)
        if auth_err:
            return None, auth_err
        return await self._read_json_body(request)

    async def _authenticated_url(self, request: Request) -> tuple[dict | None, str | None, JSONResponse | None]:
        data, err = await self._authenticated_json(request)
        if err:
            return None, None, err
        assert data is not None
        raw_url = data.get("url")
        if not raw_url:
            return None, None, error_response(code="invalid_param", message="'url' is required", status_code=400)
        try:
            return data, validate_user_url(raw_url), None
        except ValueError as e:
            return None, None, error_response(code="invalid_url", message=str(e), status_code=400)

    async def handle_search(self, request: Request) -> Response:
        data, body_err = await self._authenticated_json(request)
        if body_err:
            return body_err
        assert data is not None

        query = data.get("query")
        if not isinstance(query, str) or not query.strip():
            return error_response(code="invalid_param", message="'query' string is required and cannot be empty", status_code=400)

        platform = data.get("platform", "")
        if not isinstance(platform, str):
            return error_response(code="invalid_param", message="'platform' must be a string", status_code=400)

        model = data.get("model", "")
        if not isinstance(model, str):
            return error_response(code="invalid_param", message="'model' must be a string", status_code=400)

        extra_sources = data.get("extra_sources", 0)
        if not isinstance(extra_sources, int) or isinstance(extra_sources, bool):
            return error_response(code="invalid_param", message="'extra_sources' must be an integer", status_code=400)
        if extra_sources < 0:
            return error_response(code="invalid_param", message="'extra_sources' must be non-negative", status_code=400)

        search_fn = await self._get_search_fn()
        mcp_res, sources_list = await search_fn(
            query=query.strip(),
            platform=platform.strip(),
            model=model.strip(),
            extra_sources=extra_sources,
        )

        content = mcp_res.get("content", "")
        session_id = mcp_res.get("session_id", "")

        # Check for upstream failure envelopes or empty content
        if not content or is_failure_envelope(content):
            return error_response(
                code="upstream_error",
                message=content or "upstream returned empty content",
                status_code=502,
                extra={"session_id": session_id},
            )

        if content.startswith("配置错误:"):
            return error_response(
                code="configuration_error",
                message=content,
                status_code=502,
                extra={"session_id": session_id},
            )

        if content.startswith("无效模型:"):
            return error_response(
                code="invalid_model",
                message=content,
                status_code=400,
                extra={"session_id": session_id},
            )

        return success_response({
            "content": content,
            "sources": sources_list,
            "session_id": session_id,
            "sources_count": len(sources_list),
        })

    async def handle_fetch(self, request: Request) -> Response:
        _data, valid_url, err = await self._authenticated_url(request)
        if err:
            return err
        assert valid_url is not None

        fetch_fn = await self._get_fetch_fn()
        content = await fetch_fn(valid_url)

        if not content:
            return error_response(
                code="upstream_empty",
                message="fetch returned empty content",
                status_code=502,
            )

        if content.startswith("配置错误:"):
            return error_response(code="configuration_error", message=content, status_code=502)
        if content.startswith("提取失败:"):
            return error_response(code="upstream_error", message=content, status_code=502)

        return success_response({"content": content})

    async def handle_map(self, request: Request) -> Response:
        data, valid_url, err = await self._authenticated_url(request)
        if err:
            return err
        assert data is not None and valid_url is not None

        instructions = data.get("instructions", "")
        if not isinstance(instructions, str):
            return error_response(code="invalid_param", message="'instructions' must be a string", status_code=400)

        max_depth = data.get("max_depth", 1)
        if not isinstance(max_depth, int) or isinstance(max_depth, bool) or not (1 <= max_depth <= 5):
            return error_response(code="invalid_param", message="'max_depth' must be an integer between 1 and 5", status_code=400)

        max_breadth = data.get("max_breadth", 20)
        if not isinstance(max_breadth, int) or isinstance(max_breadth, bool) or not (1 <= max_breadth <= 500):
            return error_response(code="invalid_param", message="'max_breadth' must be an integer between 1 and 500", status_code=400)

        limit = data.get("limit", 50)
        if not isinstance(limit, int) or isinstance(limit, bool) or not (1 <= limit <= 500):
            return error_response(code="invalid_param", message="'limit' must be an integer between 1 and 500", status_code=400)

        timeout = data.get("timeout", 150)
        if not isinstance(timeout, int) or isinstance(timeout, bool) or not (10 <= timeout <= 150):
            return error_response(code="invalid_param", message="'timeout' must be an integer between 10 and 150", status_code=400)

        map_fn = await self._get_map_fn()
        content = await map_fn(
            url=valid_url,
            instructions=instructions,
            max_depth=max_depth,
            max_breadth=max_breadth,
            limit=limit,
            timeout=timeout,
        )

        if not content:
            return error_response(
                code="upstream_empty",
                message="map returned empty content",
                status_code=502,
            )

        if content.startswith("配置错误:"):
            return error_response(code="configuration_error", message=content, status_code=502)
        if content.startswith("映射超时:"):
            return error_response(code="upstream_timeout", message=content, status_code=504)
        if content.startswith("HTTP错误:") or content.startswith("映射错误:"):
            return error_response(code="upstream_error", message=content, status_code=502)

        return success_response({"content": content})


async def handle_method_not_allowed(request: Request) -> Response:
    """Return uniform 405 Method Not Allowed error."""
    return error_response(
        code="method_not_allowed",
        message=f"Method {request.method} is not allowed for this route",
        status_code=405,
    )


def create_rest_routes(handlers: RestApiHandlers) -> list[Route]:
    """Create REST API routes with explicit 405 handlers for unsupported methods."""
    disallowed_methods = ["GET", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
    return [
        Route("/api/v1/search", handlers.handle_search, methods=["POST"]),
        Route("/api/v1/search", handle_method_not_allowed, methods=disallowed_methods),
        Route("/api/v1/fetch", handlers.handle_fetch, methods=["POST"]),
        Route("/api/v1/fetch", handle_method_not_allowed, methods=disallowed_methods),
        Route("/api/v1/map", handlers.handle_map, methods=["POST"]),
        Route("/api/v1/map", handle_method_not_allowed, methods=disallowed_methods),
    ]
