"""Composed ASGI application for GrokSearch HTTP transport.

Composes Starlette around `mcp.http_app(path="/mcp", transport="http")`
preserving FastMCP lifespan and bearer auth for `/mcp`, while serving
public CLI auth endpoints (`/auth/cli/*`) without MCP bearer middleware.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastmcp.server.auth import TokenVerifier
from starlette.applications import Starlette
from starlette.routing import Mount, Route

from grok_search.cli_auth import CliAuthCoordinator, resolve_cli_auth_config
from grok_search.rest_api import RestApiHandlers, create_rest_routes


def create_composed_app(
    mcp_app: Starlette,
    coordinator: CliAuthCoordinator | None = None,
    verifier: TokenVerifier | None = None,
    rest_handlers: RestApiHandlers | None = None,
) -> Starlette:
    """Create a composed Starlette app combining CLI auth routes, REST API, and FastMCP app.

    Args:
        mcp_app: The underlying FastMCP Starlette app created via mcp.http_app().
        coordinator: Optional CliAuthCoordinator instance. If None, one is created
            using environment-resolved configuration.
        verifier: Optional TokenVerifier for authenticating REST API requests.
            If not passed, falls back to mcp_app.state.fastmcp_server.auth if present.
        rest_handlers: Optional RestApiHandlers instance. If None, one is created
            using verifier.

    Returns:
        A Starlette application with /auth/cli/*, /api/v1/* and /mcp routes.
    """
    if coordinator is None:
        coordinator = CliAuthCoordinator(config=resolve_cli_auth_config())

    cli_routes = [
        Route("/auth/cli/start", coordinator.handle_start, methods=["POST"]),
        Route("/auth/cli/approve", coordinator.handle_approve_get, methods=["GET"]),
        Route("/auth/cli/approve", coordinator.handle_approve_post, methods=["POST"]),
        Route("/auth/cli/callback", coordinator.handle_callback, methods=["GET"]),
        Route("/auth/cli/check", coordinator.handle_check, methods=["GET"]),
    ]

    effective_verifier = verifier
    if effective_verifier is None:
        fastmcp_srv = getattr(mcp_app.state, "fastmcp_server", None)
        if fastmcp_srv is not None and getattr(fastmcp_srv, "auth", None) is not None:
            effective_verifier = fastmcp_srv.auth

    if rest_handlers is None:
        rest_handlers = RestApiHandlers(verifier=effective_verifier)
    elif effective_verifier is not None and rest_handlers.verifier is None:
        rest_handlers.verifier = effective_verifier

    rest_routes = create_rest_routes(rest_handlers)

    @asynccontextmanager
    async def composed_lifespan(app: Starlette) -> AsyncGenerator[None, None]:
        # Delegate lifespan directly to the mcp_app's lifespan context
        lifespan_ctx = getattr(mcp_app.router, "lifespan_context", None)
        if lifespan_ctx is not None:
            async with lifespan_ctx(mcp_app):
                yield
        else:
            yield

    # Mount mcp_app after CLI and REST routes so they are dispatched directly
    routes = [
        *cli_routes,
        *rest_routes,
        Mount("", app=mcp_app),
    ]

    app = Starlette(
        routes=routes,
        lifespan=composed_lifespan,
    )
    # Expose state attributes matching mcp_app for compatibility
    app.state.fastmcp_server = getattr(mcp_app.state, "fastmcp_server", None)
    app.state.path = getattr(mcp_app.state, "path", "/mcp")
    app.state.transport_type = getattr(mcp_app.state, "transport_type", "streamable-http")
    app.state.cli_coordinator = coordinator
    app.state.rest_handlers = rest_handlers
    return app
