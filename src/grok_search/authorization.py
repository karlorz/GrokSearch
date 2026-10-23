"""Permissions for operations that change the server's persistent settings."""

from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_access_token, get_http_request


def require_config_write() -> None:
    try:
        get_http_request()
    except RuntimeError:
        # Local stdio retains the operator's existing filesystem authority.
        return
    token = get_access_token()
    if token is not None and (
        "config:write" in token.scopes
        or token.claims.get("oauth_owned") is False
    ):
        return
    raise ToolError("Changing server settings requires config:write permission.")
