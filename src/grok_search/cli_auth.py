"""Hosted approve+poll coordinator for GrokSearch CLI authentication.

Provides the state machine, PKCE generation, CSRF/cookie handling,
and HTTP endpoint handlers for the CLI authentication bridge.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import os
import secrets
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

import httpx
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response


class AuthRunState(str, Enum):
    PENDING = "pending"
    AWAITING_GATEWAY = "awaiting_gateway"
    READY = "ready"
    CLAIMING = "claiming"
    CONSUMED = "consumed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    FAILED = "failed"


_CSRF_COOKIE_NAME = "grok_search_cli_auth"

_SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "Pragma": "no-cache",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": (
        "default-src 'none'; style-src 'unsafe-inline'; form-action 'self' https:; frame-ancestors 'none'"
    ),
}


def _hash_secret(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _generate_pairing_code() -> str:
    # 6 uppercase alphanumeric characters (anti-confusion aid)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(6))


def _generate_pkce_pair() -> tuple[str, str]:
    verifier_bytes = secrets.token_bytes(32)
    verifier = base64.urlsafe_b64encode(verifier_bytes).decode("ascii").rstrip("=")
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge


@dataclass
class AuthRun:
    run_id: str
    poll_secret_hash: str
    approval_ref: str
    pairing_code: str
    created_at: float
    expires_at: float
    client_name: str = "GrokSearch CLI"
    state: AuthRunState = AuthRunState.PENDING
    csrf_token_hash: str | None = None
    oauth_state: str | None = None
    pkce_verifier: str | None = None
    gateway_code: str | None = None
    gateway_code_expires_at: float | None = None
    error_message: str | None = None
    agent_id: str | None = None
    invite_code: str | None = None

    def is_expired(self, now: float | None = None) -> bool:
        current = now if now is not None else time.time()
        return current >= self.expires_at


@dataclass
class CliAuthConfig:
    origin: str = "https://search.karldigi.dev"
    oauth_client_id: str = ""
    oauth_redirect_uri: str = "https://search.karldigi.dev/auth/cli/callback"
    gateway_authorize_url: str = "https://search.karldigi.dev/authorize"
    gateway_token_url: str = "http://127.0.0.1:8080/token"
    run_ttl_seconds: int = 600
    poll_interval_seconds: int = 2
    max_active_runs: int = 1000
    rate_limit_per_min: int = 60


def resolve_cli_auth_config(environ: Mapping[str, str] | None = None) -> CliAuthConfig:
    source = os.environ if environ is None else environ

    origin = source.get("GROK_SEARCH_PUBLIC_ORIGIN", "https://search.karldigi.dev").rstrip("/")
    client_id = source.get("GROK_SEARCH_CLI_OAUTH_CLIENT_ID", "").strip()
    redirect_uri = source.get(
        "GROK_SEARCH_CLI_OAUTH_REDIRECT_URI",
        f"{origin}/auth/cli/callback",
    ).strip()

    # Browser authorize redirects to public issuer or GROK_SEARCH_MCP_OAUTH_ISSUER
    issuer = source.get("GROK_SEARCH_MCP_OAUTH_ISSUER", origin).rstrip("/")
    gateway_authorize_url = f"{issuer}/authorize"

    # Gateway loopback base for server-side token exchange
    guda_base = source.get("GUDA_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
    gateway_token_url = f"{guda_base}/token"

    return CliAuthConfig(
        origin=origin,
        oauth_client_id=client_id,
        oauth_redirect_uri=redirect_uri,
        gateway_authorize_url=gateway_authorize_url,
        gateway_token_url=gateway_token_url,
    )


class AuthRunStore:
    """In-memory coordinator store for single-process CLI auth runs."""

    def __init__(self, max_runs: int = 1000):
        self._max_runs = max_runs
        self._runs_by_id: dict[str, AuthRun] = {}
        self._runs_by_approval_ref: dict[str, str] = {}  # ref -> run_id
        self._runs_by_oauth_state: dict[str, str] = {}   # oauth_state -> run_id
        self._rate_limits: dict[str, list[float]] = {}

    def prune_expired(self, now: float | None = None) -> None:
        current = now if now is not None else time.time()
        expired_ids = [
            rid for rid, r in self._runs_by_id.items()
            if r.is_expired(current) or (
                r.state in (AuthRunState.CONSUMED, AuthRunState.CANCELLED, AuthRunState.FAILED)
                and current > r.created_at + 300
            )
        ]
        for rid in expired_ids:
            run = self._runs_by_id.pop(rid, None)
            if run:
                self._runs_by_approval_ref.pop(run.approval_ref, None)
                if run.oauth_state:
                    self._runs_by_oauth_state.pop(run.oauth_state, None)

    def is_rate_limited(self, client_key: str, limit: int = 60, window: float = 60.0) -> bool:
        now = time.time()
        timestamps = self._rate_limits.setdefault(client_key, [])
        self._rate_limits[client_key] = [t for t in timestamps if now - t < window]
        if len(self._rate_limits[client_key]) >= limit:
            return True
        self._rate_limits[client_key].append(now)
        return False

    def create_run(
        self,
        ttl_seconds: int = 600,
        client_name: str = "GrokSearch CLI",
        agent_id: str | None = None,
        invite_code: str | None = None,
    ) -> tuple[AuthRun, str]:
        self.prune_expired()
        if len(self._runs_by_id) >= self._max_runs:
            raise RuntimeError("Maximum active auth runs exceeded")

        run_id = secrets.token_hex(16)
        poll_secret = secrets.token_urlsafe(32)
        poll_secret_hash = _hash_secret(poll_secret)
        approval_ref = secrets.token_urlsafe(24)
        pairing_code = _generate_pairing_code()

        now = time.time()
        expires_at = now + ttl_seconds

        run = AuthRun(
            run_id=run_id,
            poll_secret_hash=poll_secret_hash,
            approval_ref=approval_ref,
            pairing_code=pairing_code,
            created_at=now,
            expires_at=expires_at,
            client_name=client_name,
            state=AuthRunState.PENDING,
            agent_id=agent_id,
            invite_code=invite_code,
        )

        self._runs_by_id[run_id] = run
        self._runs_by_approval_ref[approval_ref] = run_id
        return run, poll_secret

    def get_by_id(self, run_id: str) -> AuthRun | None:
        run = self._runs_by_id.get(run_id)
        if run and run.is_expired():
            run.state = AuthRunState.EXPIRED
        return run

    def get_by_approval_ref(self, ref: str) -> AuthRun | None:
        run_id = self._runs_by_approval_ref.get(ref)
        if not run_id:
            return None
        return self.get_by_id(run_id)

    def get_by_oauth_state(self, state: str) -> AuthRun | None:
        run_id = self._runs_by_oauth_state.get(state)
        if not run_id:
            return None
        return self.get_by_id(run_id)

    def bind_oauth_state(self, run: AuthRun, oauth_state: str) -> None:
        if run.oauth_state and run.oauth_state in self._runs_by_oauth_state:
            del self._runs_by_oauth_state[run.oauth_state]
        run.oauth_state = oauth_state
        self._runs_by_oauth_state[oauth_state] = run.run_id


class CliAuthCoordinator:
    """Coordinator handling /auth/cli/* HTTP endpoints."""

    def __init__(
        self,
        config: CliAuthConfig | None = None,
        store: AuthRunStore | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.config = config or resolve_cli_auth_config()
        self.store = store or AuthRunStore(max_runs=self.config.max_active_runs)
        self._http_client = http_client

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=5.0)
        return self._http_client

    def _apply_security_headers(self, response: Response) -> Response:
        for k, v in _SECURITY_HEADERS.items():
            response.headers[k] = v
        return response

    def _cookie_name(self, request: Request) -> str:
        return _CSRF_COOKIE_NAME

    # --- Handlers ---

    async def handle_start(self, request: Request) -> Response:
        """POST /auth/cli/start"""
        client_ip = request.client.host if request.client else "unknown"
        if self.store.is_rate_limited(f"start:{client_ip}", limit=self.config.rate_limit_per_min):
            return self._apply_security_headers(
                JSONResponse(
                    {"error": "rate_limited", "message": "Too many auth requests"},
                    status_code=429,
                )
            )

        client_name = "GrokSearch CLI"
        agent_id: str | None = None
        invite_code: str | None = None
        # Bounded body reading if present
        try:
            body = await request.body()
            if len(body) > 4096:
                return self._apply_security_headers(
                    JSONResponse({"error": "payload_too_large"}, status_code=413)
                )
            if body:
                data = await request.json()
                if isinstance(data, dict):
                    if "client_name" in data:
                        cname = str(data["client_name"]).strip()
                        if cname and len(cname) <= 64:
                            client_name = cname
                    if "agent_id" in data:
                        aid = str(data["agent_id"]).strip()
                        if aid and len(aid) <= 128:
                            agent_id = aid
                    if "invite_code" in data:
                        icode = str(data["invite_code"]).strip()
                        if icode and len(icode) <= 128:
                            invite_code = icode
        except Exception:
            pass

        if not self.config.oauth_client_id:
            return self._apply_security_headers(
                JSONResponse(
                    {"error": "misconfigured", "message": "GROK_SEARCH_CLI_OAUTH_CLIENT_ID is required"},
                    status_code=503,
                )
            )

        try:
            run, poll_secret = self.store.create_run(
                ttl_seconds=self.config.run_ttl_seconds,
                client_name=client_name,
                agent_id=agent_id,
                invite_code=invite_code,
            )
        except RuntimeError:
            return self._apply_security_headers(
                JSONResponse({"error": "server_busy", "message": "Coordinator capacity reached"}, status_code=503)
            )

        approve_query = [("ref", run.approval_ref)]
        if run.invite_code:
            approve_query.append(("invite", run.invite_code))
        encoded_query = urllib.parse.urlencode(approve_query)
        approve_url = f"{self.config.origin}/auth/cli/approve?{encoded_query}"
        expires_at_iso = datetime.fromtimestamp(run.expires_at, tz=timezone.utc).isoformat()

        response_data = {
            "approveUrl": approve_url,
            "authRunId": run.run_id,
            "pairingCode": run.pairing_code,
            "pollSecret": poll_secret,
            "expiresAt": expires_at_iso,
            "intervalSeconds": self.config.poll_interval_seconds,
        }
        return self._apply_security_headers(JSONResponse(response_data, status_code=201))

    async def handle_approve_get(self, request: Request) -> Response:
        """GET /auth/cli/approve?ref=..."""
        ref = request.query_params.get("ref", "").strip()
        if not ref:
            return self._render_page(
                status_code=400,
                title="Invalid Request",
                content="<p>Missing approval reference.</p>",
            )

        run = self.store.get_by_approval_ref(ref)
        if not run:
            return self._render_page(
                status_code=404,
                title="Request Not Found",
                content="<p>This authorization request was not found or has expired.</p>",
            )

        if run.state != AuthRunState.PENDING or run.is_expired():
            return self._render_page(
                status_code=400,
                title="Request Expired or Inactive",
                content=f"<p>This authorization request is no longer pending (status: {html.escape(run.state.value)}).</p>",
            )

        # Generate a CSRF token bound to this browser session and approval ref
        csrf_token = secrets.token_urlsafe(32)
        run.csrf_token_hash = _hash_secret(csrf_token)

        expiry_time_str = datetime.fromtimestamp(run.expires_at, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        body_html = f"""
        <div class="card">
          <h1>Authorize Access</h1>
          <p>A client is requesting access to search tools via <strong>{html.escape(self.config.origin)}</strong>.</p>
          <div class="info-box">
            <div><strong>Client:</strong> {html.escape(run.client_name)}</div>
            <div><strong>Pairing Code:</strong> <span class="pairing-code">{html.escape(run.pairing_code)}</span></div>
            <div><strong>Expires at:</strong> {html.escape(expiry_time_str)}</div>
          </div>
          <p class="notice">Check that the pairing code matches the code shown in your terminal.</p>
          <div class="button-row">
            <form method="POST" action="/auth/cli/approve">
              <input type="hidden" name="ref" value="{html.escape(ref)}">
              <input type="hidden" name="csrf_token" value="{html.escape(csrf_token)}">
              <input type="hidden" name="action" value="deny">
              <button type="submit" class="btn btn-deny">Deny</button>
            </form>
            <form method="POST" action="/auth/cli/approve">
              <input type="hidden" name="ref" value="{html.escape(ref)}">
              <input type="hidden" name="csrf_token" value="{html.escape(csrf_token)}">
              <input type="hidden" name="action" value="approve">
              <button type="submit" class="btn btn-approve">Approve</button>
            </form>
          </div>
        </div>
        """
        response = self._render_page(status_code=200, title="Authorize GrokSearch CLI", content=body_html)

        cookie_name = self._cookie_name(request)
        is_secure = request.url.scheme == "https"
        response.set_cookie(
            key=cookie_name,
            value=csrf_token,
            max_age=int(run.expires_at - time.time()) + 60,
            httponly=True,
            samesite="lax",
            secure=is_secure,
            path="/auth/cli",
        )
        return response

    async def handle_approve_post(self, request: Request) -> Response:
        """POST /auth/cli/approve"""
        try:
            form = await request.form()
        except Exception:
            return self._render_page(status_code=400, title="Bad Request", content="<p>Invalid form data.</p>")

        ref = str(form.get("ref", "")).strip()
        form_csrf = str(form.get("csrf_token", "")).strip()
        action = str(form.get("action", "")).strip()

        cookie_name = self._cookie_name(request)
        cookie_csrf = request.cookies.get(cookie_name, "").strip()

        if not ref or not form_csrf or not cookie_csrf:
            return self._render_page(status_code=400, title="Forbidden", content="<p>Missing CSRF tokens or reference.</p>")

        if not hmac.compare_digest(form_csrf, cookie_csrf):
            return self._render_page(status_code=403, title="Forbidden", content="<p>CSRF verification failed.</p>")

        run = self.store.get_by_approval_ref(ref)
        if not run:
            return self._render_page(status_code=404, title="Not Found", content="<p>Authorization request not found.</p>")

        if not run.csrf_token_hash or not hmac.compare_digest(_hash_secret(form_csrf), run.csrf_token_hash):
            return self._render_page(status_code=403, title="Forbidden", content="<p>Session token mismatch.</p>")

        if run.state != AuthRunState.PENDING or run.is_expired():
            return self._render_page(status_code=400, title="Expired", content="<p>Request is no longer pending.</p>")

        if action == "deny":
            run.state = AuthRunState.CANCELLED
            return self._render_page(
                status_code=200,
                title="Authorization Denied",
                content="<p>You have denied access. You may close this window and return to your terminal.</p>",
            )

        if action != "approve":
            return self._render_page(status_code=400, title="Bad Request", content="<p>Unknown action.</p>")

        # Approve: generate PKCE pair & OAuth state
        verifier, challenge = _generate_pkce_pair()
        oauth_state = secrets.token_urlsafe(32)

        run.pkce_verifier = verifier
        self.store.bind_oauth_state(run, oauth_state)
        run.state = AuthRunState.AWAITING_GATEWAY

        # Build redirect to gateway /authorize
        params: dict[str, str] = {
            "response_type": "code",
            "client_id": self.config.oauth_client_id,
            "redirect_uri": self.config.oauth_redirect_uri,
            "state": oauth_state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "scope": "mcp",
        }
        if run.invite_code:
            params["invite"] = run.invite_code
        if run.agent_id:
            params["agent_label"] = run.agent_id
        redirect_url = httpx.URL(self.config.gateway_authorize_url, params=params)

        response = RedirectResponse(str(redirect_url), status_code=302)
        return self._apply_security_headers(response)

    async def handle_callback(self, request: Request) -> Response:
        """GET /auth/cli/callback"""
        query_state = request.query_params.get("state", "").strip()
        code = request.query_params.get("code", "").strip()
        error = request.query_params.get("error", "").strip()

        if error:
            return self._render_page(
                status_code=400,
                title="Authorization Error",
                content=f"<p>Gateway reported an error: {html.escape(error)}</p>",
            )

        if not query_state or not code:
            return self._render_page(
                status_code=400,
                title="Invalid Callback",
                content="<p>Missing state or authorization code.</p>",
            )

        run = self.store.get_by_oauth_state(query_state)
        if not run:
            return self._render_page(
                status_code=400,
                title="Invalid State",
                content="<p>No matching active authorization run found for state.</p>",
            )

        if run.state != AuthRunState.AWAITING_GATEWAY or run.is_expired():
            return self._render_page(
                status_code=400,
                title="Invalid State",
                content="<p>Run is not awaiting callback or has expired.</p>",
            )

        # Store code transiently and mark ready
        run.gateway_code = code
        # Gateway codes expire in 5 minutes; bound run to 5 min from now or original expiry, whichever is earlier
        gateway_exp = time.time() + 300
        run.gateway_code_expires_at = min(run.expires_at, gateway_exp)
        run.state = AuthRunState.READY

        # Invalidate oauth_state to prevent callback replay
        if run.oauth_state:
            self.store._runs_by_oauth_state.pop(run.oauth_state, None)
            run.oauth_state = None

        content = f"""
        <div class="card">
          <h1>Authorization Successful</h1>
          <p>Access has been approved for <strong>{html.escape(run.client_name)}</strong>.</p>
          <p>Pairing code: <span class="pairing-code">{html.escape(run.pairing_code)}</span></p>
          <p>You can return to your terminal. The CLI will finish setup automatically.</p>
        </div>
        """
        return self._render_page(status_code=200, title="Authorized", content=content)

    async def handle_check(self, request: Request) -> Response:
        """GET /auth/cli/check?authRunId=... Requires X-CLI-Poll-Secret header."""
        run_id = request.query_params.get("authRunId", "").strip()
        poll_secret = request.headers.get("X-CLI-Poll-Secret", "").strip()

        if not run_id or not poll_secret:
            # Do not enumerate existence
            return self._apply_security_headers(
                JSONResponse({"error": "invalid_request", "message": "Missing run ID or poll secret"}, status_code=400)
            )

        run = self.store.get_by_id(run_id)
        opaque_not_found = JSONResponse(
            {"error": "not_found", "message": "Auth run not found or expired"},
            status_code=404,
        )
        if not run:
            return self._apply_security_headers(opaque_not_found)

        secret_hash = _hash_secret(poll_secret)
        if not hmac.compare_digest(secret_hash, run.poll_secret_hash):
            return self._apply_security_headers(opaque_not_found)

        # Check expiry
        now = time.time()
        if run.is_expired(now):
            run.state = AuthRunState.EXPIRED

        if run.state == AuthRunState.PENDING or run.state == AuthRunState.AWAITING_GATEWAY:
            return self._apply_security_headers(
                JSONResponse({"status": "pending", "authRunId": run.run_id}, status_code=200)
            )

        if run.state == AuthRunState.CANCELLED:
            return self._apply_security_headers(
                JSONResponse({"status": "cancelled", "authRunId": run.run_id}, status_code=200)
            )

        if run.state == AuthRunState.EXPIRED:
            return self._apply_security_headers(
                JSONResponse({"status": "expired", "authRunId": run.run_id}, status_code=200)
            )

        if run.state == AuthRunState.CONSUMED:
            return self._apply_security_headers(
                JSONResponse({"status": "consumed", "authRunId": run.run_id}, status_code=200)
            )

        if run.state == AuthRunState.FAILED:
            return self._apply_security_headers(
                JSONResponse(
                    {
                        "status": "failed",
                        "authRunId": run.run_id,
                        "error": run.error_message or "Authentication exchange failed",
                    },
                    status_code=200,
                )
            )

        if run.state == AuthRunState.READY:
            # Transition to claiming to prevent concurrent poll races
            run.state = AuthRunState.CLAIMING
            code = run.gateway_code
            verifier = run.pkce_verifier
            run.gateway_code = None  # Single-use: zero out immediately

            if not code or not verifier:
                run.state = AuthRunState.FAILED
                run.error_message = "Missing authorization code or verifier"
                return self._apply_security_headers(
                    JSONResponse({"status": "failed", "error": run.error_message}, status_code=200)
                )

            # Atomically exchange code at gateway /token
            token_data, error_msg = await self._exchange_gateway_token(code=code, verifier=verifier)
            if error_msg or not token_data:
                run.state = AuthRunState.FAILED
                run.error_message = error_msg or "Token exchange failed"
                return self._apply_security_headers(
                    JSONResponse({"status": "failed", "error": run.error_message}, status_code=200)
                )

            # Successfully exchanged! Mark consumed.
            # Discard refresh token; never store raw access token on server.
            run.state = AuthRunState.CONSUMED
            access_token = token_data.get("access_token")
            expires_in = token_data.get("expires_in", 7200)
            token_type = token_data.get("token_type", "Bearer")

            return self._apply_security_headers(
                JSONResponse(
                    {
                        "status": "success",
                        "token": access_token,
                        "token_type": token_type,
                        "expires_in": expires_in,
                    },
                    status_code=200,
                )
            )

        if run.state == AuthRunState.CLAIMING:
            return self._apply_security_headers(
                JSONResponse({"status": "claiming", "authRunId": run.run_id}, status_code=200)
            )

        return self._apply_security_headers(
            JSONResponse({"status": run.state.value, "authRunId": run.run_id}, status_code=200)
        )

    async def _exchange_gateway_token(
        self,
        code: str,
        verifier: str,
    ) -> tuple[dict[str, Any] | None, str | None]:
        client = await self._get_client()
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.config.oauth_redirect_uri,
            "client_id": self.config.oauth_client_id,
            "code_verifier": verifier,
        }
        try:
            resp = await client.post(self.config.gateway_token_url, data=data)
            if resp.status_code != 200:
                return None, f"Gateway error {resp.status_code}: {resp.text}"
            token_resp = resp.json()
            if not isinstance(token_resp, dict) or "access_token" not in token_resp:
                return None, "Malformed gateway token response"
            return token_resp, None
        except Exception as exc:
            return None, f"Gateway token exchange network error: {str(exc)}"

    def _render_page(self, status_code: int, title: str, content: str) -> HTMLResponse:
        doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; padding: 1rem; }}
.card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; max-width: 460px; width: 100%; padding: 2rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2); box-sizing: border-box; }}
h1 {{ font-size: 1.25rem; margin-top: 0; margin-bottom: 1rem; color: #f1f5f9; }}
p {{ font-size: 0.925rem; color: #94a3b8; line-height: 1.5; margin-top: 0; }}
.info-box {{ background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 1rem; margin-bottom: 1.25rem; font-size: 0.875rem; color: #cbd5e1; }}
.info-box div {{ margin-bottom: 0.5rem; }}
.info-box div:last-child {{ margin-bottom: 0; }}
.pairing-code {{ font-family: monospace; font-size: 1.1rem; color: #38bdf8; font-weight: bold; letter-spacing: 0.1em; }}
.notice {{ font-size: 0.825rem; color: #64748b; margin-bottom: 1.5rem; }}
.button-row {{ display: flex; gap: 1rem; justify-content: flex-end; }}
.button-row form {{ margin: 0; }}
.btn {{ padding: 0.625rem 1.25rem; border-radius: 6px; border: none; font-size: 0.925rem; font-weight: 500; cursor: pointer; }}
.btn-approve {{ background: #3b82f6; color: #fff; }}
.btn-approve:hover {{ background: #2563eb; }}
.btn-deny {{ background: #334155; color: #cbd5e1; }}
.btn-deny:hover {{ background: #475569; }}
</style>
</head>
<body>
{content}
</body>
</html>"""
        response = HTMLResponse(doc, status_code=status_code)
        return self._apply_security_headers(response)
