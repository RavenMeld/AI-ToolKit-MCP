#!/usr/bin/env python3
import argparse
import asyncio
import hmac
import json
import logging
import math
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from aiohttp import web

from mcp_server import MCP_TOOLS, handle_call_tool, handle_list_tools

logger = logging.getLogger(__name__)
AUTH_TOKEN_KEY = web.AppKey("auth_token", Optional[str])
AUTH_FOR_HEALTH_KEY = web.AppKey("auth_for_health", bool)
AUTH_FOR_TOOLS_KEY = web.AppKey("auth_for_tools", bool)
TOOL_TIMEOUT_SECONDS_KEY = web.AppKey("tool_timeout_seconds", float)
MAX_BODY_BYTES_KEY = web.AppKey("max_body_bytes", int)
RATE_LIMIT_ENABLED_KEY = web.AppKey("rate_limit_enabled", bool)
RATE_LIMIT_WINDOW_SECONDS_KEY = web.AppKey("rate_limit_window_seconds", float)
RATE_LIMIT_MAX_REQUESTS_KEY = web.AppKey("rate_limit_max_requests", int)
RATE_LIMIT_TRUST_PROXY_KEY = web.AppKey("rate_limit_trust_proxy", bool)
RATE_LIMIT_CLIENT_IP_HEADER_KEY = web.AppKey("rate_limit_client_ip_header", str)
RATE_LIMIT_STATE_KEY = web.AppKey("rate_limit_state", Dict[str, Dict[str, Any]])
REQUEST_ID_KEY = web.AppKey("request_id", str)


def _error_response(
    status: int,
    code: str,
    message: str,
    request_id: str,
    details: Dict[str, Any] | None = None,
    headers: Dict[str, str] | None = None,
) -> web.Response:
    payload: Dict[str, Any] = {
        "ok": False,
        "request_id": request_id,
        "error": {
            "code": code,
            "message": message,
        },
    }
    if details is not None:
        payload["error"]["details"] = details
    return web.json_response(payload, status=status, headers=headers)


def _serialize_mcp_content(content: Any) -> Dict[str, Any]:
    if hasattr(content, "model_dump"):
        return content.model_dump(mode="json")
    if isinstance(content, dict):
        return content

    serialized: Dict[str, Any] = {}
    for field in ("type", "text", "mimeType", "data", "uri", "name"):
        if hasattr(content, field):
            value = getattr(content, field)
            if value is not None:
                serialized[field] = value

    return serialized or {"repr": repr(content)}


def _serialize_tool(tool: Any) -> Dict[str, Any]:
    if hasattr(tool, "model_dump"):
        return tool.model_dump(mode="json")

    return {
        "name": getattr(tool, "name", ""),
        "description": getattr(tool, "description", ""),
        "inputSchema": getattr(tool, "inputSchema", {}),
    }


def _request_id_from_request(request: web.Request) -> str:
    existing = request.get(REQUEST_ID_KEY)
    if isinstance(existing, str) and existing:
        return existing
    candidate = request.headers.get("X-Request-ID")
    if candidate and candidate.strip():
        cleaned = candidate.strip()
        if "\n" not in cleaned and "\r" not in cleaned:
            return cleaned[:128]
    return str(uuid.uuid4())


def _extract_auth_token(request: web.Request) -> Optional[str]:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if token:
            return token

    api_key_header = request.headers.get("X-API-Key", "").strip()
    if api_key_header:
        return api_key_header

    return None


def _require_auth(request: web.Request) -> tuple[bool, Optional[str]]:
    required_token = request.app.get(AUTH_TOKEN_KEY)
    if not required_token:
        return True, None

    provided = _extract_auth_token(request)
    if not provided:
        return False, "AUTH_REQUIRED"

    if not hmac.compare_digest(str(provided), str(required_token)):
        return False, "AUTH_INVALID"

    return True, None


def _auth_error_response(request_id: str, auth_error: Optional[str]) -> web.Response:
    if auth_error == "AUTH_REQUIRED":
        return _error_response(
            401,
            "AUTH_REQUIRED",
            "Authentication required. Provide Authorization: Bearer <token> or X-API-Key.",
            request_id=request_id,
        )
    return _error_response(
        403,
        "AUTH_INVALID",
        "Authentication token is invalid.",
        request_id=request_id,
    )


def _parse_env_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_json_text(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _normalize_tool_data(serialized_items: List[Dict[str, Any]]) -> Any:
    parsed_items: List[Dict[str, Any]] = []
    for item in serialized_items:
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            maybe_json = _parse_json_text(item["text"])
            if maybe_json is not None:
                parsed_items.append({"type": "json", "value": maybe_json})
            else:
                parsed_items.append({"type": "text", "value": item["text"]})
        else:
            parsed_items.append({"type": item.get("type", "unknown"), "value": item})

    if len(parsed_items) == 1:
        single = parsed_items[0]
        if single["type"] == "json":
            return single["value"]
        if single["type"] == "text":
            return {"message": single["value"]}
        return {"item": single["value"]}

    return {"items": parsed_items}


def _forwarded_client_identity(request: web.Request) -> Optional[str]:
    if not bool(request.app.get(RATE_LIMIT_TRUST_PROXY_KEY, False)):
        return None

    header_name = str(request.app.get(RATE_LIMIT_CLIENT_IP_HEADER_KEY, "X-Forwarded-For") or "X-Forwarded-For")
    raw_value = request.headers.get(header_name, "")
    if not raw_value:
        return None

    # Common proxy convention: first IP in comma-separated chain is original client.
    candidate = str(raw_value).split(",")[0].strip()
    if not candidate or "\n" in candidate or "\r" in candidate:
        return None

    return candidate[:128]


def _client_identity(request: web.Request) -> str:
    forwarded = _forwarded_client_identity(request)
    if forwarded:
        return forwarded
    return request.remote or "unknown"


def _prune_rate_limit_state(
    state: Dict[str, Dict[str, Any]],
    now_ts: float,
    window_seconds: float,
) -> None:
    if len(state) < 1024:
        return
    stale_cutoff = now_ts - (window_seconds * 2.0)
    stale_keys = [
        key
        for key, entry in state.items()
        if float(entry.get("window_start", 0.0)) < stale_cutoff
    ]
    for key in stale_keys:
        state.pop(key, None)


def _check_rate_limit(request: web.Request) -> Dict[str, Any] | None:
    if not bool(request.app.get(RATE_LIMIT_ENABLED_KEY, False)):
        return None

    window_seconds = max(1.0, float(request.app.get(RATE_LIMIT_WINDOW_SECONDS_KEY, 60.0) or 60.0))
    max_requests = max(1, int(request.app.get(RATE_LIMIT_MAX_REQUESTS_KEY, 120) or 120))
    now_ts = time.monotonic()
    state = request.app.get(RATE_LIMIT_STATE_KEY, {})
    _prune_rate_limit_state(state, now_ts, window_seconds)

    client_key = _client_identity(request)
    entry = state.get(client_key)

    if entry is None:
        state[client_key] = {"window_start": now_ts, "count": 1}
        return None

    window_start = float(entry.get("window_start", now_ts))
    elapsed = now_ts - window_start
    if elapsed >= window_seconds:
        entry["window_start"] = now_ts
        entry["count"] = 1
        return None

    count = int(entry.get("count", 0))
    if count >= max_requests:
        retry_after_seconds = max(1, int(math.ceil(window_seconds - elapsed)))
        return {
            "retry_after_seconds": retry_after_seconds,
            "window_seconds": window_seconds,
            "max_requests": max_requests,
        }

    entry["count"] = count + 1
    return None


@web.middleware
async def _error_middleware(
    request: web.Request,
    handler: Any,
) -> web.StreamResponse:
    request_id = _request_id_from_request(request)
    request[REQUEST_ID_KEY] = request_id
    try:
        response = await handler(request)
    except web.HTTPRequestEntityTooLarge:
        max_body_bytes = int(request.app.get(MAX_BODY_BYTES_KEY, 0))
        response = _error_response(
            413,
            "PAYLOAD_TOO_LARGE",
            "Request body exceeds configured size limit.",
            request_id=request_id,
            details={"max_body_bytes": max_body_bytes},
        )
    except web.HTTPException as exc:
        response = _error_response(
            int(getattr(exc, "status", 500)),
            f"HTTP_{int(getattr(exc, 'status', 500))}",
            getattr(exc, "reason", None) or "HTTP request failed.",
            request_id=request_id,
        )

    response.headers["X-Request-ID"] = request_id
    return response


async def health_handler(_request: web.Request) -> web.Response:
    request_id = _request_id_from_request(_request)
    if bool(_request.app.get(AUTH_FOR_HEALTH_KEY, False)):
        authorized, auth_error = _require_auth(_request)
        if not authorized:
            return _auth_error_response(request_id, auth_error)

    auth_enabled = bool(_request.app.get(AUTH_TOKEN_KEY))
    return web.json_response(
        {
            "ok": True,
            "request_id": request_id,
            "status": "healthy",
            "service": "ai-toolkit-mcp-http",
            "auth_enabled": auth_enabled,
        }
    )


async def tools_handler(_request: web.Request) -> web.Response:
    request_id = _request_id_from_request(_request)
    if bool(_request.app.get(AUTH_FOR_TOOLS_KEY, False)):
        authorized, auth_error = _require_auth(_request)
        if not authorized:
            return _auth_error_response(request_id, auth_error)

    tools = await handle_list_tools()
    serialized = [_serialize_tool(tool) for tool in tools]
    return web.json_response(
        {
            "ok": True,
            "request_id": request_id,
            "count": len(serialized),
            "tools": serialized,
        }
    )


async def mcp_tool_handler(request: web.Request) -> web.Response:
    request_id = _request_id_from_request(request)

    authorized, auth_error = _require_auth(request)
    if not authorized:
        return _auth_error_response(request_id, auth_error)

    rate_limit_result = _check_rate_limit(request)
    if rate_limit_result is not None:
        retry_after_seconds = int(rate_limit_result["retry_after_seconds"])
        return _error_response(
            429,
            "RATE_LIMITED",
            "Too many requests for this client.",
            request_id=request_id,
            details=rate_limit_result,
            headers={"Retry-After": str(retry_after_seconds)},
        )

    try:
        raw_body = await request.text()
        body = json.loads(raw_body)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return _error_response(
            400,
            "INVALID_JSON",
            "Request body must be valid JSON.",
            request_id=request_id,
        )

    if not isinstance(body, dict):
        return _error_response(
            400,
            "INVALID_REQUEST",
            "Request body must be a JSON object.",
            request_id=request_id,
        )

    name = body.get("name")
    arguments = body.get("arguments", {})

    if not isinstance(name, str) or not name.strip():
        return _error_response(
            400,
            "INVALID_REQUEST",
            "`name` is required and must be a non-empty string.",
            request_id=request_id,
        )
    name = name.strip()

    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return _error_response(
            400,
            "INVALID_REQUEST",
            "`arguments` must be a JSON object when provided.",
            request_id=request_id,
        )

    if name not in MCP_TOOLS:
        return _error_response(
            404,
            "TOOL_NOT_FOUND",
            f"Unknown tool: {name}",
            request_id=request_id,
            details={"available_tools": MCP_TOOLS},
        )

    timeout_seconds = float(request.app.get(TOOL_TIMEOUT_SECONDS_KEY, 120.0) or 120.0)
    try:
        result = await asyncio.wait_for(handle_call_tool(name, arguments), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        return _error_response(
            504,
            "TOOL_TIMEOUT",
            f"Tool execution exceeded timeout of {timeout_seconds:.2f}s.",
            request_id=request_id,
            details={"tool_timeout_seconds": timeout_seconds},
        )
    except ValueError as exc:
        return _error_response(
            404,
            "TOOL_NOT_FOUND",
            str(exc),
            request_id=request_id,
            details={"available_tools": MCP_TOOLS},
        )
    except Exception as exc:
        logger.exception("Tool execution failed for %s", name)
        return _error_response(
            500,
            "TOOL_EXECUTION_FAILED",
            f"Tool execution failed for {name}: {exc}",
            request_id=request_id,
        )

    serialized_result = [_serialize_mcp_content(item) for item in result]
    normalized_data = _normalize_tool_data(serialized_result)
    return web.json_response(
        {
            "ok": True,
            "request_id": request_id,
            "tool": name,
            "data": normalized_data,
            "result": serialized_result,
        }
    )


def create_app(
    auth_token: Optional[str] = None,
    auth_for_health: bool = False,
    auth_for_tools: bool = False,
    tool_timeout_seconds: float = 120.0,
    max_body_bytes: int = 1024 * 1024,
    rate_limit_enabled: bool = False,
    rate_limit_window_seconds: float = 60.0,
    rate_limit_max_requests: int = 120,
    trust_proxy_for_rate_limit: bool = False,
    rate_limit_client_ip_header: str = "X-Forwarded-For",
) -> web.Application:
    safe_timeout = max(0.01, float(tool_timeout_seconds))
    safe_max_body = max(1, int(max_body_bytes))
    safe_rate_limit_window_seconds = max(1.0, float(rate_limit_window_seconds))
    safe_rate_limit_max_requests = max(1, int(rate_limit_max_requests))
    safe_rate_limit_header = str(rate_limit_client_ip_header or "X-Forwarded-For").strip() or "X-Forwarded-For"
    app = web.Application(client_max_size=safe_max_body, middlewares=[_error_middleware])
    app[AUTH_TOKEN_KEY] = auth_token.strip() if isinstance(auth_token, str) and auth_token.strip() else None
    app[AUTH_FOR_HEALTH_KEY] = bool(auth_for_health)
    app[AUTH_FOR_TOOLS_KEY] = bool(auth_for_tools)
    app[TOOL_TIMEOUT_SECONDS_KEY] = safe_timeout
    app[MAX_BODY_BYTES_KEY] = safe_max_body
    app[RATE_LIMIT_ENABLED_KEY] = bool(rate_limit_enabled)
    app[RATE_LIMIT_WINDOW_SECONDS_KEY] = safe_rate_limit_window_seconds
    app[RATE_LIMIT_MAX_REQUESTS_KEY] = safe_rate_limit_max_requests
    app[RATE_LIMIT_TRUST_PROXY_KEY] = bool(trust_proxy_for_rate_limit)
    app[RATE_LIMIT_CLIENT_IP_HEADER_KEY] = safe_rate_limit_header
    app[RATE_LIMIT_STATE_KEY] = {}
    app.router.add_get("/health", health_handler)
    app.router.add_get("/tools", tools_handler)
    app.router.add_post("/mcp/tool", mcp_tool_handler)
    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTTP wrapper for AI-ToolKit MCP tools")
    parser.add_argument("--host", default=os.getenv("MCP_HTTP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_HTTP_PORT", "8080")))
    parser.add_argument(
        "--auth-token",
        default=os.getenv("MCP_HTTP_AUTH_TOKEN"),
        help="Optional auth token accepted via Authorization: Bearer <token> or X-API-Key.",
    )
    auth_for_health_default = _parse_env_bool(os.getenv("MCP_HTTP_AUTH_FOR_HEALTH"), default=False)
    auth_for_health_group = parser.add_mutually_exclusive_group()
    auth_for_health_group.add_argument(
        "--auth-for-health",
        dest="auth_for_health",
        action="store_true",
        help="Require the auth token on GET /health as well.",
    )
    auth_for_health_group.add_argument(
        "--no-auth-for-health",
        dest="auth_for_health",
        action="store_false",
        help="Disable auth requirement on GET /health.",
    )
    parser.set_defaults(auth_for_health=auth_for_health_default)
    auth_for_tools_default = _parse_env_bool(os.getenv("MCP_HTTP_AUTH_FOR_TOOLS"), default=False)
    auth_for_tools_group = parser.add_mutually_exclusive_group()
    auth_for_tools_group.add_argument(
        "--auth-for-tools",
        dest="auth_for_tools",
        action="store_true",
        help="Require the auth token on GET /tools as well.",
    )
    auth_for_tools_group.add_argument(
        "--no-auth-for-tools",
        dest="auth_for_tools",
        action="store_false",
        help="Disable auth requirement on GET /tools.",
    )
    parser.set_defaults(auth_for_tools=auth_for_tools_default)
    rate_limit_enabled_default = _parse_env_bool(os.getenv("MCP_HTTP_RATE_LIMIT_ENABLED"), default=False)
    rate_limit_enabled_group = parser.add_mutually_exclusive_group()
    rate_limit_enabled_group.add_argument(
        "--rate-limit-enabled",
        dest="rate_limit_enabled",
        action="store_true",
        help="Enable per-client rate limiting for POST /mcp/tool.",
    )
    rate_limit_enabled_group.add_argument(
        "--no-rate-limit",
        dest="rate_limit_enabled",
        action="store_false",
        help="Disable per-client rate limiting for POST /mcp/tool.",
    )
    parser.set_defaults(rate_limit_enabled=rate_limit_enabled_default)
    trust_proxy_for_rate_limit_default = _parse_env_bool(
        os.getenv("MCP_HTTP_TRUST_PROXY_FOR_RATE_LIMIT"),
        default=False,
    )
    trust_proxy_group = parser.add_mutually_exclusive_group()
    trust_proxy_group.add_argument(
        "--trust-proxy-for-rate-limit",
        dest="trust_proxy_for_rate_limit",
        action="store_true",
        help="Trust forwarded client-IP header when deriving rate-limit client identity.",
    )
    trust_proxy_group.add_argument(
        "--no-trust-proxy-for-rate-limit",
        dest="trust_proxy_for_rate_limit",
        action="store_false",
        help="Ignore forwarded client-IP headers for rate limiting.",
    )
    parser.set_defaults(trust_proxy_for_rate_limit=trust_proxy_for_rate_limit_default)
    parser.add_argument(
        "--rate-limit-client-ip-header",
        default=os.getenv("MCP_HTTP_RATE_LIMIT_CLIENT_IP_HEADER", "X-Forwarded-For"),
        help="Header used for client identity when proxy trust is enabled.",
    )
    parser.add_argument(
        "--rate-limit-window-seconds",
        type=float,
        default=float(os.getenv("MCP_HTTP_RATE_LIMIT_WINDOW_SECONDS", "60")),
        help="Rate-limit window size for POST /mcp/tool.",
    )
    parser.add_argument(
        "--rate-limit-max-requests",
        type=int,
        default=int(os.getenv("MCP_HTTP_RATE_LIMIT_MAX_REQUESTS", "120")),
        help="Maximum requests per client within the rate-limit window for POST /mcp/tool.",
    )
    parser.add_argument(
        "--tool-timeout-seconds",
        type=float,
        default=float(os.getenv("MCP_HTTP_TOOL_TIMEOUT_SECONDS", "120")),
        help="Timeout for each tool execution request.",
    )
    parser.add_argument(
        "--max-body-bytes",
        type=int,
        default=int(os.getenv("MCP_HTTP_MAX_BODY_BYTES", str(1024 * 1024))),
        help="Maximum accepted HTTP request body size in bytes.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = _parse_args()
    auth_enabled = bool(args.auth_token)
    logger.info("Starting HTTP wrapper on %s:%s", args.host, args.port)
    logger.info("HTTP auth is %s", "enabled" if auth_enabled else "disabled")
    logger.info("GET /health auth is %s", "enabled" if bool(args.auth_for_health) else "disabled")
    logger.info("GET /tools auth is %s", "enabled" if bool(args.auth_for_tools) else "disabled")
    logger.info(
        "POST /mcp/tool rate limit is %s (window=%ss, max_requests=%s, trust_proxy=%s, header=%s)",
        "enabled" if bool(args.rate_limit_enabled) else "disabled",
        max(1.0, float(args.rate_limit_window_seconds)),
        max(1, int(args.rate_limit_max_requests)),
        "enabled" if bool(args.trust_proxy_for_rate_limit) else "disabled",
        str(args.rate_limit_client_ip_header or "X-Forwarded-For"),
    )
    logger.info("Tool timeout: %.2fs", max(0.01, float(args.tool_timeout_seconds)))
    logger.info("Max request body bytes: %d", max(1, int(args.max_body_bytes)))
    web.run_app(
        create_app(
            auth_token=args.auth_token,
            auth_for_health=args.auth_for_health,
            auth_for_tools=args.auth_for_tools,
            tool_timeout_seconds=args.tool_timeout_seconds,
            max_body_bytes=args.max_body_bytes,
            rate_limit_enabled=args.rate_limit_enabled,
            rate_limit_window_seconds=args.rate_limit_window_seconds,
            rate_limit_max_requests=args.rate_limit_max_requests,
            trust_proxy_for_rate_limit=args.trust_proxy_for_rate_limit,
            rate_limit_client_ip_header=args.rate_limit_client_ip_header,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
