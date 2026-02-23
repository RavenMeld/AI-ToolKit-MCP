#!/usr/bin/env python3
import argparse
import asyncio
import hmac
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from aiohttp import web

from mcp_server import MCP_TOOLS, handle_call_tool, handle_list_tools

logger = logging.getLogger(__name__)
AUTH_TOKEN_KEY = web.AppKey("auth_token", Optional[str])
TOOL_TIMEOUT_SECONDS_KEY = web.AppKey("tool_timeout_seconds", float)
MAX_BODY_BYTES_KEY = web.AppKey("max_body_bytes", int)
REQUEST_ID_KEY = web.AppKey("request_id", str)


def _error_response(
    status: int,
    code: str,
    message: str,
    request_id: str,
    details: Dict[str, Any] | None = None,
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
    return web.json_response(payload, status=status)


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
        return candidate.strip()
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


@web.middleware
async def _error_middleware(
    request: web.Request,
    handler: Any,
) -> web.StreamResponse:
    request_id = _request_id_from_request(request)
    request[REQUEST_ID_KEY] = request_id
    try:
        return await handler(request)
    except web.HTTPRequestEntityTooLarge:
        max_body_bytes = int(request.app.get(MAX_BODY_BYTES_KEY, 0))
        return _error_response(
            413,
            "PAYLOAD_TOO_LARGE",
            "Request body exceeds configured size limit.",
            request_id=request_id,
            details={"max_body_bytes": max_body_bytes},
        )


async def health_handler(_request: web.Request) -> web.Response:
    request_id = _request_id_from_request(_request)
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

    try:
        body = await request.json()
    except json.JSONDecodeError:
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
    tool_timeout_seconds: float = 120.0,
    max_body_bytes: int = 1024 * 1024,
) -> web.Application:
    safe_timeout = max(0.01, float(tool_timeout_seconds))
    safe_max_body = max(1, int(max_body_bytes))
    app = web.Application(client_max_size=safe_max_body, middlewares=[_error_middleware])
    app[AUTH_TOKEN_KEY] = auth_token.strip() if isinstance(auth_token, str) and auth_token.strip() else None
    app[TOOL_TIMEOUT_SECONDS_KEY] = safe_timeout
    app[MAX_BODY_BYTES_KEY] = safe_max_body
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
    logger.info("Tool timeout: %.2fs", max(0.01, float(args.tool_timeout_seconds)))
    logger.info("Max request body bytes: %d", max(1, int(args.max_body_bytes)))
    web.run_app(
        create_app(
            auth_token=args.auth_token,
            tool_timeout_seconds=args.tool_timeout_seconds,
            max_body_bytes=args.max_body_bytes,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
