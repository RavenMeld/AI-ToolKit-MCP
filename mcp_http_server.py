#!/usr/bin/env python3
import argparse
import json
import logging
import os
from typing import Any, Dict, List

from aiohttp import web

from mcp_server import MCP_TOOLS, handle_call_tool, handle_list_tools

logger = logging.getLogger(__name__)


def _error_response(
    status: int,
    code: str,
    message: str,
    details: Dict[str, Any] | None = None,
) -> web.Response:
    payload: Dict[str, Any] = {
        "ok": False,
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


async def health_handler(_request: web.Request) -> web.Response:
    return web.json_response(
        {
            "ok": True,
            "status": "healthy",
            "service": "ai-toolkit-mcp-http",
        }
    )


async def tools_handler(_request: web.Request) -> web.Response:
    tools = await handle_list_tools()
    serialized = [_serialize_tool(tool) for tool in tools]
    return web.json_response({"ok": True, "count": len(serialized), "tools": serialized})


async def mcp_tool_handler(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _error_response(400, "INVALID_JSON", "Request body must be valid JSON.")

    if not isinstance(body, dict):
        return _error_response(400, "INVALID_REQUEST", "Request body must be a JSON object.")

    name = body.get("name")
    arguments = body.get("arguments", {})

    if not isinstance(name, str) or not name.strip():
        return _error_response(
            400,
            "INVALID_REQUEST",
            "`name` is required and must be a non-empty string.",
        )
    name = name.strip()

    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return _error_response(
            400,
            "INVALID_REQUEST",
            "`arguments` must be a JSON object when provided.",
        )

    if name not in MCP_TOOLS:
        return _error_response(
            404,
            "TOOL_NOT_FOUND",
            f"Unknown tool: {name}",
            details={"available_tools": MCP_TOOLS},
        )

    try:
        result = await handle_call_tool(name, arguments)
    except ValueError as exc:
        return _error_response(
            404,
            "TOOL_NOT_FOUND",
            str(exc),
            details={"available_tools": MCP_TOOLS},
        )
    except Exception as exc:
        logger.exception("Tool execution failed for %s", name)
        return _error_response(
            500,
            "TOOL_EXECUTION_FAILED",
            f"Tool execution failed for {name}: {exc}",
        )

    serialized_result = [_serialize_mcp_content(item) for item in result]
    return web.json_response({"ok": True, "tool": name, "result": serialized_result})


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_get("/tools", tools_handler)
    app.router.add_post("/mcp/tool", mcp_tool_handler)
    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTTP wrapper for AI-ToolKit MCP tools")
    parser.add_argument("--host", default=os.getenv("MCP_HTTP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_HTTP_PORT", "8080")))
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = _parse_args()
    logger.info("Starting HTTP wrapper on %s:%s", args.host, args.port)
    web.run_app(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
