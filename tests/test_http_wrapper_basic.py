import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import mcp.types as types
from aiohttp.test_utils import AioHTTPTestCase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp_http_server import create_app  # noqa: E402


class HttpWrapperBasicTests(AioHTTPTestCase):
    async def get_application(self):
        return create_app()

    async def test_health_endpoint(self):
        response = await self.client.get("/health")
        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertTrue(payload["ok"])
        self.assertIn("request_id", payload)
        self.assertEqual(payload["status"], "healthy")

    async def test_tool_call_success_path(self):
        mocked_result = [types.TextContent(type="text", text='{"status":"ok","count":2}')]
        with patch(
            "mcp_http_server.handle_call_tool",
            new=AsyncMock(return_value=mocked_result),
        ) as mocked_handle:
            response = await self.client.post(
                "/mcp/tool",
                json={"name": "list-configs", "arguments": {}},
            )

        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["tool"], "list-configs")
        self.assertIn("request_id", payload)
        self.assertEqual(payload["data"]["status"], "ok")
        self.assertEqual(payload["data"]["count"], 2)
        self.assertEqual(payload["result"][0]["type"], "text")
        self.assertEqual(payload["result"][0]["text"], '{"status":"ok","count":2}')
        mocked_handle.assert_awaited_once_with("list-configs", {})

    async def test_tool_call_plain_text_is_normalized(self):
        mocked_result = [types.TextContent(type="text", text="plain output")]
        with patch(
            "mcp_http_server.handle_call_tool",
            new=AsyncMock(return_value=mocked_result),
        ):
            response = await self.client.post(
                "/mcp/tool",
                json={"name": "list-configs", "arguments": {}},
            )

        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertEqual(payload["data"]["message"], "plain output")

    async def test_tool_call_invalid_tool(self):
        response = await self.client.post(
            "/mcp/tool",
            json={"name": "does-not-exist", "arguments": {}},
        )
        self.assertEqual(response.status, 404)
        payload = await response.json()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "TOOL_NOT_FOUND")
        self.assertIn("request_id", payload)


class HttpWrapperAuthTests(AioHTTPTestCase):
    async def get_application(self):
        return create_app(auth_token="secret-token")

    async def test_auth_required_without_token(self):
        response = await self.client.post(
            "/mcp/tool",
            json={"name": "list-configs", "arguments": {}},
        )
        self.assertEqual(response.status, 401)
        payload = await response.json()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "AUTH_REQUIRED")

    async def test_auth_invalid_with_wrong_token(self):
        response = await self.client.post(
            "/mcp/tool",
            json={"name": "list-configs", "arguments": {}},
            headers={"Authorization": "Bearer wrong-token"},
        )
        self.assertEqual(response.status, 403)
        payload = await response.json()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "AUTH_INVALID")

    async def test_auth_success_with_api_key_header(self):
        mocked_result = [types.TextContent(type="text", text='{"ok":true}')]
        with patch(
            "mcp_http_server.handle_call_tool",
            new=AsyncMock(return_value=mocked_result),
        ):
            response = await self.client.post(
                "/mcp/tool",
                json={"name": "list-configs", "arguments": {}},
                headers={"X-API-Key": "secret-token"},
            )

        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["data"]["ok"], True)


class HttpWrapperHardeningTests(AioHTTPTestCase):
    async def get_application(self):
        return create_app(tool_timeout_seconds=0.01, max_body_bytes=128)

    async def test_tool_timeout(self):
        async def _slow_tool(*_args, **_kwargs):
            await asyncio.sleep(0.05)
            return [types.TextContent(type="text", text='{"ok":true}')]

        with patch("mcp_http_server.handle_call_tool", new=_slow_tool):
            response = await self.client.post(
                "/mcp/tool",
                json={"name": "list-configs", "arguments": {}},
            )

        self.assertEqual(response.status, 504)
        payload = await response.json()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "TOOL_TIMEOUT")
        self.assertIn("tool_timeout_seconds", payload["error"]["details"])

    async def test_payload_too_large(self):
        response = await self.client.post(
            "/mcp/tool",
            json={"name": "list-configs", "arguments": {"blob": "x" * 1024}},
        )

        self.assertEqual(response.status, 413)
        payload = await response.json()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "PAYLOAD_TOO_LARGE")
        self.assertEqual(payload["error"]["details"]["max_body_bytes"], 128)


if __name__ == "__main__":
    unittest.main()
