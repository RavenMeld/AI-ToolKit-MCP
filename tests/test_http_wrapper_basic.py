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
        self.assertEqual(payload["status"], "healthy")

    async def test_tool_call_success_path(self):
        mocked_result = [types.TextContent(type="text", text="ok")]
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
        self.assertEqual(payload["result"][0]["type"], "text")
        self.assertEqual(payload["result"][0]["text"], "ok")
        mocked_handle.assert_awaited_once_with("list-configs", {})

    async def test_tool_call_invalid_tool(self):
        response = await self.client.post(
            "/mcp/tool",
            json={"name": "does-not-exist", "arguments": {}},
        )
        self.assertEqual(response.status, 404)
        payload = await response.json()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "TOOL_NOT_FOUND")


if __name__ == "__main__":
    unittest.main()
