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

    async def test_request_id_header_is_echoed(self):
        mocked_result = [types.TextContent(type="text", text='{"ok":true}')]
        with patch("mcp_http_server.handle_call_tool", new=AsyncMock(return_value=mocked_result)):
            response = await self.client.post(
                "/mcp/tool",
                json={"name": "list-configs", "arguments": {}},
                headers={"X-Request-ID": "req-123"},
            )

        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers.get("X-Request-ID"), "req-123")
        payload = await response.json()
        self.assertEqual(payload["request_id"], "req-123")

    async def test_invalid_json_returns_structured_error(self):
        response = await self.client.post(
            "/mcp/tool",
            data='{"name": "list-configs",',
            headers={"Content-Type": "application/json", "X-Request-ID": "req-bad-json"},
        )
        self.assertEqual(response.status, 400)
        self.assertEqual(response.headers.get("X-Request-ID"), "req-bad-json")
        payload = await response.json()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["request_id"], "req-bad-json")
        self.assertEqual(payload["error"]["code"], "INVALID_JSON")

    async def test_unknown_route_returns_json_error(self):
        response = await self.client.get("/does-not-exist", headers={"X-Request-ID": "req-404"})
        self.assertEqual(response.status, 404)
        self.assertEqual(response.headers.get("X-Request-ID"), "req-404")
        payload = await response.json()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["request_id"], "req-404")
        self.assertEqual(payload["error"]["code"], "HTTP_404")


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


class HttpWrapperHealthEndpointAuthTests(AioHTTPTestCase):
    async def get_application(self):
        return create_app(auth_token="secret-token")

    async def test_health_endpoint_remains_open_by_default(self):
        response = await self.client.get("/health")
        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["status"], "healthy")


class HttpWrapperHealthEndpointProtectedTests(AioHTTPTestCase):
    async def get_application(self):
        return create_app(auth_token="secret-token", auth_for_health=True)

    async def test_health_endpoint_requires_auth_when_enabled(self):
        response = await self.client.get("/health")
        self.assertEqual(response.status, 401)
        payload = await response.json()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "AUTH_REQUIRED")

    async def test_health_endpoint_accepts_api_key_when_enabled(self):
        response = await self.client.get("/health", headers={"X-API-Key": "secret-token"})
        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["status"], "healthy")


class HttpWrapperToolsEndpointAuthTests(AioHTTPTestCase):
    async def get_application(self):
        return create_app(auth_token="secret-token")

    async def test_tools_endpoint_remains_open_by_default(self):
        response = await self.client.get("/tools")
        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertTrue(payload["ok"])
        self.assertIn("tools", payload)


class HttpWrapperToolsEndpointProtectedTests(AioHTTPTestCase):
    async def get_application(self):
        return create_app(auth_token="secret-token", auth_for_tools=True)

    async def test_tools_endpoint_requires_auth_when_enabled(self):
        response = await self.client.get("/tools")
        self.assertEqual(response.status, 401)
        payload = await response.json()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "AUTH_REQUIRED")

    async def test_tools_endpoint_accepts_api_key_when_enabled(self):
        response = await self.client.get("/tools", headers={"X-API-Key": "secret-token"})
        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertTrue(payload["ok"])
        self.assertIn("tools", payload)


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


class HttpWrapperRateLimitTests(AioHTTPTestCase):
    async def get_application(self):
        return create_app(
            rate_limit_enabled=True,
            rate_limit_window_seconds=60,
            rate_limit_max_requests=2,
        )

    async def test_tool_rate_limit_after_threshold(self):
        mocked_result = [types.TextContent(type="text", text='{"ok":true}')]
        with patch(
            "mcp_http_server.handle_call_tool",
            new=AsyncMock(return_value=mocked_result),
        ) as mocked_handle:
            first = await self.client.post("/mcp/tool", json={"name": "list-configs", "arguments": {}})
            second = await self.client.post("/mcp/tool", json={"name": "list-configs", "arguments": {}})
            third = await self.client.post("/mcp/tool", json={"name": "list-configs", "arguments": {}})

        self.assertEqual(first.status, 200)
        self.assertEqual(second.status, 200)
        self.assertEqual(third.status, 429)
        self.assertEqual(mocked_handle.await_count, 2)
        payload = await third.json()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "RATE_LIMITED")
        self.assertEqual(payload["error"]["details"]["max_requests"], 2)
        self.assertGreaterEqual(payload["error"]["details"]["retry_after_seconds"], 1)
        self.assertEqual(third.headers.get("Retry-After"), str(payload["error"]["details"]["retry_after_seconds"]))

    async def test_health_not_rate_limited(self):
        mocked_result = [types.TextContent(type="text", text='{"ok":true}')]
        with patch("mcp_http_server.handle_call_tool", new=AsyncMock(return_value=mocked_result)):
            await self.client.post("/mcp/tool", json={"name": "list-configs", "arguments": {}})
            await self.client.post("/mcp/tool", json={"name": "list-configs", "arguments": {}})
            await self.client.post("/mcp/tool", json={"name": "list-configs", "arguments": {}})

        response = await self.client.get("/health")
        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["status"], "healthy")


class HttpWrapperRateLimitProxyTrustTests(AioHTTPTestCase):
    async def get_application(self):
        return create_app(
            rate_limit_enabled=True,
            rate_limit_window_seconds=60,
            rate_limit_max_requests=1,
            trust_proxy_for_rate_limit=False,
        )

    async def test_forwarded_header_ignored_when_proxy_not_trusted(self):
        mocked_result = [types.TextContent(type="text", text='{"ok":true}')]
        with patch("mcp_http_server.handle_call_tool", new=AsyncMock(return_value=mocked_result)) as mocked_handle:
            first = await self.client.post(
                "/mcp/tool",
                json={"name": "list-configs", "arguments": {}},
                headers={"X-Forwarded-For": "10.0.0.1"},
            )
            second = await self.client.post(
                "/mcp/tool",
                json={"name": "list-configs", "arguments": {}},
                headers={"X-Forwarded-For": "10.0.0.2"},
            )

        self.assertEqual(first.status, 200)
        self.assertEqual(second.status, 429)
        self.assertEqual(mocked_handle.await_count, 1)


class HttpWrapperRateLimitProxyTrustedTests(AioHTTPTestCase):
    async def get_application(self):
        return create_app(
            rate_limit_enabled=True,
            rate_limit_window_seconds=60,
            rate_limit_max_requests=1,
            trust_proxy_for_rate_limit=True,
        )

    async def test_forwarded_header_used_when_proxy_trusted(self):
        mocked_result = [types.TextContent(type="text", text='{"ok":true}')]
        with patch("mcp_http_server.handle_call_tool", new=AsyncMock(return_value=mocked_result)) as mocked_handle:
            first = await self.client.post(
                "/mcp/tool",
                json={"name": "list-configs", "arguments": {}},
                headers={"X-Forwarded-For": "10.0.0.1"},
            )
            second = await self.client.post(
                "/mcp/tool",
                json={"name": "list-configs", "arguments": {}},
                headers={"X-Forwarded-For": "10.0.0.2"},
            )
            third = await self.client.post(
                "/mcp/tool",
                json={"name": "list-configs", "arguments": {}},
                headers={"X-Forwarded-For": "10.0.0.1"},
            )

        self.assertEqual(first.status, 200)
        self.assertEqual(second.status, 200)
        self.assertEqual(third.status, 429)
        self.assertEqual(mocked_handle.await_count, 2)


if __name__ == "__main__":
    unittest.main()
