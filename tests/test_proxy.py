"""Unit tests for Colab MCP stdio proxy."""

import json
import unittest
from unittest.mock import MagicMock, patch

from colab_mcp.mcp_proxy import forward_request


class TestMCPProxy(unittest.TestCase):

    def test_forward_tools_call_without_token(self):
        res = forward_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "get_subscription"},
            },
            token="",
        )
        self.assertIsNotNone(res)
        self.assertIn("error", res)
        self.assertEqual(res["error"]["code"], -32603)

    @patch("urllib.request.urlopen")
    def test_forward_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.read.return_value = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "result": {}}
        ).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        res = forward_request(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, token="valid-tok"
        )
        self.assertIsNotNone(res)
        self.assertIn("result", res)

    @patch("urllib.request.urlopen")
    def test_notification_returns_none(self, mock_urlopen):
        """JSON-RPC notifications (no `id`) must not emit a response on stdout."""
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.read.return_value = b""
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        res = forward_request(
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            token="valid-tok",
        )
        self.assertIsNone(res)


if __name__ == "__main__":
    unittest.main()
