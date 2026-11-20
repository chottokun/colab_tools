"""Unit tests for Colab MCP stdio proxy."""

import io
import json
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from colab_mcp.config import ColabConfig
from colab_mcp.mcp_proxy import forward_request, run_proxy


class TestMCPProxy(unittest.TestCase):

    def setUp(self):
        ColabConfig.clear_token_cache()

    def tearDown(self):
        ColabConfig.clear_token_cache()

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

    @patch("urllib.request.urlopen")
    def test_forward_sets_quota_project_header(self, mock_urlopen):
        """quota_project 指定時に x-goog-user-project ヘッダが転送されることを検証する。"""
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.read.return_value = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "result": {}}
        ).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        res = forward_request(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "get_subscription"}},
            token="valid-tok",
            quota_project="my-proj",
        )
        self.assertIsNotNone(res)
        sent_req = mock_urlopen.call_args[0][0]
        headers_lower = {k.lower(): v for k, v in sent_req.header_items()}
        self.assertEqual(headers_lower.get("x-goog-user-project"), "my-proj")

    @patch("urllib.request.urlopen")
    def test_public_methods_skip_token_resolution(self, mock_urlopen):
        """tools/list や initialize ではトークン解決が呼ばれないことを検証する。"""
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.read.return_value = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}
        ).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        with patch.object(ColabConfig, "get_token") as mock_get_tok:
            for method in ("initialize", "tools/list"):
                res = forward_request(
                    {"jsonrpc": "2.0", "id": 1, "method": method},
                    session_state={},
                )
                self.assertIn("result", res)
            mock_get_tok.assert_not_called()

    @patch("urllib.request.urlopen")
    def test_tools_call_401_refresh_retry_and_session_id(self, mock_urlopen):
        """tools/call の 401 自動更新リトライと Mcp-Session-Id 引き継ぎを検証する。"""
        err_401 = urllib.error.HTTPError(
            "https://colaboratory.googleapis.com/mcp",
            401,
            "Unauthorized",
            {},
            io.BytesIO(b"expired"),
        )
        ok_resp = MagicMock()
        ok_resp.headers = {
            "Content-Type": "application/json",
            "Mcp-Session-Id": "sess-xyz-99",
        }
        ok_resp.read.return_value = json.dumps(
            {"jsonrpc": "2.0", "id": 2, "result": {"structuredContent": {"ok": True}}}
        ).encode("utf-8")
        ok_ctx = MagicMock()
        ok_ctx.__enter__.return_value = ok_resp

        ok_resp2 = MagicMock()
        ok_resp2.headers = {"Content-Type": "application/json"}
        ok_resp2.read.return_value = json.dumps(
            {"jsonrpc": "2.0", "id": 3, "result": {"structuredContent": {"ok": True}}}
        ).encode("utf-8")
        ok_ctx2 = MagicMock()
        ok_ctx2.__enter__.return_value = ok_resp2

        mock_urlopen.side_effect = [err_401, ok_ctx, ok_ctx2]
        session_state = {}

        with (
            patch.object(ColabConfig, "get_quota_project", return_value="proj-1"),
            patch.object(
                ColabConfig, "get_token", side_effect=["stale-tok", "fresh-tok", "fresh-tok"]
            ),
        ):
            r1 = forward_request(
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "get_subscription"}},
                session_state=session_state,
            )
            self.assertEqual(r1["result"]["structuredContent"]["ok"], True)
            self.assertEqual(session_state.get("mcp_session_id"), "sess-xyz-99")

            forward_request(
                {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "list_runtimes"}},
                session_state=session_state,
            )
            last_req = mock_urlopen.call_args_list[-1][0][0]
            headers_lower = {k.lower(): v for k, v in last_req.header_items()}
            self.assertEqual(headers_lower.get("mcp-session-id"), "sess-xyz-99")

    def test_run_proxy_invalid_json_returns_parse_error(self):
        """不正 JSON 入力時に JSON-RPC Parse Error (-32700) が返されることを検証する。"""
        fake_in = io.StringIO("{invalid json\n")
        fake_out = io.StringIO()
        with (
            patch("sys.stdin", fake_in),
            patch("sys.stdout", fake_out),
            patch("sys.stderr", io.StringIO()),
        ):
            run_proxy()
        parsed = json.loads(fake_out.getvalue().strip())
        self.assertEqual(parsed["error"]["code"], -32700)
        self.assertIsNone(parsed["id"])


if __name__ == "__main__":
    unittest.main()
