"""Unit tests for Colab MCP stdio proxy.
"""

import json
import unittest
from unittest.mock import patch, MagicMock

from colab_mcp.mcp_proxy import forward_request


class TestMCPProxy(unittest.TestCase):

    def test_forward_tools_call_without_token(self):
        res = forward_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "get_subscription"}
        }, token="")
        self.assertIn("error", res)
        self.assertEqual(res["error"]["code"], -32603)

    @patch("urllib.request.urlopen")
    def test_forward_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        res = forward_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, token="valid-tok")
        self.assertIn("result", res)


if __name__ == "__main__":
    unittest.main()
