"""Live test against Google Colab MCP Endpoint.
Validates protocol, tools discovery, and schemas against https://colaboratory.googleapis.com/mcp.
"""

import json
import unittest
import urllib.request
import urllib.error


class TestColabLiveMCP(unittest.TestCase):
    ENDPOINT = "https://colaboratory.googleapis.com/mcp"

    def test_live_mcp_tools_discovery(self):
        """Verifies that the live Colab MCP endpoint exposes all official tools."""
        payload = {
            "jsonrpc": "2.0",
            "id": 100,
            "method": "tools/list"
        }
        req = urllib.request.Request(
            self.ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))

        tool_names = {t["name"] for t in data["result"]["tools"]}
        expected_tools = {
            "create_runtime",
            "create_session",
            "delete_session",
            "delete_runtime",
            "execute_code",
            "get_runtime",
            "get_subscription",
            "list_runtimes",
            "list_runtime_specs",
            "list_sessions"
        }
        self.assertTrue(expected_tools.issubset(tool_names))

    def test_live_auth_challenge(self):
        """Verifies that unauthorized calls return expected 401 with standard error payload."""
        payload = {
            "jsonrpc": "2.0",
            "id": 102,
            "method": "tools/call",
            "params": {"name": "get_subscription", "arguments": {}}
        }
        req = urllib.request.Request(
            self.ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                self.fail("Expected 401 Unauthorized, but request succeeded.")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 401)
            body = json.loads(e.read().decode("utf-8"))
            is_err = body.get("isError") or body.get("result", {}).get("isError") or ("error" in body)
            self.assertTrue(is_err)


if __name__ == "__main__":
    unittest.main()
