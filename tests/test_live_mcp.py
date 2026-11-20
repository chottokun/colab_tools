"""Live contract tests against Google Colab MCP Endpoint.

Validates protocol, tools discovery, and client argument schema compliance
against https://colaboratory.googleapis.com/mcp.
"""

import json
import unittest
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

from colab_mcp.mcp_client import ColabMCPClient


class TestColabLiveMCP(unittest.TestCase):
    ENDPOINT = "https://colaboratory.googleapis.com/mcp"

    def _fetch_live_tools(self):
        payload = {"jsonrpc": "2.0", "id": 100, "method": "tools/list"}
        req = urllib.request.Request(
            self.ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
        return {t["name"]: t for t in data["result"]["tools"]}

    def test_live_mcp_tools_discovery(self):
        """Verifies that the live Colab MCP endpoint exposes all official tools."""
        tools = self._fetch_live_tools()
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
            "list_sessions",
        }
        self.assertTrue(expected_tools.issubset(set(tools.keys())))

    def test_client_arguments_match_live_input_schemas(self):
        """Contract test: verifies ColabMCPClient argument keys match live MCP schemas."""
        live_tools = self._fetch_live_tools()
        client = ColabMCPClient(access_token="contract-test-token")

        captured = {}

        def fake_call_tool(tool_name, arguments=None):
            captured[tool_name] = arguments or {}
            return {"jsonrpc": "2.0", "id": "1", "result": {"structuredContent": {}}}

        with patch.object(client, "call_tool", side_effect=fake_call_tool):
            client.create_runtime(
                spec={"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_STANDARD"},
                runtime_id="rt-1",
                request_id="00000000-0000-4000-8000-000000000001",
            )
            client.delete_runtime("runtimes/rt-1", allow_missing=True)
            client.create_session("runtimes/rt-1", session_id="sess-1")
            client.list_sessions("runtimes/rt-1")
            client.delete_session("runtimes/rt-1/sessions/sess-1")
            client.execute_code(
                code="print(1)",
                session="runtimes/rt-1/sessions/sess-1",
                execution_id="00000000-0000-4000-8000-000000000002",
            )
            client.get_runtime("runtimes/rt-1")

        for tool_name, sent_args in captured.items():
            schema = live_tools[tool_name]["inputSchema"]
            allowed_props = set(schema.get("properties", {}).keys())
            required_props = set(schema.get("required", []))

            self.assertTrue(
                required_props.issubset(set(sent_args.keys())),
                f"{tool_name}: missing required keys {required_props - set(sent_args.keys())}",
            )
            self.assertTrue(
                set(sent_args.keys()).issubset(allowed_props),
                f"{tool_name}: unknown keys {set(sent_args.keys()) - allowed_props}",
            )

        # Also validate nested `runtime.runtimeSpec` for create_runtime
        rt_props = set(
            live_tools["create_runtime"]["inputSchema"]["$defs"]["Runtime"]["properties"].keys()
        )
        self.assertTrue(set(captured["create_runtime"]["runtime"].keys()).issubset(rt_props))

    def test_live_auth_challenge(self):
        """Verifies that unauthorized calls return expected 401 with standard error payload."""
        payload = {
            "jsonrpc": "2.0",
            "id": 102,
            "method": "tools/call",
            "params": {"name": "get_subscription", "arguments": {}},
        }
        req = urllib.request.Request(
            self.ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10):
                self.fail("Expected 401 Unauthorized, but request succeeded.")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 401)
            body = json.loads(e.read().decode("utf-8"))
            is_err = (
                body.get("isError")
                or body.get("result", {}).get("isError")
                or ("error" in body)
            )
            self.assertTrue(is_err)


if __name__ == "__main__":
    unittest.main()
