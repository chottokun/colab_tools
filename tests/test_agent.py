"""Unit tests for Colab Agent Harness, ContextOptimizer, and Tool Schemas.
"""

import json
import unittest
from unittest.mock import patch, MagicMock

from colab_mcp.agent.runner import ColabAgentHarness
from colab_mcp.agent.context import ContextOptimizer
from colab_mcp.agent.tools import to_openai_tools, to_anthropic_tools


class TestColabAgent(unittest.TestCase):

    def test_context_optimizer_truncation(self):
        huge_text = "A" * 10000
        raw = {"result": {"output": huge_text}, "execution_id": "test-id"}
        res = ContextOptimizer.format_execution_result(raw)
        self.assertTrue(res["output_truncated"])
        self.assertLess(len(res["output"]), 5000)
        self.assertIn("TRUNCATED", res["output"])

    def test_tool_schemas(self):
        openai_tools = to_openai_tools()
        self.assertEqual(len(openai_tools), 4)
        self.assertEqual(openai_tools[0]["type"], "function")
        self.assertEqual(openai_tools[0]["function"]["name"], "colab_run_code")

        anthropic_tools = to_anthropic_tools()
        self.assertEqual(len(anthropic_tools), 4)
        self.assertEqual(anthropic_tools[0]["name"], "colab_run_code")
        self.assertIn("input_schema", anthropic_tools[0])

    @patch("colab_mcp.mcp_client.ColabMCPClient.list_runtimes")
    @patch("colab_mcp.mcp_client.ColabMCPClient.create_runtime")
    def test_harness_ensure_runtime_reuse(self, mock_create, mock_list):
        mock_list.return_value = {
            "result": {
                "runtimes": [
                    {
                        "name": "runtimes/existing-t4",
                        "runtimeSpec": {"accelerator": "T4", "variant": "VARIANT_GPU"}
                    }
                ]
            }
        }
        harness = ColabAgentHarness(access_token="fake-token", default_accelerator="T4")
        rt = harness.ensure_runtime(accelerator="T4", reuse_existing=True)
        self.assertEqual(rt, "runtimes/existing-t4")
        mock_create.assert_not_called()

    @patch("colab_mcp.mcp_client.ColabMCPClient.create_session")
    @patch("colab_mcp.mcp_client.ColabMCPClient.execute_code")
    def test_harness_execute(self, mock_exec, mock_create_sess):
        mock_create_sess.return_value = {"result": {"name": "runtimes/rt-1/sessions/s-1"}}
        mock_exec.return_value = {
            "result": {"output": "Agent training complete: loss=0.01\n"}
        }

        harness = ColabAgentHarness(access_token="fake-token")
        harness.active_runtime = "runtimes/rt-1"
        res = harness.execute("print('training')")

        self.assertEqual(res["status"], "SUCCESS")
        self.assertIn("loss=0.01", res["output"])


if __name__ == "__main__":
    unittest.main()
