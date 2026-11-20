"""Unit tests for Colab Agent Harness, ContextOptimizer, and Tool Schemas."""

import datetime
import unittest
from unittest.mock import patch

from colab_mcp.agent.context import ContextOptimizer
from colab_mcp.agent.runner import ColabAgentHarness
from colab_mcp.agent.tools import to_anthropic_tools, to_openai_tools


class TestColabAgent(unittest.TestCase):

    def test_context_optimizer_official_schema_and_truncation(self):
        huge_stdout = "A" * 10000
        raw = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {
                "structuredContent": {
                    "stdout": huge_stdout,
                    "stderr": "warn: deprecated",
                    "result": "42",
                    "executionId": "exec-123",
                    "executionCount": 3,
                    "session": "runtimes/rt-1/sessions/s-1",
                    "outputTruncated": False,
                }
            },
        }
        res = ContextOptimizer.format_execution_result(raw)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["execution_id"], "exec-123")
        self.assertEqual(res["execution_count"], 3)
        self.assertTrue(res["output_truncated"])
        self.assertLess(len(res["output"]), 5000)
        self.assertIn("TRUNCATED", res["output"])
        self.assertEqual(res["result"], "42")

    def test_context_optimizer_execution_error(self):
        raw = {
            "result": {
                "structuredContent": {
                    "stdout": "",
                    "stderr": "",
                    "executionError": {
                        "name": "ZeroDivisionError",
                        "value": "division by zero",
                        "traceback": ["line1", "line2", "ZeroDivisionError: division by zero"],
                    },
                }
            }
        }
        res = ContextOptimizer.format_execution_result(raw)
        self.assertEqual(res["status"], "ERROR")
        self.assertEqual(res["error"]["name"], "ZeroDivisionError")
        self.assertEqual(res["error"]["value"], "division by zero")

    def test_tool_schemas(self):
        openai_tools = to_openai_tools()
        self.assertEqual(len(openai_tools), 5)
        self.assertEqual(openai_tools[0]["type"], "function")
        self.assertEqual(openai_tools[0]["function"]["name"], "colab_run_code")
        self.assertIn("L4", openai_tools[0]["function"]["parameters"]["properties"]["accelerator"]["enum"])

        anthropic_tools = to_anthropic_tools()
        self.assertEqual(len(anthropic_tools), 5)
        self.assertEqual(anthropic_tools[0]["name"], "colab_run_code")
        self.assertIn("input_schema", anthropic_tools[0])

    @patch("colab_mcp.mcp_client.ColabMCPClient.list_runtimes")
    @patch("colab_mcp.mcp_client.ColabMCPClient.create_runtime")
    def test_harness_ensure_runtime_reuse(self, mock_create, mock_list):
        mock_list.return_value = {
            "result": {
                "structuredContent": {
                    "runtimes": [
                        {
                            "name": "runtimes/existing-t4",
                            "runtimeSpec": {"accelerator": "T4", "variant": "VARIANT_GPU"},
                        }
                    ]
                }
            }
        }
        harness = ColabAgentHarness(access_token="fake-token", default_accelerator="T4")
        rt = harness.ensure_runtime(accelerator="T4", reuse_existing=True)
        self.assertEqual(rt, "runtimes/existing-t4")
        mock_create.assert_not_called()

    @patch("colab_mcp.mcp_client.ColabMCPClient.list_runtimes")
    @patch("colab_mcp.mcp_client.ColabMCPClient.create_runtime")
    def test_harness_tpu_provisioning(self, mock_create, mock_list):
        mock_list.return_value = {"result": {"structuredContent": {"runtimes": []}}}
        mock_create.return_value = {
            "result": {"structuredContent": {"name": "runtimes/tpu-1"}}
        }
        harness = ColabAgentHarness(access_token="fake-token")
        rt = harness.ensure_runtime(accelerator="V6E1")
        self.assertEqual(rt, "runtimes/tpu-1")
        sent_spec = mock_create.call_args[1]["spec"]
        self.assertEqual(sent_spec["variant"], "VARIANT_TPU")
        self.assertEqual(sent_spec["accelerator"], "V6E1")

    @patch("colab_mcp.mcp_client.ColabMCPClient.list_runtime_specs")
    @patch("colab_mcp.mcp_client.ColabMCPClient.list_runtimes")
    def test_harness_spec_validation_rejects_ineligible(self, mock_list, mock_specs):
        mock_list.return_value = {"result": {"structuredContent": {"runtimes": []}}}
        mock_specs.return_value = {
            "result": {
                "structuredContent": {
                    "runtimeSpecs": [
                        {
                            "eligible": True,
                            "key": {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_STANDARD"},
                        },
                        {
                            "eligible": False,
                            "key": {"variant": "VARIANT_GPU", "accelerator": "H100", "shape": "SHAPE_HIGHMEM"},
                        },
                    ]
                }
            }
        }
        harness = ColabAgentHarness(access_token="fake-token", validate_spec=True)
        with self.assertRaises(ValueError) as ctx:
            harness.ensure_runtime(accelerator="H100", shape="SHAPE_HIGHMEM")
        self.assertIn("not eligible", str(ctx.exception))

    @patch("colab_mcp.mcp_client.ColabMCPClient.get_runtime")
    def test_harness_auto_refreshes_expired_proxy_token(self, mock_get_rt):
        expired_time = (
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=1)
        ).isoformat()
        fresh_time = (
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=50)
        ).isoformat()
        mock_get_rt.return_value = {
            "result": {
                "structuredContent": {
                    "name": "runtimes/rt-1",
                    "connectionInfo": {
                        "url": "https://rt-1.colab.google.com",
                        "token": "fresh-token",
                        "expireTime": fresh_time,
                    },
                }
            }
        }
        harness = ColabAgentHarness(access_token="fake-token")
        harness.active_runtime = "runtimes/rt-1"
        harness.runtime_connection = {
            "url": "https://rt-1.colab.google.com",
            "token": "stale-token",
            "expireTime": expired_time,
        }
        conn = harness._get_connection_info()
        self.assertEqual(conn["token"], "fresh-token")
        mock_get_rt.assert_called_once_with("runtimes/rt-1")

    @patch("colab_mcp.mcp_client.ColabMCPClient.create_session")
    @patch("colab_mcp.mcp_client.ColabMCPClient.execute_code")
    def test_harness_execute(self, mock_exec, mock_create_sess):
        mock_create_sess.return_value = {
            "result": {"structuredContent": {"name": "runtimes/rt-1/sessions/s-1"}}
        }
        mock_exec.return_value = {
            "result": {
                "structuredContent": {
                    "stdout": "Agent training complete: loss=0.01\n",
                    "stderr": "",
                }
            }
        }

        harness = ColabAgentHarness(access_token="fake-token")
        harness.active_runtime = "runtimes/rt-1"
        res = harness.execute("print('training')")

        self.assertEqual(res["status"], "SUCCESS")
        self.assertIn("loss=0.01", res["output"])


if __name__ == "__main__":
    unittest.main()
