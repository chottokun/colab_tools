"""Unit tests for Colab Agent Harness, ContextOptimizer, and Tool Schemas."""

import datetime
import unittest
from unittest.mock import patch

from colab_mcp.agent.context import ContextOptimizer
from colab_mcp.agent.runner import ColabAgentHarness
from colab_mcp.agent.tools import execute_agent_tool, to_anthropic_tools, to_openai_tools
from colab_mcp.config import ColabConfig


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

    @patch("colab_mcp.mcp_client.ColabMCPClient.list_runtimes")
    @patch("colab_mcp.mcp_client.ColabMCPClient.create_runtime")
    def test_harness_does_not_reuse_mismatched_shape(self, mock_create, mock_list):
        """既存ランタイムの shape が不一致の場合、再利用せず新規プロビジョニングすることを検証する。"""
        mock_list.return_value = {
            "result": {
                "structuredContent": {
                    "runtimes": [
                        {
                            "name": "runtimes/std-t4",
                            "runtimeSpec": {
                                "variant": "VARIANT_GPU",
                                "accelerator": "T4",
                                "shape": "SHAPE_STANDARD",
                            },
                        }
                    ]
                }
            }
        }
        mock_create.return_value = {
            "result": {"structuredContent": {"name": "runtimes/highmem-t4"}}
        }
        harness = ColabAgentHarness(access_token="fake-token")
        rt = harness.ensure_runtime(accelerator="T4", shape="SHAPE_HIGHMEM", reuse_existing=True)
        self.assertEqual(rt, "runtimes/highmem-t4")
        self.assertTrue(harness._owns_runtime)
        mock_create.assert_called_once()

    @patch("colab_mcp.mcp_client.ColabMCPClient.delete_session")
    @patch("colab_mcp.mcp_client.ColabMCPClient.delete_runtime")
    @patch("colab_mcp.mcp_client.ColabMCPClient.list_runtimes")
    def test_harness_teardown_ownership_and_error_handling(
        self, mock_list, mock_del_rt, mock_del_sess
    ):
        """未所有ランタイムの保護、force=True 削除、削除失敗時のエラー記録を検証する。"""
        mock_list.return_value = {
            "result": {
                "structuredContent": {
                    "runtimes": [
                        {
                            "name": "runtimes/shared-cpu",
                            "runtimeSpec": {
                                "variant": "VARIANT_CPU",
                                "accelerator": "NONE",
                                "shape": "SHAPE_STANDARD",
                            },
                        }
                    ]
                }
            }
        }
        harness = ColabAgentHarness(access_token="fake-token")
        harness.ensure_runtime(accelerator="NONE", reuse_existing=True)
        self.assertFalse(harness._owns_runtime)

        # 未所有ランタイムは通常 teardown で削除しない
        res_skip = harness.teardown()
        self.assertTrue(res_skip["skipped_unowned_runtime"])
        self.assertFalse(res_skip["runtime_deleted"])
        mock_del_rt.assert_not_called()

        # force=True なら未所有でも削除する
        harness.active_runtime = "runtimes/shared-cpu"
        harness.active_session = "runtimes/shared-cpu/sessions/s-1"
        res_force = harness.teardown(force=True)
        self.assertTrue(res_force["session_deleted"])
        self.assertTrue(res_force["runtime_deleted"])
        mock_del_sess.assert_called_once()
        mock_del_rt.assert_called_once()

        # 削除失敗時は例外黙殺せず errors に記録し active_runtime を保持する
        mock_del_rt.side_effect = RuntimeError("network error")
        harness.active_runtime = "runtimes/owned-rt"
        harness._owns_runtime = True
        with patch("time.sleep", return_value=None):
            res_err = harness.teardown(retries=1)
        self.assertFalse(res_err["runtime_deleted"])
        self.assertEqual(len(res_err["errors"]), 1)
        self.assertEqual(harness.active_runtime, "runtimes/owned-rt")

    @patch("colab_mcp.mcp_client.ColabMCPClient.list_runtime_specs")
    @patch("colab_mcp.mcp_client.ColabMCPClient.list_runtimes")
    def test_execute_agent_tool_dispatch(self, mock_list_rt, mock_specs):
        """execute_agent_tool のディスパッチと機密マスクを検証する。"""
        mock_specs.return_value = {"result": {"structuredContent": {"runtimeSpecs": []}}}
        mock_list_rt.return_value = {
            "result": {
                "structuredContent": {
                    "runtimes": [
                        {
                            "name": "runtimes/rt-1",
                            "connectionInfo": {"token": "secret-proxy-tok", "url": "https://rt-1.colab.dev"},
                        }
                    ]
                }
            }
        }

        harness = ColabAgentHarness(access_token="fake-token")
        with (
            patch.object(harness, "ensure_runtime", return_value="runtimes/rt-1") as mock_ensure,
            patch.object(harness, "execute", return_value={"status": "SUCCESS", "output": "2"}) as mock_exec,
            patch.object(harness, "sync_and_run", return_value={"status": "SUCCESS", "output": "ok"}) as mock_sync,
            patch.object(harness, "teardown", return_value={"runtime_deleted": True, "errors": []}) as mock_td,
        ):
            r1 = execute_agent_tool(harness, "colab_run_code", {"code": "print(2)", "accelerator": "NONE"})
            self.assertEqual(r1["status"], "SUCCESS")
            mock_ensure.assert_called_with(accelerator="NONE", shape="SHAPE_STANDARD")
            mock_exec.assert_called_once_with("print(2)")

            r2 = execute_agent_tool(
                harness,
                "colab_sync_and_run",
                {"code": "pass", "upload_files": {"a": "b"}, "download_files": {"c": "d"}},
            )
            self.assertEqual(r2["status"], "SUCCESS")
            mock_sync.assert_called_once()

            r3 = execute_agent_tool(harness, "colab_list_hardware_specs")
            self.assertEqual(r3, {"runtimeSpecs": []})

            r4 = execute_agent_tool(harness, "colab_list_runtimes")
            self.assertEqual(
                r4["runtimes"][0]["connectionInfo"]["token"], ColabConfig.REDACTED_VALUE
            )

            r5 = execute_agent_tool(harness, "colab_teardown")
            self.assertTrue(r5["runtime_deleted"])
            mock_td.assert_called_once_with(force=True)

            with self.assertRaises(ValueError):
                execute_agent_tool(harness, "unknown_tool")

    @patch("time.sleep", return_value=None)
    @patch("colab_mcp.mcp_client.ColabMCPClient.create_session")
    @patch("colab_mcp.mcp_client.ColabMCPClient.execute_code")
    def test_harness_execute_retries_lowercase_unavailable(
        self, mock_exec, mock_create_sess, _mock_sleep
    ):
        """execute() が小文字の unavailable エラーでも同一 execution_id で自動リトライすることを検証する。"""
        mock_create_sess.return_value = {
            "result": {"structuredContent": {"name": "runtimes/rt-1/sessions/s-1"}}
        }
        mock_exec.side_effect = [
            RuntimeError("MCP tool execution error: The service is currently unavailable."),
            {
                "result": {
                    "structuredContent": {
                        "stdout": "recovered ok\n",
                        "stderr": "",
                    }
                }
            },
        ]

        harness = ColabAgentHarness(access_token="fake-token")
        harness.active_runtime = "runtimes/rt-1"
        res = harness.execute("print('ok')")

        self.assertEqual(res["status"], "SUCCESS")
        self.assertIn("recovered ok", res["output"])
        self.assertEqual(mock_exec.call_count, 2)
        exec_id_1 = mock_exec.call_args_list[0].kwargs["execution_id"]
        exec_id_2 = mock_exec.call_args_list[1].kwargs["execution_id"]
        self.assertEqual(exec_id_1, exec_id_2)

    @patch("time.sleep", return_value=None)
    @patch("colab_mcp.mcp_client.ColabMCPClient.create_session")
    def test_harness_ensure_session_retries_unavailable(
        self, mock_create_sess, _mock_sleep
    ):
        """ensure_session() が一時的 unavailable エラー時に同一 session_id で再試行することを検証する。"""
        mock_create_sess.side_effect = [
            RuntimeError("MCP tool execution error: The service is currently unavailable."),
            {"result": {"structuredContent": {"name": "runtimes/rt-1/sessions/agent-ok"}}},
        ]

        harness = ColabAgentHarness(access_token="fake-token")
        harness.active_runtime = "runtimes/rt-1"
        sess = harness.ensure_session()

        self.assertEqual(sess, "runtimes/rt-1/sessions/agent-ok")
        self.assertEqual(mock_create_sess.call_count, 2)
        sid_1 = mock_create_sess.call_args_list[0].kwargs["session_id"]
        sid_2 = mock_create_sess.call_args_list[1].kwargs["session_id"]
        self.assertEqual(sid_1, sid_2)


if __name__ == "__main__":
    unittest.main()
