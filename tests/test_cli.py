"""Unit tests for colab-cli subcommands.
"""

import io
import json
import unittest
from unittest.mock import patch, MagicMock

import colab_mcp.cli as cli
from colab_mcp.config import ColabConfig


class TestColabCLI(unittest.TestCase):

    def test_check_command_unconfigured(self):
        # .env が存在する環境でも未設定ケースを確実に検証できるよう get_token_with_source をモック化
        with patch(
            "colab_mcp.config.ColabConfig.get_token_with_source",
            return_value=(None, "none"),
        ):
            with patch("sys.stdout", new=io.StringIO()) as fake_out:
                args = MagicMock()
                cli.cmd_check(args)
                output = fake_out.getvalue()
                self.assertIn("[ACTION REQUIRED]", output)

    @patch("colab_mcp.cli.get_client")
    def test_subscription_command(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.get_subscription.return_value = {"tier": "TIER_PRO"}
        mock_get_client.return_value = mock_client

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            args = MagicMock()
            cli.cmd_subscription(args)
            output = fake_out.getvalue()
            self.assertIn("TIER_PRO", output)

    @patch("colab_mcp.cli.get_client")
    def test_exec_command_code(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.execute_code.return_value = {"output": "result ok"}
        mock_get_client.return_value = mock_client

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            args = MagicMock()
            args.code = "print(1)"
            args.file = None
            args.session = "sess-1"
            args.runtime = None
            cli.cmd_exec(args)
            output = fake_out.getvalue()
            self.assertIn("result ok", output)

    @patch("colab_mcp.cli.get_client")
    def test_runtimes_commands_mask_secrets_by_default(self, mock_get_client):
        """runtimes list/get の既定出力が token をマスクし、--show-secrets で解除することを検証する。"""
        mock_client = MagicMock()
        mock_client.get_runtime.return_value = {
            "result": {
                "structuredContent": {
                    "name": "runtimes/rt-1",
                    "connectionInfo": {"token": "raw-secret-token", "url": "https://rt-1.colab.dev"},
                }
            }
        }
        mock_client.list_runtimes.return_value = {
            "result": {
                "structuredContent": {
                    "runtimes": [
                        {
                            "name": "runtimes/rt-1",
                            "connectionInfo": {"token": "raw-secret-token", "url": "https://rt-1.colab.dev"},
                        }
                    ]
                }
            }
        }
        mock_get_client.return_value = mock_client

        args = MagicMock()
        args.name = "rt-1"
        args.show_secrets = False
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            cli.cmd_runtimes_get(args)
            out_masked = fake_out.getvalue()
            self.assertIn(ColabConfig.REDACTED_VALUE, out_masked)
            self.assertNotIn("raw-secret-token", out_masked)

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            cli.cmd_runtimes_list(args)
            self.assertIn(ColabConfig.REDACTED_VALUE, fake_out.getvalue())

        args.show_secrets = True
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            cli.cmd_runtimes_get(args)
            self.assertIn("raw-secret-token", fake_out.getvalue())

    def test_agent_run_cpu_flag_passes_none_accelerator(self):
        """agent run --cpu が NONE アクセラレータ（VARIANT_CPU）を渡すことを検証する。"""
        with patch("colab_mcp.agent.runner.ColabAgentHarness") as mock_harness_cls:
            mock_harness = mock_harness_cls.return_value.__enter__.return_value
            mock_harness.execute.return_value = {"status": "SUCCESS", "output": "ok"}

            args = MagicMock()
            args.code = "print(1)"
            args.file = None
            args.cpu = True
            args.gpu = None
            args.tpu = None
            args.shape = "standard"
            args.validate_spec = False
            args.teardown = True

            with patch("sys.stdout", new=io.StringIO()):
                cli.cmd_agent_run(args)

            mock_harness.ensure_runtime.assert_called_once_with(
                accelerator="NONE", shape="SHAPE_STANDARD"
            )


if __name__ == "__main__":
    unittest.main()
