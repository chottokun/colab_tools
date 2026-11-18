"""Unit tests for colab-cli subcommands.
"""

import io
import json
import unittest
from unittest.mock import patch, MagicMock

import colab_mcp.cli as cli


class TestColabCLI(unittest.TestCase):

    def test_check_command_unconfigured(self):
        with patch("colab_mcp.config.ColabConfig.get_token", return_value=None):
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


if __name__ == "__main__":
    unittest.main()
