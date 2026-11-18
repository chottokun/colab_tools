"""Unit tests for ColabMCPClient, ColabRESTClient, and JupyterProxy.
"""

import json
import unittest
from unittest.mock import patch, MagicMock

from colab_mcp.mcp_client import ColabMCPClient
from colab_mcp.api import ColabRESTClient
from colab_mcp.jupyter_proxy import JupyterProxy


class TestColabClients(unittest.TestCase):

    def setUp(self):
        self.client = ColabMCPClient(access_token="test-access-token")
        self.rest_client = ColabRESTClient(access_token="test-access-token")

    @patch("urllib.request.urlopen")
    def test_mcp_get_subscription(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"result": {"subscription": {"tier": "TIER_PRO"}}}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        res = self.client.get_subscription()
        self.assertEqual(res["result"]["subscription"]["tier"], "TIER_PRO")

    @patch("urllib.request.urlopen")
    def test_mcp_create_runtime(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"result": {"name": "runtimes/rt-100"}}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        spec = {"variant": "VARIANT_GPU", "accelerator": "T4"}
        res = self.client.create_runtime(spec=spec, runtime_id="rt-100")
        self.assertEqual(res["result"]["name"], "runtimes/rt-100")

    @patch("urllib.request.urlopen")
    def test_mcp_execute_code(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"result": {"output": "42\n"}}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        res = self.client.execute_code(code="print(42)", session="runtimes/rt/sessions/s1")
        self.assertEqual(res["result"]["output"], "42\n")

    @patch("urllib.request.urlopen")
    def test_rest_list_runtimes(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"runtimes": [{"name": "runtimes/rt-1"}]}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        res = self.rest_client.list_runtimes()
        self.assertEqual(len(res["runtimes"]), 1)

    @patch("urllib.request.urlopen")
    def test_jupyter_proxy_transfer(self, mock_urlopen):
        # upload
        mock_up = MagicMock()
        mock_up.read.return_value = json.dumps({"name": "file.txt"}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_up

        res = JupyterProxy.upload_file("https://rt.colab.google.com", "token", "content/file.txt", b"abc")
        self.assertEqual(res["name"], "file.txt")

        # download
        mock_down = MagicMock()
        mock_down.read.return_value = b"downloaded bytes"
        mock_urlopen.return_value.__enter__.return_value = mock_down

        content = JupyterProxy.download_file("https://rt.colab.google.com", "token", "content/file.txt")
        self.assertEqual(content, b"downloaded bytes")


if __name__ == "__main__":
    unittest.main()
