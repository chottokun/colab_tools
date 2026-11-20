"""Unit tests for ColabMCPClient, ColabRESTClient, and JupyterProxy."""

import json
import unittest
from unittest.mock import MagicMock, patch

from colab_mcp.api import ColabRESTClient
from colab_mcp.jupyter_proxy import JupyterProxy
from colab_mcp.mcp_client import (
    ColabMCPClient,
    extract_tool_payload,
    parse_mcp_http_response,
)


class TestColabClients(unittest.TestCase):

    def setUp(self):
        self.client = ColabMCPClient(access_token="test-access-token")
        self.rest_client = ColabRESTClient(access_token="test-access-token")

    @patch("urllib.request.urlopen")
    def test_mcp_get_subscription(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.read.return_value = json.dumps(
            {"result": {"structuredContent": {"tier": "TIER_PRO"}}}
        ).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        res = self.client.get_subscription()
        payload = extract_tool_payload(res)
        self.assertEqual(payload["tier"], "TIER_PRO")

    @patch("urllib.request.urlopen")
    def test_mcp_create_runtime_schema_fields(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.read.return_value = json.dumps(
            {"result": {"structuredContent": {"name": "runtimes/rt-100"}}}
        ).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        spec = {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_STANDARD"}
        res = self.client.create_runtime(spec=spec, runtime_id="rt-100")
        self.assertEqual(extract_tool_payload(res)["name"], "runtimes/rt-100")

        sent_req = mock_urlopen.call_args[0][0]
        sent_body = json.loads(sent_req.data.decode("utf-8"))
        args = sent_body["params"]["arguments"]
        self.assertIn("runtimeSpec", args["runtime"])
        self.assertEqual(args["runtimeId"], "rt-100")
        self.assertIn("requestId", args)

    @patch("urllib.request.urlopen")
    def test_mcp_execute_code_and_sse_parsing(self, mock_urlopen):
        sse_body = (
            'event: message\n'
            'data: {"jsonrpc":"2.0","id":"1","result":{"structuredContent":{"stdout":"42\\n","stderr":"","outputTruncated":false}}}\n\n'
        )
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "text/event-stream"}
        mock_resp.read.return_value = sse_body.encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        res = self.client.execute_code(code="print(42)", session="runtimes/rt/sessions/s1")
        payload = extract_tool_payload(res)
        self.assertEqual(payload["stdout"], "42\n")

    def test_extract_tool_payload_variants(self):
        # structuredContent
        p1 = extract_tool_payload({"result": {"structuredContent": {"a": 1}}})
        self.assertEqual(p1, {"a": 1})

        # content text JSON
        p2 = extract_tool_payload(
            {"result": {"content": [{"type": "text", "text": '{"b": 2}'}]}}
        )
        self.assertEqual(p2, {"b": 2})

        # isError raises
        with self.assertRaises(RuntimeError):
            extract_tool_payload(
                {"result": {"isError": True, "content": [{"type": "text", "text": "boom"}]}}
            )

        # JSON-RPC error raises
        with self.assertRaises(RuntimeError):
            extract_tool_payload({"error": {"code": -32000, "message": "failed"}})

    def test_parse_mcp_http_response_json(self):
        parsed = parse_mcp_http_response('{"jsonrpc": "2.0", "result": {"ok": true}}', "application/json")
        self.assertTrue(parsed["result"]["ok"])

    @patch("urllib.request.urlopen")
    def test_rest_list_runtimes(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"runtimes": [{"name": "runtimes/rt-1"}]}
        ).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        res = self.rest_client.list_runtimes()
        self.assertEqual(len(res["runtimes"]), 1)

    @patch("urllib.request.urlopen")
    def test_jupyter_proxy_transfer_and_parents(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"name": "file.txt"}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        res = JupyterProxy.upload_file(
            "https://rt.colab.google.com",
            "token",
            "content/subdir/nested/file.txt",
            b"abc",
        )
        self.assertEqual(res["name"], "file.txt")
        # 2 directory creations (content/subdir, content/subdir/nested) + 1 file upload
        self.assertEqual(mock_urlopen.call_count, 3)

        # download with Range
        mock_down = MagicMock()
        mock_down.read.return_value = b"downloaded bytes"
        mock_urlopen.return_value.__enter__.return_value = mock_down

        content = JupyterProxy.download_file(
            "https://rt.colab.google.com",
            "token",
            "content/file.txt",
            byte_range="bytes=0-15",
        )
        self.assertEqual(content, b"downloaded bytes")
        last_req = mock_urlopen.call_args[0][0]
        self.assertEqual(last_req.get_header("Range"), "bytes=0-15")

    @patch("urllib.request.urlopen")
    def test_jupyter_proxy_chunked_upload(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"name": "big.bin"}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        data = b"x" * 25
        JupyterProxy.upload_file(
            "https://rt.colab.google.com",
            "token",
            "content/big.bin",
            data,
            create_parents=False,
            chunk_size=10,
        )
        # 25 bytes with chunk_size=10 -> 3 requests (chunk=1, chunk=2, chunk=-1)
        self.assertEqual(mock_urlopen.call_count, 3)
        chunks_sent = [
            json.loads(call[0][0].data.decode("utf-8"))["chunk"]
            for call in mock_urlopen.call_args_list
        ]
        self.assertEqual(chunks_sent, [1, 2, -1])


if __name__ == "__main__":
    unittest.main()
