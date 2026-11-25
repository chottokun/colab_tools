"""Unit tests for ColabMCPClient, ColabRESTClient, and JupyterProxy."""

import io
import json
import os
import tempfile
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from colab_mcp.api import ColabRESTClient
from colab_mcp.config import ColabConfig
from colab_mcp.jupyter_proxy import JupyterProxy
from colab_mcp.mcp_client import (
    ColabMCPClient,
    extract_tool_payload,
    parse_mcp_http_response,
)


class TestColabClients(unittest.TestCase):

    def setUp(self):
        ColabConfig.clear_token_cache()
        self.client = ColabMCPClient(access_token="test-access-token")
        self.rest_client = ColabRESTClient(access_token="test-access-token")

    def tearDown(self):
        ColabConfig.clear_token_cache()

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

    @patch("urllib.request.urlopen")
    def test_mcp_client_sends_quota_project_header(self, mock_urlopen):
        """ColabMCPClient が x-goog-user-project ヘッダを送信することを検証する。"""
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.read.return_value = json.dumps(
            {"result": {"structuredContent": {"tier": "TIER_PRO"}}}
        ).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        client = ColabMCPClient(access_token="tok", quota_project="my-proj")
        client.get_subscription()
        sent_req = mock_urlopen.call_args[0][0]
        self.assertEqual(sent_req.get_header("X-goog-user-project"), "my-proj")

    @patch("urllib.request.urlopen")
    def test_rest_client_list_runtime_specs_path_and_header(self, mock_urlopen):
        """ColabRESTClient.list_runtime_specs が /v1beta/runtimespecs とヘッダを使うことを検証する。"""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"runtimeSpecs": []}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        rest = ColabRESTClient(access_token="tok", quota_project="my-proj")
        res = rest.list_runtime_specs()
        self.assertEqual(res, {"runtimeSpecs": []})
        sent_req = mock_urlopen.call_args[0][0]
        self.assertEqual(
            sent_req.full_url, "https://colaboratory.googleapis.com/v1beta/runtimespecs"
        )
        self.assertEqual(sent_req.get_header("X-goog-user-project"), "my-proj")

    def test_config_get_quota_project_env_and_adc_fallback(self):
        """ColabConfig.get_quota_project の環境変数および ADC JSON フォールバックを検証する。"""
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump({"quota_project_id": "adc-proj-123"}, tf)
            adc_file = tf.name
        try:
            with (
                patch.object(ColabConfig, "load_env", return_value=None),
                patch.dict(os.environ, {"COLAB_QUOTA_PROJECT": "env-proj-456"}, clear=True),
                patch.object(ColabConfig, "_get_adc_json_path", return_value=adc_file),
            ):
                self.assertEqual(ColabConfig.get_quota_project(), "env-proj-456")

            with (
                patch.object(ColabConfig, "load_env", return_value=None),
                patch.dict(os.environ, {"COLAB_QUOTA_PROJECT": "YOUR_GCP_PROJECT_ID"}, clear=True),
                patch.object(ColabConfig, "_get_adc_json_path", return_value=adc_file),
            ):
                self.assertEqual(ColabConfig.get_quota_project(), "adc-proj-123")
        finally:
            os.unlink(adc_file)

    @patch("urllib.request.urlopen")
    def test_config_adc_json_token_refresh(self, mock_urlopen):
        """ColabConfig._get_adc_json_token が ADC JSON からアクセストークンを更新できることを検証する。"""
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(
                {
                    "type": "authorized_user",
                    "client_id": "cid",
                    "client_secret": "csec",
                    "refresh_token": "rtok",
                    "token_uri": "https://oauth2.googleapis.com/token",
                },
                tf,
            )
            adc_file = tf.name
        try:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {"access_token": "refreshed-access-tok"}
            ).encode("utf-8")
            mock_urlopen.return_value.__enter__.return_value = mock_resp

            with patch.object(ColabConfig, "_get_adc_json_path", return_value=adc_file):
                tok = ColabConfig._get_adc_json_token()
                self.assertEqual(tok, "refreshed-access-tok")
        finally:
            os.unlink(adc_file)

    def test_config_validate_url(self):
        """URL スキーム検証が HTTPS と localhost HTTP のみを許可することを検証する。"""
        ColabConfig.validate_url("https://colaboratory.googleapis.com/mcp")
        ColabConfig.validate_url("http://localhost:8080/api")
        ColabConfig.validate_url("http://127.0.0.1:8888/files")
        with self.assertRaises(ValueError):
            ColabConfig.validate_url("http://example.com/api")
        with self.assertRaises(ValueError):
            ColabConfig.validate_url("file:///etc/passwd")

    def test_config_redact_secrets(self):
        """ネストした辞書・リスト内の機密キーのみが安全にマスクされることを検証する。"""
        raw = {
            "name": "runtimes/rt-1",
            "connectionInfo": {
                "url": "https://rt-1.colab.dev",
                "token": "secret-proxy-token",
                "expireTime": "2026-01-01T00:00:00Z",
            },
            "nested": [{"refresh_token": "secret-rt", "safe": "ok"}],
        }
        masked = ColabConfig.redact_secrets(raw)
        self.assertEqual(masked["connectionInfo"]["token"], ColabConfig.REDACTED_VALUE)
        self.assertEqual(masked["connectionInfo"]["url"], "https://rt-1.colab.dev")
        self.assertEqual(masked["nested"][0]["refresh_token"], ColabConfig.REDACTED_VALUE)
        self.assertEqual(masked["nested"][0]["safe"], "ok")
        # 元の辞書が破壊されていないこと
        self.assertEqual(raw["connectionInfo"]["token"], "secret-proxy-token")

    def test_config_redact_secrets_embedded_json_text(self):
        """MCP content text に埋め込まれた JSON 文字列内の機密値もマスクされることを検証する。"""
        raw = {
            "content": [
                {
                    "type": "text",
                    "text": '{"runtimes": [{"connectionInfo": {"token": "secret-proxy-token", "url": "https://rt-1.colab.dev"}}]}',
                }
            ]
        }
        masked = ColabConfig.redact_secrets(raw)
        text = masked["content"][0]["text"]
        self.assertNotIn("secret-proxy-token", text)
        self.assertIn(ColabConfig.REDACTED_VALUE, text)
        self.assertIn("https://rt-1.colab.dev", text)

    def test_config_verify_adc_without_quota_project(self):
        """ADC 由来トークンでクオータプロジェクト未設定なら check が要対応を返すことを検証する。"""
        with (
            patch.object(
                ColabConfig, "get_token_with_source", return_value=("tok", "adc:json")
            ),
            patch.object(ColabConfig, "get_quota_project", return_value=None),
        ):
            ok, msg = ColabConfig.verify()
        self.assertFalse(ok)
        self.assertIn("quota_project is not set", msg)

    def test_config_api_error_hint(self):
        """既知の 403 メッセージに対処ヒントが付くことを検証する。"""
        hint = ColabConfig.api_error_hint(
            "The colaboratory.googleapis.com API requires a quota project"
        )
        self.assertIn("COLAB_QUOTA_PROJECT", hint)
        hint = ColabConfig.api_error_hint(
            "Colab API has not been used in project my-proj before or it is disabled",
            quota_project="my-proj",
        )
        self.assertIn("project=my-proj", hint)
        self.assertEqual(ColabConfig.api_error_hint("some other error"), "")

    def test_config_token_cache_and_force_refresh(self):
        """ADC トークンのキャッシュ再利用と force_refresh=True による再取得を検証する。"""
        with (
            patch.object(ColabConfig, "load_env", return_value=None),
            patch.dict(os.environ, {}, clear=True),
            patch.object(
                ColabConfig, "_get_gcloud_adc_token", side_effect=["tok-1", "tok-2"]
            ) as mock_gcloud,
        ):
            t1, s1 = ColabConfig.get_token_with_source()
            t2, s2 = ColabConfig.get_token_with_source()
            self.assertEqual((t1, s1), ("tok-1", "gcloud:adc"))
            self.assertEqual((t2, s2), ("tok-1", "gcloud:adc"))
            self.assertEqual(mock_gcloud.call_count, 1)

            t3, s3 = ColabConfig.get_token_with_source(force_refresh=True)
            self.assertEqual((t3, s3), ("tok-2", "gcloud:adc"))
            self.assertEqual(mock_gcloud.call_count, 2)

    def test_config_quota_project_non_dict_json(self):
        """ADC JSON が辞書以外（配列等）でも get_quota_project がクラッシュしないことを検証する。"""
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(["not", "a", "dict"], tf)
            adc_file = tf.name
        try:
            with (
                patch.object(ColabConfig, "load_env", return_value=None),
                patch.dict(os.environ, {}, clear=True),
                patch.object(ColabConfig, "_get_adc_json_path", return_value=adc_file),
            ):
                self.assertIsNone(ColabConfig.get_quota_project())
        finally:
            os.unlink(adc_file)

    @patch("urllib.request.urlopen")
    def test_mcp_client_retries_on_401_with_refresh(self, mock_urlopen):
        """ColabMCPClient が HTTP 401 受信時にトークンを更新して 1 回リトライすることを検証する。"""
        err_401 = urllib.error.HTTPError(
            "https://colaboratory.googleapis.com/mcp",
            401,
            "Unauthorized",
            {},
            io.BytesIO(b"expired"),
        )
        ok_resp = MagicMock()
        ok_resp.headers = {"Content-Type": "application/json"}
        ok_resp.read.return_value = json.dumps(
            {"result": {"structuredContent": {"tier": "SUBSCRIPTION_TIER_PRO"}}}
        ).encode("utf-8")
        ok_ctx = MagicMock()
        ok_ctx.__enter__.return_value = ok_resp

        mock_urlopen.side_effect = [err_401, ok_ctx]

        with (
            patch.object(ColabConfig, "get_quota_project", return_value="proj-1"),
            patch.object(
                ColabConfig,
                "get_token",
                side_effect=["init-tok", "stale-tok", "fresh-tok"],
            ) as mock_get_token,
        ):
            client = ColabMCPClient()
            res = client.get_subscription()
            self.assertEqual(extract_tool_payload(res)["tier"], "SUBSCRIPTION_TIER_PRO")
            self.assertEqual(mock_urlopen.call_count, 2)
            # 2回目は force_refresh=True で呼ばれていること
            self.assertEqual(mock_get_token.call_args_list[-1].kwargs.get("force_refresh"), True)

    @patch("urllib.request.urlopen")
    def test_rest_client_lro_polling_and_401_retry(self, mock_urlopen):
        """ColabRESTClient の 401 リトライと LRO ポーリング（完了・エラー）を検証する。"""
        err_401 = urllib.error.HTTPError(
            "https://colaboratory.googleapis.com/v1beta/runtimes",
            401,
            "Unauthorized",
            {},
            io.BytesIO(b"expired"),
        )
        op_pending = MagicMock()
        op_pending.read.return_value = json.dumps(
            {"name": "operations/op-1", "done": False}
        ).encode("utf-8")
        op_done = MagicMock()
        op_done.read.return_value = json.dumps(
            {
                "name": "operations/op-1",
                "done": True,
                "response": {"name": "runtimes/rt-rest-1"},
            }
        ).encode("utf-8")

        ctx_pending = MagicMock()
        ctx_pending.__enter__.return_value = op_pending
        ctx_done = MagicMock()
        ctx_done.__enter__.return_value = op_done
        mock_urlopen.side_effect = [err_401, ctx_pending, ctx_done]

        with (
            patch.object(ColabConfig, "get_quota_project", return_value="proj-1"),
            patch.object(
                ColabConfig,
                "get_token",
                side_effect=["init-tok", "stale-tok", "fresh-tok", "fresh-tok"],
            ),
            patch("time.sleep", return_value=None),
        ):
            rest = ColabRESTClient()
            rt = rest.create_runtime(
                {"variant": "VARIANT_CPU", "accelerator": "NONE", "shape": "SHAPE_STANDARD"},
                wait=True,
            )
            self.assertEqual(rt["name"], "runtimes/rt-rest-1")
            self.assertIn(
                "/v1/operations/op-1", mock_urlopen.call_args_list[-1][0][0].full_url
            )

        # LRO がエラー終了した場合に RuntimeError が送出されること
        rest_err = ColabRESTClient(access_token="tok", quota_project="proj-1")
        with self.assertRaises(RuntimeError):
            rest_err.wait_operation(
                {"name": "operations/op-err", "done": True, "error": {"code": 13, "message": "boom"}}
            )

    @patch("urllib.request.urlopen")
    def test_jupyter_proxy_url_encoding_and_streaming(self, mock_urlopen):
        """JupyterProxy のパス URL エンコードおよびストリーミング送受信を検証する。"""
        mock_resp = MagicMock()
        mock_resp.read.side_effect = [
            json.dumps({"name": "dir"}).encode("utf-8"),
            json.dumps({"name": "データ #1.txt"}).encode("utf-8"),
            json.dumps({"name": "データ #1.txt"}).encode("utf-8"),
        ]
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        with tempfile.NamedTemporaryFile("wb", delete=False) as tf:
            tf.write(b"0123456789ABCDEF")
            src_path = tf.name
        dst_path = src_path + ".down"
        try:
            res = JupyterProxy.upload_file_path(
                "https://rt.colab.google.com",
                "token",
                src_path,
                "content/test dir/データ #1.txt",
                chunk_size=10,
            )
            self.assertEqual(res["name"], "データ #1.txt")
            # 1 回目: ディレクトリ作成, 2・3 回目: チャンク 1 と -1
            self.assertEqual(mock_urlopen.call_count, 3)
            dir_url = mock_urlopen.call_args_list[0][0][0].full_url
            file_url = mock_urlopen.call_args_list[1][0][0].full_url
            self.assertIn("content/test%20dir", dir_url)
            self.assertIn("%E3%83%87%E3%83%BC%E3%82%BF%20%231.txt", file_url)

            # ダウンロードストリーミングの検証
            down_resp = MagicMock()
            down_resp.read.side_effect = [b"012345", b"6789", b""]
            mock_urlopen.return_value.__enter__.return_value = down_resp
            n = JupyterProxy.download_file_to_path(
                "https://rt.colab.google.com",
                "token",
                "content/test dir/データ #1.txt",
                dst_path,
                chunk_size=6,
            )
            self.assertEqual(n, 10)
            with open(dst_path, "rb") as df:
                self.assertEqual(df.read(), b"0123456789")
            down_url = mock_urlopen.call_args_list[-1][0][0].full_url
            self.assertIn("%E3%83%87%E3%83%BC%E3%82%BF%20%231.txt", down_url)
        finally:
            for p in (src_path, dst_path):
                if os.path.exists(p):
                    os.unlink(p)

    @patch("urllib.request.urlopen")
    def test_expired_env_token_falls_back_and_stays_cached(self, mock_urlopen):
        """失効した環境変数トークンでも 401 後に ADC へ切替・キャッシュされ 2 回目は 1 発成功することを検証する。"""
        err_401 = urllib.error.HTTPError(
            "https://colaboratory.googleapis.com/mcp",
            401,
            "Unauthorized",
            {},
            io.BytesIO(b"expired env token"),
        )
        ok_resp1 = MagicMock()
        ok_resp1.headers = {"Content-Type": "application/json"}
        ok_resp1.read.return_value = json.dumps(
            {"result": {"structuredContent": {"tier": "SUBSCRIPTION_TIER_PRO"}}}
        ).encode("utf-8")
        ok_ctx1 = MagicMock()
        ok_ctx1.__enter__.return_value = ok_resp1

        ok_resp2 = MagicMock()
        ok_resp2.headers = {"Content-Type": "application/json"}
        ok_resp2.read.return_value = json.dumps(
            {"result": {"structuredContent": {"runtimeSpecs": []}}}
        ).encode("utf-8")
        ok_ctx2 = MagicMock()
        ok_ctx2.__enter__.return_value = ok_resp2

        mock_urlopen.side_effect = [err_401, ok_ctx1, ok_ctx2]

        with (
            patch.object(ColabConfig, "load_env", return_value=None),
            patch.dict(
                os.environ,
                {"COLAB_API_TOKEN": "expired-env-tok", "COLAB_QUOTA_PROJECT": "proj-1"},
                clear=True,
            ),
            patch.object(
                ColabConfig, "_get_gcloud_adc_token", return_value="fresh-adc-tok"
            ) as mock_gcloud,
        ):
            client = ColabMCPClient()
            r1 = client.get_subscription()
            self.assertEqual(extract_tool_payload(r1)["tier"], "SUBSCRIPTION_TIER_PRO")

            # 2 回目は無効化された env トークンではなくキャッシュ済み ADC トークンが最初から使われる
            r2 = client.list_runtime_specs()
            self.assertEqual(extract_tool_payload(r2), {"runtimeSpecs": []})
            self.assertEqual(mock_urlopen.call_count, 3)
            self.assertEqual(mock_gcloud.call_count, 1)

            auth_headers = [
                call[0][0].get_header("Authorization")
                for call in mock_urlopen.call_args_list
            ]
            self.assertEqual(
                auth_headers,
                [
                    "Bearer expired-env-tok",
                    "Bearer fresh-adc-tok",
                    "Bearer fresh-adc-tok",
                ],
            )


if __name__ == "__main__":
    unittest.main()
