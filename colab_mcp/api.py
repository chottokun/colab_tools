"""REST API Client for Google Colab (v1beta).

Endpoint: https://colaboratory.googleapis.com/v1beta
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Dict, Optional

from .config import ColabConfig


class ColabRESTClient:
    """Direct REST client for Google Colab REST API (v1beta)."""

    def __init__(
        self,
        access_token: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = ColabConfig.DEFAULT_HTTP_TIMEOUT,
        quota_project: Optional[str] = None,
        operations_base_url: Optional[str] = None,
    ):
        self._explicit_token: bool = access_token is not None
        self.access_token = (
            access_token if self._explicit_token else ColabConfig.get_token()
        )
        self.base_url = (base_url or ColabConfig.DEFAULT_REST_ENDPOINT).rstrip("/")
        if operations_base_url is not None:
            self.operations_base_url = operations_base_url.rstrip("/")
        elif self.base_url.endswith("/v1beta"):
            # LRO 確認用エンドポイントは /v1/operations 配下を使用する
            self.operations_base_url = self.base_url[: -len("/v1beta")] + "/v1"
        else:
            self.operations_base_url = self.base_url
        self.timeout = timeout
        # Google Cloud クオータプロジェクト（x-goog-user-project ヘッダ用）を解決
        self.quota_project = (
            quota_project if quota_project is not None else ColabConfig.get_quota_project()
        )

    def _resolve_token(self, force_refresh: bool = False) -> Optional[str]:
        """認証トークンを解決する。明示指定がなければ Config から取得・更新する。"""
        if self._explicit_token:
            return self.access_token
        self.access_token = ColabConfig.get_token(force_refresh=force_refresh)
        return self.access_token

    def _request(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        base_url_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        token = self._resolve_token()
        if not token:
            raise ValueError("Access token is required for Colab REST API.")

        base = (base_url_override or self.base_url).rstrip("/")
        url = f"{base}/{path.lstrip('/')}"
        if params:
            filtered = {k: v for k, v in params.items() if v is not None}
            if filtered:
                url = f"{url}?{urllib.parse.urlencode(filtered)}"

        ColabConfig.validate_url(url)

        encoded_data = json.dumps(data).encode("utf-8") if data is not None else None
        for attempt in range(2):
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {token}",
            }
            # ADC 認証等で必須となるクオータプロジェクトヘッダを付与
            if self.quota_project:
                headers["x-goog-user-project"] = self.quota_project
            req = urllib.request.Request(
                url, data=encoded_data, headers=headers, method=method
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # nosec B310
                    body = resp.read().decode("utf-8")
                    return json.loads(body) if body else {}
            except urllib.error.HTTPError as e:
                if e.code == 401 and attempt == 0 and not self._explicit_token:
                    new_token = self._resolve_token(force_refresh=True)
                    if new_token:
                        token = new_token
                        continue
                err = e.read().decode("utf-8", errors="replace")
                hint = ColabConfig.api_error_hint(err, self.quota_project)
                raise RuntimeError(
                    f"Colab REST API request failed [HTTP {e.code}]: {err}"
                    + (f"\n{hint}" if hint else "")
                ) from e

        raise RuntimeError("Colab REST API request failed after token refresh retry.")

    def list_operations(self) -> Dict[str, Any]:
        """Long-Running Operation 一覧を取得する（GET /v1/operations）。"""
        return self._request(
            "GET", "operations", base_url_override=self.operations_base_url
        )

    def get_operation(self, operation_name: str) -> Dict[str, Any]:
        """指定した Long-Running Operation の最新状態を取得する。"""
        clean = (
            operation_name
            if operation_name.startswith("operations/")
            else f"operations/{operation_name}"
        )
        return self._request(
            "GET", clean, base_url_override=self.operations_base_url
        )

    def wait_operation(
        self,
        operation: Dict[str, Any],
        timeout: int = 180,
        poll_interval: float = 2.0,
    ) -> Dict[str, Any]:
        """Long-Running Operation の完了を待機し、結果の response 辞書を返す。"""
        if not isinstance(operation, dict):
            return operation

        op_name = operation.get("name", "")
        is_lro = (
            isinstance(op_name, str) and op_name.startswith("operations/")
        ) or ("done" in operation)
        if not is_lro:
            return operation

        current = operation
        deadline = time.monotonic() + timeout
        while True:
            if current.get("done"):
                if "error" in current and current["error"]:
                    raise RuntimeError(
                        f"Colab REST operation failed: {current['error']}"
                    )
                resp = current.get("response")
                return resp if isinstance(resp, dict) else current

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Colab REST operation timed out after {timeout}s: {op_name}"
                )

            if not op_name:
                raise RuntimeError(
                    f"Incomplete LRO missing operation name: {current}"
                )

            time.sleep(poll_interval)
            current = self.get_operation(op_name)

    def get_subscription(self) -> Dict[str, Any]:
        """GET /v1beta/subscription"""
        return self._request("GET", "subscription")

    def list_runtime_specs(self) -> Dict[str, Any]:
        """GET /v1beta/runtimespecs"""
        # 公式 REST エンドポイントは小文字の runtimespecs を使用する
        return self._request("GET", "runtimespecs")

    def list_runtimes(self) -> Dict[str, Any]:
        """GET /v1beta/runtimes"""
        return self._request("GET", "runtimes")

    def get_runtime(self, runtime_id: str) -> Dict[str, Any]:
        """GET /v1beta/runtimes/{runtime_id}"""
        clean = runtime_id.removeprefix("runtimes/")
        return self._request("GET", f"runtimes/{clean}")

    def create_runtime(
        self,
        runtime_spec: Dict[str, Any],
        runtime_id: Optional[str] = None,
        request_id: Optional[str] = None,
        version: Optional[str] = None,
        wait: bool = True,
        wait_timeout: int = 180,
    ) -> Dict[str, Any]:
        """POST /v1beta/runtimes（wait=True の場合は LRO 完了まで待機して Runtime を返す）"""
        payload: Dict[str, Any] = {"runtimeSpec": runtime_spec}
        if version:
            payload["version"] = version
        params = {
            "runtimeId": runtime_id,
            "requestId": request_id or str(uuid.uuid4()),
        }
        res = self._request("POST", "runtimes", payload, params=params)
        if wait:
            return self.wait_operation(res, timeout=wait_timeout)
        return res

    def delete_runtime(
        self,
        runtime_id: str,
        allow_missing: bool = False,
        wait: bool = True,
        wait_timeout: int = 120,
    ) -> Dict[str, Any]:
        """DELETE /v1beta/runtimes/{runtime_id}（wait=True の場合は LRO 完了まで待機する）"""
        clean = runtime_id.removeprefix("runtimes/")
        params = {"allowMissing": "true"} if allow_missing else None
        res = self._request("DELETE", f"runtimes/{clean}", params=params)
        if wait:
            return self.wait_operation(res, timeout=wait_timeout)
        return res
