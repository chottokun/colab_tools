"""REST API Client for Google Colab (v1beta).

Endpoint: https://colaboratory.googleapis.com/v1beta
"""

import json
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
    ):
        self.access_token = access_token if access_token is not None else ColabConfig.get_token()
        self.base_url = (base_url or ColabConfig.DEFAULT_REST_ENDPOINT).rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.access_token:
            raise ValueError("Access token is required for Colab REST API.")

        url = f"{self.base_url}/{path.lstrip('/')}"
        if params:
            filtered = {k: v for k, v in params.items() if v is not None}
            if filtered:
                url = f"{url}?{urllib.parse.urlencode(filtered)}"

        if not url.startswith(("https://", "http://")):
            raise ValueError(f"Invalid URL scheme: {url}")

        encoded_data = json.dumps(data).encode("utf-8") if data is not None else None
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }
        req = urllib.request.Request(
            url, data=encoded_data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # nosec B310
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Colab REST API request failed [HTTP {e.code}]: {err}"
            ) from e

    def get_subscription(self) -> Dict[str, Any]:
        """GET /v1beta/subscription"""
        return self._request("GET", "subscription")

    def list_runtime_specs(self) -> Dict[str, Any]:
        """GET /v1beta/runtimeSpecs"""
        return self._request("GET", "runtimeSpecs")

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
    ) -> Dict[str, Any]:
        """POST /v1beta/runtimes"""
        payload: Dict[str, Any] = {"runtimeSpec": runtime_spec}
        if version:
            payload["version"] = version
        params = {
            "runtimeId": runtime_id,
            "requestId": request_id or str(uuid.uuid4()),
        }
        return self._request("POST", "runtimes", payload, params=params)

    def delete_runtime(
        self, runtime_id: str, allow_missing: bool = False
    ) -> Dict[str, Any]:
        """DELETE /v1beta/runtimes/{runtime_id}"""
        clean = runtime_id.removeprefix("runtimes/")
        params = {"allowMissing": "true"} if allow_missing else None
        return self._request("DELETE", f"runtimes/{clean}", params=params)
