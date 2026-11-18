"""REST API Client for Google Colab (v1beta).
Endpoint: https://colaboratory.googleapis.com/v1beta
"""

import json
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from .config import ColabConfig


class ColabRESTClient:
    """Direct REST client for Google Colab REST API (v1beta)."""

    def __init__(self, access_token: Optional[str] = None, base_url: Optional[str] = None):
        self.access_token = access_token or ColabConfig.get_token()
        self.base_url = (base_url or ColabConfig.DEFAULT_REST_ENDPOINT).rstrip("/")

    def _request(self, method: str, path: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.access_token:
            raise ValueError("Access token is required for Colab REST API.")

        url = f"{self.base_url}/{path.lstrip('/')}"
        encoded_data = json.dumps(data).encode("utf-8") if data is not None else None
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token}"
        }
        req = urllib.request.Request(url, data=encoded_data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Colab REST API request failed [HTTP {e.code}]: {err}") from e

    def list_runtime_specs(self) -> Dict[str, Any]:
        """GET /v1beta/runtimespecs"""
        return self._request("GET", "runtimespecs")

    def list_runtimes(self) -> Dict[str, Any]:
        """GET /v1beta/runtimes"""
        return self._request("GET", "runtimes")

    def get_runtime(self, runtime_id: str) -> Dict[str, Any]:
        """GET /v1beta/runtimes/{runtime_id}"""
        return self._request("GET", f"runtimes/{runtime_id}")

    def create_runtime(self, runtime_spec: Dict[str, Any], runtime_id: Optional[str] = None) -> Dict[str, Any]:
        """POST /v1beta/runtimes"""
        payload = {"runtimeSpec": runtime_spec}
        query = f"?runtimeId={runtime_id}" if runtime_id else ""
        return self._request("POST", f"runtimes{query}", payload)

    def delete_runtime(self, runtime_id: str) -> Dict[str, Any]:
        """DELETE /v1beta/runtimes/{runtime_id}"""
        return self._request("DELETE", f"runtimes/{runtime_id}")
