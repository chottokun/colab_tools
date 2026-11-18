"""Direct file transfer support with running Colab runtimes.
Uses authenticated connection URL and X-Colab-Runtime-Proxy-Token header.
"""

import base64
import json
import urllib.request
import urllib.error
from typing import Any, Dict


class JupyterProxy:
    """Handles direct file transfers with Colab runtime Jupyter proxy."""

    @staticmethod
    def upload_file(runtime_url: str, proxy_token: str, remote_path: str, content: bytes) -> Dict[str, Any]:
        """Uploads file content to the Colab runtime (/content/...).
        
        Args:
            runtime_url: connectionInfo.url from get_runtime
            proxy_token: connectionInfo.token (X-Colab-Runtime-Proxy-Token)
            remote_path: Target path (e.g. 'content/model.pt' or 'data.csv')
            content: Raw bytes to upload
        """
        clean_url = runtime_url.rstrip("/")
        clean_path = remote_path.lstrip("/")
        url = f"{clean_url}/api/contents/{clean_path}"

        b64_content = base64.b64encode(content).decode("ascii")
        payload = json.dumps({
            "type": "file",
            "format": "base64",
            "content": b64_content
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "X-Colab-Runtime-Proxy-Token": proxy_token
        }
        req = urllib.request.Request(url, data=payload, headers=headers, method="PUT")
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"File upload failed [HTTP {e.code}]: {err}") from e

    @staticmethod
    def download_file(runtime_url: str, proxy_token: str, remote_path: str) -> bytes:
        """Downloads raw bytes of a file from the Colab runtime.
        
        Args:
            runtime_url: connectionInfo.url from get_runtime
            proxy_token: connectionInfo.token
            remote_path: Source path (e.g. 'content/output.csv')
        """
        clean_url = runtime_url.rstrip("/")
        clean_path = remote_path.lstrip("/")
        url = f"{clean_url}/files/{clean_path}"

        headers = {
            "X-Colab-Runtime-Proxy-Token": proxy_token
        }
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"File download failed [HTTP {e.code}]: {err}") from e
