"""Direct file transfer support with running Colab runtimes.

Uses authenticated connection URL and X-Colab-Runtime-Proxy-Token header.
Complies with Colab Jupyter Contents API rules:
- Paths are absolute without leading slash (e.g. `content/data.csv` for `/content/data.csv`).
- Parent directories must exist (created one level per request with `{"type": "directory"}`).
- Each request must stay under 50 MB; larger files are chunked (`chunk`: 1, 2, ..., -1).
- Downloads support HTTP `Range` requests.
"""

import base64
import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from .config import ColabConfig


class JupyterProxy:
    """Handles direct file transfers with Colab runtime Jupyter proxy."""

    # Keep raw chunk size well under 50 MB (base64 expands by ~33%, so 30 MB -> ~40 MB)
    DEFAULT_CHUNK_SIZE_BYTES = 30 * 1024 * 1024

    @staticmethod
    def _validate_url(url: str) -> None:
        if not url.startswith(("https://", "http://")):
            raise ValueError(f"Invalid URL scheme: {url}")

    @classmethod
    def ensure_directory(
        cls,
        runtime_url: str,
        proxy_token: str,
        remote_dir: str,
        timeout: int = ColabConfig.DEFAULT_HTTP_TIMEOUT,
    ) -> None:
        """Creates parent directories one level at a time via Jupyter Contents API."""
        clean_url = runtime_url.rstrip("/")
        clean_dir = remote_dir.strip("/")
        if not clean_dir:
            return

        parts = clean_dir.split("/")
        current = ""
        for part in parts:
            current = f"{current}/{part}" if current else part
            # `/content` always exists on Colab runtimes
            if current == "content":
                continue
            url = f"{clean_url}/api/contents/{current}"
            cls._validate_url(url)
            payload = json.dumps({"type": "directory"}).encode("utf-8")
            headers = {
                "Content-Type": "application/json",
                "X-Colab-Runtime-Proxy-Token": proxy_token,
            }
            req = urllib.request.Request(url, data=payload, headers=headers, method="PUT")
            try:
                with urllib.request.urlopen(req, timeout=timeout):  # nosec B310
                    pass
            except urllib.error.HTTPError as e:
                # 409 Conflict indicates directory already exists
                if e.code != 409:
                    err = e.read().decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"Directory creation failed for '{current}' [HTTP {e.code}]: {err}"
                    ) from e

    @classmethod
    def upload_file(
        cls,
        runtime_url: str,
        proxy_token: str,
        remote_path: str,
        content: bytes,
        create_parents: bool = True,
        chunk_size: int = DEFAULT_CHUNK_SIZE_BYTES,
        timeout: int = ColabConfig.DEFAULT_HTTP_TIMEOUT,
    ) -> Dict[str, Any]:
        """Uploads file content to the Colab runtime (`/content/...`).

        Automatically creates missing parent directories and chunks files larger
        than `chunk_size` to comply with Colab's 50 MB per-request limit.
        """
        clean_url = runtime_url.rstrip("/")
        clean_path = remote_path.lstrip("/")
        url = f"{clean_url}/api/contents/{clean_path}"
        cls._validate_url(url)

        if create_parents and "/" in clean_path:
            parent_dir = clean_path.rsplit("/", 1)[0]
            cls.ensure_directory(clean_url, proxy_token, parent_dir, timeout=timeout)

        headers = {
            "Content-Type": "application/json",
            "X-Colab-Runtime-Proxy-Token": proxy_token,
        }

        # Single-shot upload for files within chunk_size
        if len(content) <= chunk_size:
            b64_content = base64.b64encode(content).decode("ascii")
            payload = json.dumps(
                {"type": "file", "format": "base64", "content": b64_content}
            ).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers=headers, method="PUT")
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                err = e.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"File upload failed [HTTP {e.code}]: {err}") from e

        # Chunked upload: chunk = 1, 2, ..., -1 on final piece
        total_len = len(content)
        offset = 0
        chunk_idx = 1
        last_resp: Dict[str, Any] = {}

        while offset < total_len:
            end = min(offset + chunk_size, total_len)
            piece = content[offset:end]
            is_last = end >= total_len
            chunk_num = -1 if is_last else chunk_idx

            b64_piece = base64.b64encode(piece).decode("ascii")
            payload = json.dumps(
                {
                    "type": "file",
                    "format": "base64",
                    "content": b64_piece,
                    "chunk": chunk_num,
                }
            ).encode("utf-8")

            req = urllib.request.Request(url, data=payload, headers=headers, method="PUT")
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
                    last_resp = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                err = e.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Chunked upload failed at chunk {chunk_num} [HTTP {e.code}]: {err}"
                ) from e

            offset = end
            chunk_idx += 1

        return last_resp

    @classmethod
    def download_file(
        cls,
        runtime_url: str,
        proxy_token: str,
        remote_path: str,
        byte_range: Optional[str] = None,
        timeout: int = ColabConfig.DEFAULT_HTTP_TIMEOUT,
    ) -> bytes:
        """Downloads raw bytes of a file from the Colab runtime.

        Args:
            runtime_url: connectionInfo.url from get_runtime
            proxy_token: connectionInfo.token
            remote_path: Source path (e.g. 'content/output.csv')
            byte_range: Optional HTTP Range header value (e.g. 'bytes=0-1023')
        """
        clean_url = runtime_url.rstrip("/")
        clean_path = remote_path.lstrip("/")
        url = f"{clean_url}/files/{clean_path}"
        cls._validate_url(url)

        headers = {"X-Colab-Runtime-Proxy-Token": proxy_token}
        if byte_range:
            headers["Range"] = byte_range

        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
                return resp.read()
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"File download failed [HTTP {e.code}]: {err}") from e
