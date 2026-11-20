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
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

from .config import ColabConfig


class JupyterProxy:
    """Handles direct file transfers with Colab runtime Jupyter proxy."""

    # Keep raw chunk size well under 50 MB (base64 expands by ~33%, so 30 MB -> ~40 MB)
    DEFAULT_CHUNK_SIZE_BYTES = 30 * 1024 * 1024

    @staticmethod
    def _validate_url(url: str) -> None:
        ColabConfig.validate_url(url)

    @staticmethod
    def _encode_remote_path(remote_path: str) -> str:
        """リモートパスを URL 安全な形式へパーセントエンコードする。"""
        return urllib.parse.quote(remote_path.lstrip("/"), safe="/")

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

        parts = [p for p in clean_dir.split("/") if p]
        current = ""
        for part in parts:
            current = f"{current}/{part}" if current else part
            # `/content` always exists on Colab runtimes
            if current == "content":
                continue
            url = f"{clean_url}/api/contents/{cls._encode_remote_path(current)}"
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
    def _prepare_upload_url(
        cls,
        runtime_url: str,
        proxy_token: str,
        remote_path: str,
        create_parents: bool,
        timeout: int,
    ) -> str:
        """アップロード先 URL を構築・検証し、必要なら親ディレクトリを作成する。"""
        clean_url = runtime_url.rstrip("/")
        clean_path = remote_path.lstrip("/")
        url = f"{clean_url}/api/contents/{cls._encode_remote_path(clean_path)}"
        cls._validate_url(url)

        if create_parents and "/" in clean_path:
            parent_dir = clean_path.rsplit("/", 1)[0]
            cls.ensure_directory(clean_url, proxy_token, parent_dir, timeout=timeout)
        return url

    @classmethod
    def _put_file_chunk(
        cls,
        url: str,
        proxy_token: str,
        piece: bytes,
        chunk_num: Optional[int],
        timeout: int,
    ) -> Dict[str, Any]:
        """ファイルデータまたは 1 チャンクを Jupyter Contents API へ PUT 送信する。"""
        headers = {
            "Content-Type": "application/json",
            "X-Colab-Runtime-Proxy-Token": proxy_token,
        }
        body: Dict[str, Any] = {
            "type": "file",
            "format": "base64",
            "content": base64.b64encode(piece).decode("ascii"),
        }
        if chunk_num is not None:
            body["chunk"] = chunk_num

        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers=headers, method="PUT")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            if chunk_num is not None:
                raise RuntimeError(
                    f"Chunked upload failed at chunk {chunk_num} [HTTP {e.code}]: {err}"
                ) from e
            raise RuntimeError(f"File upload failed [HTTP {e.code}]: {err}") from e

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
        url = cls._prepare_upload_url(
            runtime_url, proxy_token, remote_path, create_parents, timeout
        )

        # Single-shot upload for files within chunk_size
        if len(content) <= chunk_size:
            return cls._put_file_chunk(url, proxy_token, content, None, timeout)

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
            last_resp = cls._put_file_chunk(url, proxy_token, piece, chunk_num, timeout)
            offset = end
            chunk_idx += 1

        return last_resp

    @classmethod
    def upload_file_path(
        cls,
        runtime_url: str,
        proxy_token: str,
        local_path: str,
        remote_path: str,
        create_parents: bool = True,
        chunk_size: int = DEFAULT_CHUNK_SIZE_BYTES,
        timeout: int = ColabConfig.DEFAULT_HTTP_TIMEOUT,
    ) -> Dict[str, Any]:
        """ローカルファイルをチャンク単位でストリーミング読み込みしアップロードする。"""
        url = cls._prepare_upload_url(
            runtime_url, proxy_token, remote_path, create_parents, timeout
        )
        total_len = os.path.getsize(local_path)

        with open(local_path, "rb") as f:
            if total_len <= chunk_size:
                return cls._put_file_chunk(url, proxy_token, f.read(), None, timeout)

            offset = 0
            chunk_idx = 1
            last_resp: Dict[str, Any] = {}
            while offset < total_len:
                piece = f.read(chunk_size)
                if not piece:
                    break
                end = offset + len(piece)
                is_last = end >= total_len
                chunk_num = -1 if is_last else chunk_idx
                last_resp = cls._put_file_chunk(
                    url, proxy_token, piece, chunk_num, timeout
                )
                offset = end
                chunk_idx += 1
            return last_resp

    @classmethod
    def _build_download_request(
        cls,
        runtime_url: str,
        proxy_token: str,
        remote_path: str,
        byte_range: Optional[str] = None,
    ) -> urllib.request.Request:
        """ファイルダウンロード用の HTTP GET リクエストを構築する。"""
        clean_url = runtime_url.rstrip("/")
        url = f"{clean_url}/files/{cls._encode_remote_path(remote_path)}"
        cls._validate_url(url)

        headers = {"X-Colab-Runtime-Proxy-Token": proxy_token}
        if byte_range:
            headers["Range"] = byte_range
        return urllib.request.Request(url, headers=headers, method="GET")

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
        req = cls._build_download_request(
            runtime_url, proxy_token, remote_path, byte_range=byte_range
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
                return resp.read()
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"File download failed [HTTP {e.code}]: {err}") from e

    @classmethod
    def download_file_to_path(
        cls,
        runtime_url: str,
        proxy_token: str,
        remote_path: str,
        local_path: str,
        byte_range: Optional[str] = None,
        chunk_size: int = 1024 * 1024,
        timeout: int = ColabConfig.DEFAULT_HTTP_TIMEOUT,
    ) -> int:
        """リモートファイルをローカルファイルへストリーミングダウンロードし、バイト数を返す。"""
        req = cls._build_download_request(
            runtime_url, proxy_token, remote_path, byte_range=byte_range
        )
        total_bytes = 0
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
                with open(local_path, "wb") as f:
                    while True:
                        buf = resp.read(chunk_size)
                        if not buf:
                            break
                        f.write(buf)
                        total_bytes += len(buf)
            return total_bytes
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"File download failed [HTTP {e.code}]: {err}") from e
