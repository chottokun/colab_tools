"""Google Colab stdio MCP Server Proxy.

Bridges standard local MCP clients (Claude Desktop, Claude Code, Cursor, etc.)
over standard I/O (stdin/stdout) to Google's hosted Colab MCP endpoint.
"""

import json
import sys
import urllib.error
import urllib.request
from typing import Dict, Optional

from .config import ColabConfig
from .mcp_client import parse_mcp_http_response

# 認証不要で呼び出せる公開 MCP メソッド（無駄なトークン取得・更新を回避する）
PUBLIC_METHODS = frozenset(
    {"initialize", "notifications/initialized", "tools/list", "ping"}
)
_SESSION_STATE: Dict[str, Optional[str]] = {"mcp_session_id": None}


def log(msg: str) -> None:
    sys.stderr.write(f"[Colab MCP Proxy] {msg}\n")
    sys.stderr.flush()


def _extract_mcp_session_id(headers_obj) -> Optional[str]:
    """HTTP レスポンスヘッダから大文字小文字を区別せず Mcp-Session-Id を取得する。"""
    if not headers_obj:
        return None
    if hasattr(headers_obj, "items"):
        for k, v in headers_obj.items():
            if isinstance(k, str) and k.lower() == "mcp-session-id" and v:
                return str(v)
    return None


def forward_request(
    req_dict: dict,
    token: Optional[str] = None,
    timeout: int = ColabConfig.DEFAULT_HTTP_TIMEOUT,
    quota_project: Optional[str] = None,
    session_state: Optional[Dict[str, Optional[str]]] = None,
) -> Optional[dict]:
    """JSON-RPC 2.0 リクエストを Colab Hosted MCP エンドポイントへ転送する。"""
    is_notification = "id" not in req_dict
    req_id = req_dict.get("id")
    method = req_dict.get("method", "")
    explicit_token = token is not None

    if explicit_token:
        auth_token = token
    elif method in PUBLIC_METHODS:
        auth_token = None
    else:
        auth_token = ColabConfig.get_token()

    # クオータプロジェクトが指定されていれば x-goog-user-project ヘッダに付与する
    qp = quota_project if quota_project is not None else ColabConfig.get_quota_project()
    state = session_state if session_state is not None else _SESSION_STATE

    # tools/list 等は認証不要だが tools/call にはトークンが必須
    if method == "tools/call" and not auth_token:
        if is_notification:
            return None
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32603,
                "message": (
                    "Authentication required. Set COLAB_API_TOKEN environment variable "
                    "or authenticate via gcloud ADC."
                ),
            },
        }

    data = json.dumps(req_dict).encode("utf-8")
    endpoint = ColabConfig.DEFAULT_MCP_ENDPOINT
    ColabConfig.validate_url(endpoint)

    for attempt in range(2):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
            if qp:
                headers["x-goog-user-project"] = qp
        if state.get("mcp_session_id"):
            headers["Mcp-Session-Id"] = state["mcp_session_id"]  # type: ignore[assignment]

        req = urllib.request.Request(
            endpoint,
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
                resp_headers = getattr(resp, "headers", None)
                sid = _extract_mcp_session_id(resp_headers)
                if sid:
                    state["mcp_session_id"] = sid
                if is_notification:
                    return None
                raw_body = resp.read().decode("utf-8")
                ctype = resp_headers.get("Content-Type", "") if resp_headers else ""
                return parse_mcp_http_response(raw_body, ctype)
        except urllib.error.HTTPError as e:
            if (
                e.code == 401
                and attempt == 0
                and not explicit_token
                and method not in PUBLIC_METHODS
            ):
                refreshed = ColabConfig.get_token(force_refresh=True)
                if refreshed:
                    auth_token = refreshed
                    continue
            if is_notification:
                log(f"Notification HTTP error {e.code}")
                return None
            body = e.read().decode("utf-8", errors="replace")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": e.code,
                    "message": f"Colab endpoint HTTP error {e.code}: {body}",
                },
            }
        except Exception as e:
            if is_notification:
                log(f"Notification dispatch error: {e}")
                return None
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32603,
                    "message": f"Proxy dispatch failed: {str(e)}",
                },
            }

    return None


def run_proxy() -> None:
    """Runs the stdio loop for MCP clients."""
    log("Started Colab MCP stdio proxy. Listening on stdin...")
    session_state: Dict[str, Optional[str]] = {"mcp_session_id": None}
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            log(f"JSON decode error: {e}")
            # JSON-RPC 2.0 規定の Parse Error (-32700) を返す
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {e}"},
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()
            continue

        resp = forward_request(req, session_state=session_state)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_proxy()
