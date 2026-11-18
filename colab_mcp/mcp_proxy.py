"""Google Colab stdio MCP Server Proxy.

Bridges standard local MCP clients (Claude Desktop, Claude Code, Cursor, etc.)
over standard I/O (stdin/stdout) to Google's hosted Colab MCP endpoint.
"""

import json
import sys
import urllib.request
import urllib.error
from typing import Optional

from .config import ColabConfig


def log(msg: str):
    sys.stderr.write(f"[Colab MCP Proxy] {msg}\n")
    sys.stderr.flush()


def forward_request(req_dict: dict, token: Optional[str] = None) -> dict:
    auth_token = token or ColabConfig.get_token()
    req_id = req_dict.get("id")

    # tools/list can be fetched without token, but tools/call requires it
    if req_dict.get("method") == "tools/call" and not auth_token:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32603,
                "message": "Authentication required. Set COLAB_API_TOKEN environment variable."
            }
        }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    data = json.dumps(req_dict).encode("utf-8")
    endpoint = ColabConfig.DEFAULT_MCP_ENDPOINT
    if not endpoint.startswith(("https://", "http://")):
        raise ValueError(f"Insecure or invalid URL scheme in MCP endpoint: {endpoint}")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers=headers,
        method="POST"
    )
    try:
        with urllib.request.urlopen(req) as resp:  # nosec B310
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": e.code,
                "message": f"Colab endpoint HTTP error {e.code}: {body}"
            }
        }
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32603,
                "message": f"Proxy dispatch failed: {str(e)}"
            }
        }


def run_proxy():
    """Runs the stdio loop for MCP clients."""
    log("Started Colab MCP stdio proxy. Listening on stdin...")
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
            continue

        resp = forward_request(req)
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    run_proxy()
