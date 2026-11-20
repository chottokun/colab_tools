"""Client for interacting with the Google Colab hosted MCP Endpoint.

Protocol: Model Context Protocol (JSON-RPC 2.0 over Streamable HTTP / JSON)
Endpoint: https://colaboratory.googleapis.com/mcp
"""

import json
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, List, Optional

from .config import ColabConfig


def parse_mcp_http_response(raw_body: str, content_type: str = "") -> Dict[str, Any]:
    """Parses a JSON or Server-Sent Events (SSE) MCP HTTP response.

    Streamable HTTP MCP servers may return either application/json or
    text/event-stream containing one or more `data: {...}` events.
    When SSE is returned, the terminal JSON-RPC object is extracted.
    """
    text = raw_body.strip()
    if not text:
        return {}

    is_sse = "text/event-stream" in content_type.lower() or text.startswith("data:") or "\ndata:" in text
    if not is_sse:
        return json.loads(text)

    last_json: Optional[Dict[str, Any]] = None
    data_lines: List[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("data:"):
            data_lines.append(stripped[5:].strip())
        elif stripped == "" and data_lines:
            chunk = "\n".join(data_lines)
            data_lines = []
            if chunk and chunk != "[DONE]":
                try:
                    parsed = json.loads(chunk)
                    if isinstance(parsed, dict):
                        last_json = parsed
                except json.JSONDecodeError:
                    continue

    if data_lines:
        chunk = "\n".join(data_lines)
        if chunk and chunk != "[DONE]":
            try:
                parsed = json.loads(chunk)
                if isinstance(parsed, dict):
                    last_json = parsed
            except json.JSONDecodeError:
                pass

    if last_json is not None:
        return last_json

    raise ValueError(f"Unable to parse MCP SSE response: {text[:200]}")


def extract_tool_payload(raw_response: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts structured payload from a JSON-RPC MCP tools/call response.

    Handles:
      1. JSON-RPC top-level `error`
      2. MCP `result.isError == True`
      3. MCP `result.structuredContent` (official Colab MCP format)
      4. MCP `result.content[0].text` (JSON-encoded text block)
      5. Direct `result` dict (legacy/mock fallback)
    """
    if not isinstance(raw_response, dict):
        return {}

    if "error" in raw_response and raw_response["error"]:
        err = raw_response["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        code = err.get("code", "UNKNOWN") if isinstance(err, dict) else "UNKNOWN"
        raise RuntimeError(f"MCP JSON-RPC error [{code}]: {msg}")

    result = raw_response.get("result")
    if not isinstance(result, dict):
        return raw_response

    if result.get("isError"):
        content = result.get("content", [])
        err_text = ""
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict):
                err_text = first.get("text", str(first))
        raise RuntimeError(f"MCP tool execution error: {err_text or result}")

    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured

    content = result.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict) and isinstance(first.get("text"), str):
            try:
                parsed = json.loads(first["text"])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {"text": first["text"]}

    return result


class ColabMCPClient:
    """Client for Google Colab hosted MCP tools (schema-aligned)."""

    def __init__(
        self,
        access_token: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: int = ColabConfig.DEFAULT_HTTP_TIMEOUT,
    ):
        self.access_token = access_token if access_token is not None else ColabConfig.get_token()
        self.endpoint = endpoint or ColabConfig.DEFAULT_MCP_ENDPOINT
        self.timeout = timeout

    def _validate_endpoint(self) -> None:
        if not self.endpoint.startswith(("https://", "http://")):
            raise ValueError(f"Invalid endpoint scheme: {self.endpoint}")

    def call_tool(
        self, tool_name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Calls an MCP tool by name with arguments."""
        if not self.access_token:
            raise ValueError(
                "Access token is required. Set COLAB_API_TOKEN environment variable "
                "or pass access_token to ColabMCPClient."
            )

        self._validate_endpoint()
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {},
            },
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {self.access_token}",
        }

        req = urllib.request.Request(
            self.endpoint, data=data, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # nosec B310
                raw_body = resp.read().decode("utf-8")
                ctype = resp.headers.get("Content-Type", "") if getattr(resp, "headers", None) else ""
                return parse_mcp_http_response(raw_body, ctype)
        except urllib.error.HTTPError as e:
            error_content = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Colab MCP call failed [HTTP {e.code}]: {error_content}"
            ) from e

    def list_tools(self) -> List[Dict[str, Any]]:
        """Lists available MCP tools and their schemas from the server."""
        self._validate_endpoint()
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/list",
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # nosec B310
            raw_body = resp.read().decode("utf-8")
            ctype = resp.headers.get("Content-Type", "") if getattr(resp, "headers", None) else ""
            data = parse_mcp_http_response(raw_body, ctype)
            return data.get("result", {}).get("tools", [])

    # High-level tool methods aligned with official colaboratory.googleapis.com/mcp schemas

    def get_subscription(self) -> Dict[str, Any]:
        """Gets user's Colab subscription tier (free, Pro, Pro+)."""
        return self.call_tool("get_subscription")

    def list_runtime_specs(self) -> Dict[str, Any]:
        """Lists CPU, GPU, TPU hardware specs and eligibility."""
        return self.call_tool("list_runtime_specs")

    def list_runtimes(self) -> Dict[str, Any]:
        """Lists all active runtimes assigned to the user."""
        return self.call_tool("list_runtimes")

    def get_runtime(self, runtime_name: str) -> Dict[str, Any]:
        """Gets runtime details, including connectionInfo and proxy token."""
        return self.call_tool("get_runtime", {"name": runtime_name})

    def create_runtime(
        self,
        spec: Dict[str, Any],
        runtime_id: Optional[str] = None,
        request_id: Optional[str] = None,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Provisions a new runtime assignment.

        Uses AIP-155 idempotent `requestId` by default to prevent duplicate allocations on retries.
        """
        runtime_obj: Dict[str, Any] = {"runtimeSpec": spec}
        if version:
            runtime_obj["version"] = version

        args: Dict[str, Any] = {
            "runtime": runtime_obj,
            "requestId": request_id or str(uuid.uuid4()),
        }
        if runtime_id:
            args["runtimeId"] = runtime_id
        return self.call_tool("create_runtime", args)

    def delete_runtime(
        self, runtime_name: str, allow_missing: bool = False
    ) -> Dict[str, Any]:
        """Terminates and deletes a Colab runtime."""
        args: Dict[str, Any] = {"name": runtime_name}
        if allow_missing:
            args["allowMissing"] = True
        return self.call_tool("delete_runtime", args)

    def create_session(
        self, runtime_name: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Opens a new Jupyter kernel session on an existing runtime."""
        args: Dict[str, Any] = {"parent": runtime_name}
        if session_id:
            args["sessionId"] = session_id
        return self.call_tool("create_session", args)

    def list_sessions(self, runtime_name: str) -> Dict[str, Any]:
        """Lists all active sessions on a runtime."""
        return self.call_tool("list_sessions", {"parent": runtime_name})

    def execute_code(
        self,
        code: str,
        session: Optional[str] = None,
        runtime: Optional[str] = None,
        execution_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Executes Python code on the Colab runtime or session."""
        if session and runtime:
            raise ValueError("Specify only one of `session` or `runtime`, not both.")
        if not session and not runtime:
            raise ValueError("Either `session` or `runtime` must be specified.")

        args: Dict[str, Any] = {
            "code": code,
            "executionId": execution_id or str(uuid.uuid4()),
        }
        if session:
            args["session"] = session
        else:
            args["runtime"] = runtime

        return self.call_tool("execute_code", args)

    def delete_session(self, session_name: str) -> Dict[str, Any]:
        """Terminates a Jupyter kernel session."""
        return self.call_tool("delete_session", {"name": session_name})
