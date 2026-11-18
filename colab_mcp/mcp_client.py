"""Client for interacting with the Google Colab hosted MCP Endpoint.
Protocol: Model Context Protocol (JSON-RPC 2.0 over HTTP/Stream)
Endpoint: https://colaboratory.googleapis.com/mcp
"""

import json
import urllib.request
import urllib.error
import uuid
from typing import Any, Dict, List, Optional

from .config import ColabConfig


class ColabMCPClient:
    """Client for Google Colab hosted MCP tools."""

    def __init__(self, access_token: Optional[str] = None, endpoint: Optional[str] = None):
        self.access_token = access_token or ColabConfig.get_token()
        self.endpoint = endpoint or ColabConfig.DEFAULT_MCP_ENDPOINT

    def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calls an MCP tool by name with arguments."""
        if not self.access_token:
            raise ValueError(
                "Access token is required. Set COLAB_API_TOKEN environment variable "
                "or pass access_token to ColabMCPClient."
            )

        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {}
            }
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {self.access_token}"
        }

        if not self.endpoint.startswith(("https://", "http://")):
            raise ValueError(f"Invalid endpoint scheme: {self.endpoint}")

        req = urllib.request.Request(self.endpoint, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req) as resp:  # nosec B310
                raw_body = resp.read().decode("utf-8")
                return json.loads(raw_body)
        except urllib.error.HTTPError as e:
            error_content = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Colab MCP call failed [HTTP {e.code}]: {error_content}") from e

    def list_tools(self) -> List[Dict[str, Any]]:
        """Lists available MCP tools and their schemas from the server."""
        if not self.endpoint.startswith(("https://", "http://")):
            raise ValueError(f"Invalid endpoint scheme: {self.endpoint}")

        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/list"
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req) as resp:  # nosec B310
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("result", {}).get("tools", [])

    # High-level tool methods

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

    def create_runtime(self, spec: Dict[str, Any], runtime_id: Optional[str] = None) -> Dict[str, Any]:
        """Provisions a new runtime assignment."""
        args: Dict[str, Any] = {"runtime": {"spec": spec}}
        if runtime_id:
            args["runtime_id"] = runtime_id
        return self.call_tool("create_runtime", args)

    def delete_runtime(self, runtime_name: str) -> Dict[str, Any]:
        """Terminates and deletes a Colab runtime."""
        return self.call_tool("delete_runtime", {"name": runtime_name})

    def create_session(self, runtime_name: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Opens a new Jupyter kernel session on an existing runtime."""
        args: Dict[str, Any] = {"runtime": runtime_name}
        if session_id:
            args["session_id"] = session_id
        return self.call_tool("create_session", args)

    def list_sessions(self, runtime_name: str) -> Dict[str, Any]:
        """Lists all active sessions on a runtime."""
        return self.call_tool("list_sessions", {"runtime": runtime_name})

    def execute_code(
        self,
        code: str,
        session: Optional[str] = None,
        runtime: Optional[str] = None,
        execution_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executes Python code on the Colab runtime or session."""
        args: Dict[str, Any] = {
            "code": code,
            "execution_id": execution_id or str(uuid.uuid4())
        }
        if session:
            args["session"] = session
        elif runtime:
            args["runtime"] = runtime
        else:
            raise ValueError("Either session or runtime must be specified.")

        return self.call_tool("execute_code", args)

    def delete_session(self, session_name: str) -> Dict[str, Any]:
        """Terminates a Jupyter kernel session."""
        return self.call_tool("delete_session", {"name": session_name})
