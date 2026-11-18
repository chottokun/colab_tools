"""Autonomous Agent Harness for Google Colab.

Provides high-level lifecycle management, session isolation, auto-provisioning,
context-optimized code execution, and safe teardown for AI agents.
"""

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from ..config import ColabConfig
from ..mcp_client import ColabMCPClient
from ..jupyter_proxy import JupyterProxy
from .context import ContextOptimizer


class ColabAgentHarness:
    """High-level management harness designed specifically for autonomous AI agents."""

    def __init__(
        self,
        access_token: Optional[str] = None,
        auto_teardown: bool = False,
        default_accelerator: str = "T4"
    ):
        self.client = ColabMCPClient(access_token=access_token)
        self.auto_teardown = auto_teardown
        self.default_accelerator = default_accelerator.upper()

        self.active_runtime: Optional[str] = None
        self.active_session: Optional[str] = None
        self.runtime_connection: Optional[Dict[str, Any]] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.auto_teardown:
            self.teardown()

    def ensure_runtime(
        self,
        accelerator: Optional[str] = None,
        shape: str = "SHAPE_STANDARD",
        reuse_existing: bool = True
    ) -> str:
        """Finds an existing matching runtime or provisions a new one."""
        target_acc = (accelerator or self.default_accelerator).upper()

        if reuse_existing:
            runtimes_resp = self.client.list_runtimes()
            runtimes = runtimes_resp.get("result", {}).get("runtimes", [])
            for rt in runtimes:
                spec = rt.get("runtimeSpec", {})
                rt_acc = spec.get("accelerator", "").upper()
                if rt_acc == target_acc or (target_acc in ("NONE", "CPU") and spec.get("variant") == "VARIANT_CPU"):
                    self.active_runtime = rt["name"]
                    self.runtime_connection = rt.get("connectionInfo")
                    return self.active_runtime

        # Create new runtime
        variant = "VARIANT_CPU" if target_acc in ("NONE", "CPU") else "VARIANT_GPU"
        spec = {
            "variant": variant,
            "accelerator": "NONE" if variant == "VARIANT_CPU" else target_acc,
            "shape": shape
        }
        res = self.client.create_runtime(spec=spec)
        rt_data = res.get("result", {}).get("runtime") or res.get("result", {})
        self.active_runtime = rt_data.get("name")
        self.runtime_connection = rt_data.get("connectionInfo")
        return self.active_runtime

    def ensure_session(self, runtime_name: Optional[str] = None) -> str:
        """Ensures an active isolated Jupyter session exists for the agent."""
        rt = runtime_name or self.active_runtime
        if not rt:
            rt = self.ensure_runtime()

        if self.active_session and self.active_session.startswith(rt):
            return self.active_session

        session_id = f"agent-{uuid.uuid4().hex[:8]}"
        res = self.client.create_session(runtime_name=rt, session_id=session_id)
        sess_name = res.get("result", {}).get("name") or f"{rt}/sessions/{session_id}"
        self.active_session = sess_name
        return self.active_session

    def execute(self, code: str, timeout_retries: int = 2) -> Dict[str, Any]:
        """Executes code in the agent's session with context optimization & retry logic."""
        session = self.ensure_session()
        exec_id = str(uuid.uuid4())

        for attempt in range(timeout_retries + 1):
            try:
                raw_res = self.client.execute_code(code=code, session=session, execution_id=exec_id)
                return ContextOptimizer.format_execution_result(raw_res)
            except RuntimeError as e:
                if "DEADLINE_EXCEEDED" in str(e) or "UNAVAILABLE" in str(e):
                    if attempt < timeout_retries:
                        time.sleep(2)
                        continue  # Repeat identical call with same exec_id to reattach
                return {
                    "status": "ERROR",
                    "error": {"evalue": str(e)},
                    "output_truncated": False
                }

    def sync_and_run(
        self,
        code: str,
        upload_files: Optional[Dict[str, str]] = None,
        download_files: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """End-to-end agent task: upload dependencies, run code, and retrieve outputs.
        
        Args:
            code: Python code to run
            upload_files: {local_path: remote_path_in_content}
            download_files: {remote_path_in_content: local_path}
        """
        self.ensure_runtime()
        conn = self._get_connection_info()

        # 1. Upload files
        if upload_files and conn.get("url") and conn.get("token"):
            for local_p, remote_p in upload_files.items():
                with open(local_p, "rb") as f:
                    JupyterProxy.upload_file(conn["url"], conn["token"], remote_p, f.read())

        # 2. Execute code
        exec_result = self.execute(code)

        # 3. Download files
        if download_files and conn.get("url") and conn.get("token"):
            for remote_p, local_p in download_files.items():
                data = JupyterProxy.download_file(conn["url"], conn["token"], remote_p)
                with open(local_p, "wb") as f:
                    f.write(data)

        return exec_result

    def teardown(self):
        """Cleans up session and runtime to release cloud compute resources."""
        if self.active_session:
            try:
                self.client.delete_session(self.active_session)
            except Exception:
                pass
            self.active_session = None

        if self.active_runtime:
            try:
                self.client.delete_runtime(self.active_runtime)
            except Exception:
                pass
            self.active_runtime = None

    def _get_connection_info(self) -> Dict[str, Any]:
        if not self.runtime_connection and self.active_runtime:
            info = self.client.get_runtime(self.active_runtime)
            rt = info.get("result", {}).get("runtime") or info.get("result") or {}
            self.runtime_connection = rt.get("connectionInfo", {})
        return self.runtime_connection or {}
