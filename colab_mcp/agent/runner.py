"""Autonomous Agent Harness for Google Colab.

Provides high-level lifecycle management, session isolation, hardware spec
eligibility checks, auto-provisioning (CPU/GPU/TPU), automatic proxy token
refresh, context-optimized code execution, and safe teardown for AI agents.
"""

import contextlib
import datetime
import time
import uuid
from typing import Any, Dict, Optional

from ..jupyter_proxy import JupyterProxy
from ..mcp_client import ColabMCPClient, extract_tool_payload
from .context import ContextOptimizer


class ColabAgentHarness:
    """High-level management harness designed specifically for autonomous AI agents."""

    TOKEN_REFRESH_MARGIN_SECONDS = 300  # Refresh proxy token 5 minutes before expiry

    def __init__(
        self,
        access_token: Optional[str] = None,
        auto_teardown: bool = False,
        default_accelerator: str = "T4",
        validate_spec: bool = False,
    ):
        self.client = ColabMCPClient(access_token=access_token)
        self.auto_teardown = auto_teardown
        self.default_accelerator = default_accelerator.upper()
        self.validate_spec = validate_spec

        self.active_runtime: Optional[str] = None
        self.active_session: Optional[str] = None
        self.runtime_connection: Optional[Dict[str, Any]] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.auto_teardown:
            self.teardown()

    @staticmethod
    def _resolve_variant(accelerator: str) -> str:
        acc = accelerator.upper()
        if acc in ("NONE", "CPU", ""):
            return "VARIANT_CPU"
        if acc.startswith("V") or "TPU" in acc:
            return "VARIANT_TPU"
        return "VARIANT_GPU"

    def _validate_runtime_spec(
        self, variant: str, accelerator: str, shape: str
    ) -> None:
        """Checks eligibility against `list_runtime_specs` before provisioning."""
        specs_raw = self.client.list_runtime_specs()
        payload = extract_tool_payload(specs_raw)
        specs = payload.get("runtimeSpecs", [])
        if not specs:
            return

        eligible_specs = []
        for entry in specs:
            if not entry.get("eligible"):
                continue
            key = entry.get("key", {})
            eligible_specs.append(key)
            if (
                key.get("variant") == variant
                and key.get("accelerator", "").upper() == accelerator.upper()
                and key.get("shape") == shape
            ):
                return

        available_desc = ", ".join(
            f"{k.get('variant')}/{k.get('accelerator')}/{k.get('shape')}"
            for k in eligible_specs
        )
        raise ValueError(
            f"Requested runtime spec ({variant}/{accelerator}/{shape}) is not eligible "
            f"for your subscription. Eligible specs: [{available_desc}]"
        )

    def ensure_runtime(
        self,
        accelerator: Optional[str] = None,
        shape: str = "SHAPE_STANDARD",
        reuse_existing: bool = True,
    ) -> str:
        """Finds an existing matching runtime or provisions a new one."""
        target_acc = (accelerator or self.default_accelerator).upper()
        variant = self._resolve_variant(target_acc)
        normalized_acc = "NONE" if variant == "VARIANT_CPU" else target_acc

        if reuse_existing:
            runtimes_resp = self.client.list_runtimes()
            payload = extract_tool_payload(runtimes_resp)
            runtimes = payload.get("runtimes", [])
            for rt in runtimes:
                spec = rt.get("runtimeSpec", {})
                rt_acc = spec.get("accelerator", "").upper()
                rt_var = spec.get("variant", "")
                if (variant == "VARIANT_CPU" and rt_var == "VARIANT_CPU") or (
                    rt_var == variant and rt_acc == normalized_acc
                ):
                    self.active_runtime = rt["name"]
                    self.runtime_connection = rt.get("connectionInfo")
                    return self.active_runtime

        if self.validate_spec:
            self._validate_runtime_spec(variant, normalized_acc, shape)

        spec = {
            "variant": variant,
            "accelerator": normalized_acc,
            "shape": shape,
        }
        res = self.client.create_runtime(spec=spec)
        rt_data = extract_tool_payload(res)
        if "runtime" in rt_data and isinstance(rt_data["runtime"], dict):
            rt_data = rt_data["runtime"]

        self.active_runtime = rt_data.get("name")
        self.runtime_connection = rt_data.get("connectionInfo")
        if not self.active_runtime:
            raise RuntimeError(f"Failed to obtain runtime name from response: {res}")
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
        payload = extract_tool_payload(res)
        sess_name = payload.get("name") or f"{rt}/sessions/{session_id}"
        self.active_session = sess_name
        return self.active_session

    def execute(self, code: str, timeout_retries: int = 2) -> Dict[str, Any]:
        """Executes code in the agent's session with context optimization & resumable retry."""
        session = self.ensure_session()
        exec_id = str(uuid.uuid4())

        for attempt in range(timeout_retries + 1):
            try:
                raw_res = self.client.execute_code(
                    code=code, session=session, execution_id=exec_id
                )
                return ContextOptimizer.format_execution_result(raw_res)
            except RuntimeError as e:
                if "DEADLINE_EXCEEDED" in str(e) or "UNAVAILABLE" in str(e):
                    if attempt < timeout_retries:
                        time.sleep(2)
                        continue  # Repeat identical call with same exec_id to reattach
                return {
                    "status": "ERROR",
                    "execution_id": exec_id,
                    "error": {"name": "RuntimeError", "value": str(e), "traceback": []},
                    "output": "",
                    "output_truncated": False,
                }
        return {
            "status": "ERROR",
            "execution_id": exec_id,
            "error": {"name": "TimeoutError", "value": "Execution retries exhausted.", "traceback": []},
            "output": "",
            "output_truncated": False,
        }

    def sync_and_run(
        self,
        code: str,
        upload_files: Optional[Dict[str, str]] = None,
        download_files: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """End-to-end agent task: upload dependencies, run code, and retrieve outputs.

        Args:
            code: Python code to run
            upload_files: {local_path: remote_path_in_content}
            download_files: {remote_path_in_content: local_path}
        """
        self.ensure_runtime()

        # 1. Upload files
        if upload_files:
            conn = self._get_connection_info()
            if not conn.get("url") or not conn.get("token"):
                raise RuntimeError("Missing runtime connectionInfo for file upload.")
            for local_p, remote_p in upload_files.items():
                with open(local_p, "rb") as f:
                    JupyterProxy.upload_file(conn["url"], conn["token"], remote_p, f.read())

        # 2. Execute code
        exec_result = self.execute(code)

        # 3. Download files (re-check connectionInfo in case token expired during long execution)
        if download_files:
            conn = self._get_connection_info()
            if not conn.get("url") or not conn.get("token"):
                raise RuntimeError("Missing runtime connectionInfo for file download.")
            for remote_p, local_p in download_files.items():
                data = JupyterProxy.download_file(conn["url"], conn["token"], remote_p)
                with open(local_p, "wb") as f:
                    f.write(data)

        return exec_result

    def teardown(self) -> None:
        """Cleans up session and runtime to release cloud compute resources."""
        if self.active_session:
            with contextlib.suppress(Exception):
                self.client.delete_session(self.active_session)
            self.active_session = None

        if self.active_runtime:
            with contextlib.suppress(Exception):
                self.client.delete_runtime(self.active_runtime, allow_missing=True)
            self.active_runtime = None
            self.runtime_connection = None

    @classmethod
    def _is_connection_expired(cls, conn: Optional[Dict[str, Any]]) -> bool:
        if not conn or not conn.get("url") or not conn.get("token"):
            return True
        expire_str = conn.get("expireTime")
        if not expire_str:
            return False
        try:
            normalized = expire_str.replace("Z", "+00:00")
            expire_dt = datetime.datetime.fromisoformat(normalized)
            if expire_dt.tzinfo is None:
                expire_dt = expire_dt.replace(tzinfo=datetime.timezone.utc)
            now = datetime.datetime.now(datetime.timezone.utc)
            return (expire_dt - now).total_seconds() <= cls.TOKEN_REFRESH_MARGIN_SECONDS
        except ValueError:
            return False

    def _get_connection_info(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Returns valid runtime connectionInfo, refreshing via get_runtime if near expiry."""
        needs_refresh = force_refresh or self._is_connection_expired(self.runtime_connection)
        if needs_refresh and self.active_runtime:
            info = self.client.get_runtime(self.active_runtime)
            rt = extract_tool_payload(info)
            if "runtime" in rt and isinstance(rt["runtime"], dict):
                rt = rt["runtime"]
            self.runtime_connection = rt.get("connectionInfo", {})
        return self.runtime_connection or {}
