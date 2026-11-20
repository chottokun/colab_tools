"""Autonomous Agent Harness for Google Colab.

Provides high-level lifecycle management, session isolation, hardware spec
eligibility checks, auto-provisioning (CPU/GPU/TPU), automatic proxy token
refresh, context-optimized code execution, and safe teardown for AI agents.
"""

import datetime
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

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
        quota_project: Optional[str] = None,
    ):
        self.client = ColabMCPClient(
            access_token=access_token, quota_project=quota_project
        )
        self.auto_teardown = auto_teardown
        self.default_accelerator = default_accelerator.upper()
        self.validate_spec = validate_spec

        self.active_runtime: Optional[str] = None
        self.active_session: Optional[str] = None
        self.runtime_connection: Optional[Dict[str, Any]] = None
        self._owns_runtime: bool = False

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
                rt_shape = spec.get("shape", "SHAPE_STANDARD")
                # shape が一致する場合のみ既存ランタイムを再利用し、メモリ不足を防ぐ
                if rt_shape == shape and (
                    (variant == "VARIANT_CPU" and rt_var == "VARIANT_CPU")
                    or (rt_var == variant and rt_acc == normalized_acc)
                ):
                    self.active_runtime = rt["name"]
                    self.runtime_connection = rt.get("connectionInfo")
                    self._owns_runtime = False
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
        self._owns_runtime = True
        return self.active_runtime

    @staticmethod
    def _is_transient_error(err: Exception) -> bool:
        """一時的なサーバー障害・レート制限・タイムアウト等のリトライ対象エラーか判定する。"""
        msg = str(err).lower()
        transient_markers = (
            "deadline_exceeded",
            "deadline exceeded",
            "unavailable",
            "resource_exhausted",
            "http 429",
            "http 502",
            "http 503",
            "http 504",
        )
        return any(marker in msg for marker in transient_markers)

    def ensure_session(self, runtime_name: Optional[str] = None) -> str:
        """エージェント用の独立した Jupyter カーネルセッションを確保する（一時障害時は同一 sessionId で再試行）。"""
        rt = runtime_name or self.active_runtime
        if not rt:
            rt = self.ensure_runtime()

        if self.active_session and self.active_session.startswith(rt):
            return self.active_session

        session_id = f"agent-{uuid.uuid4().hex[:8]}"
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                res = self.client.create_session(runtime_name=rt, session_id=session_id)
                payload = extract_tool_payload(res)
                sess_name = payload.get("name") or f"{rt}/sessions/{session_id}"
                self.active_session = sess_name
                return self.active_session
            except RuntimeError as e:
                if self._is_transient_error(e) and attempt < max_retries:
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise

        raise RuntimeError("セッション作成のリトライ回数を超過しました。")

    def execute(self, code: str, timeout_retries: int = 2) -> Dict[str, Any]:
        """Executes code in the agent's session with context optimization & resumable retry."""
        exec_id = str(uuid.uuid4())
        try:
            session = self.ensure_session()
        except (RuntimeError, ValueError) as e:
            return {
                "status": "ERROR",
                "execution_id": exec_id,
                "error": {"name": type(e).__name__, "value": str(e), "traceback": []},
                "output": "",
                "output_truncated": False,
            }

        for attempt in range(timeout_retries + 1):
            try:
                raw_res = self.client.execute_code(
                    code=code, session=session, execution_id=exec_id
                )
                return ContextOptimizer.format_execution_result(raw_res)
            except RuntimeError as e:
                if self._is_transient_error(e) and attempt < timeout_retries:
                    time.sleep(min(2 ** (attempt + 1), 8))
                    continue  # 同一 execution_id で再接続・再試行する
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

    def _transfer_with_token_retry(self, transfer_fn) -> Any:
        """プロキシトークンの期限切れ（HTTP 401/403）時に 1 回だけトークン更新して再試行する。"""
        for attempt in range(2):
            conn = self._get_connection_info(force_refresh=(attempt > 0))
            if not conn.get("url") or not conn.get("token"):
                raise RuntimeError("ファイル転送に必要なランタイム connectionInfo が取得できません。")
            try:
                return transfer_fn(conn["url"], conn["token"])
            except RuntimeError as e:
                msg = str(e)
                if attempt == 0 and ("HTTP 401" in msg or "HTTP 403" in msg):
                    continue
                raise

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

        # 1. ストリーミング形式でローカルファイルをアップロード
        if upload_files:
            for local_p, remote_p in upload_files.items():
                self._transfer_with_token_retry(
                    lambda url, tok, lp=local_p, rp=remote_p: JupyterProxy.upload_file_path(
                        url, tok, lp, rp
                    )
                )

        # 2. Execute code
        exec_result = self.execute(code)

        # 3. ストリーミング形式でリモートファイルをダウンロード
        if download_files:
            for remote_p, local_p in download_files.items():
                self._transfer_with_token_retry(
                    lambda url, tok, rp=remote_p, lp=local_p: JupyterProxy.download_file_to_path(
                        url, tok, rp, lp
                    )
                )

        return exec_result

    def teardown(
        self,
        force: bool = False,
        retries: int = 2,
        raise_on_error: bool = False,
    ) -> Dict[str, Any]:
        """セッションと（自己所有または force=True の）ランタイムを安全に解放する。"""
        errors: List[str] = []
        session_deleted = False
        runtime_deleted = False
        skipped_unowned_runtime = False

        if self.active_session:
            last_err: Optional[Exception] = None
            for attempt in range(retries + 1):
                try:
                    self.client.delete_session(self.active_session)
                    self.active_session = None
                    session_deleted = True
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if attempt < retries:
                        time.sleep(0.2)
            if last_err is not None:
                msg = f"セッション削除に失敗しました ({self.active_session}): {last_err}"
                sys.stderr.write(f"[ColabAgentHarness] 警告: {msg}\n")
                sys.stderr.flush()
                errors.append(msg)

        if self.active_runtime:
            if not (self._owns_runtime or force):
                # 他タスクと共有している既存ランタイムは誤削除せず参照のみ解除する
                skipped_unowned_runtime = True
                self.active_runtime = None
                self.runtime_connection = None
                self._owns_runtime = False
            else:
                last_err = None
                for attempt in range(retries + 1):
                    try:
                        self.client.delete_runtime(self.active_runtime, allow_missing=True)
                        self.active_runtime = None
                        self.runtime_connection = None
                        self._owns_runtime = False
                        runtime_deleted = True
                        last_err = None
                        break
                    except Exception as e:
                        last_err = e
                        if attempt < retries:
                            time.sleep(0.2)
                if last_err is not None:
                    msg = (
                        f"ランタイムの削除に失敗しました（課金継続の恐れあり） "
                        f"({self.active_runtime}): {last_err}"
                    )
                    sys.stderr.write(f"[ColabAgentHarness] 警告: {msg}\n")
                    sys.stderr.flush()
                    errors.append(msg)

        if errors and raise_on_error:
            raise RuntimeError("; ".join(errors))

        return {
            "session_deleted": session_deleted,
            "runtime_deleted": runtime_deleted,
            "skipped_unowned_runtime": skipped_unowned_runtime,
            "errors": errors,
        }

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
