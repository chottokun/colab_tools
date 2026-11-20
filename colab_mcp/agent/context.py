"""Context and output optimizer for AI agents interacting with Colab.

Handles official Google Colab MCP `ExecuteCode` structured payloads
(`stdout`, `stderr`, `result`, `executionError`, `outputTruncated`, `session`)
and prevents huge terminal outputs from overwhelming the LLM's context window.
"""

from typing import Any, Dict, Tuple

from ..mcp_client import extract_tool_payload


class ContextOptimizer:
    """Formats and trims execution outputs to maximize AI agent reasoning efficiency."""

    MAX_OUTPUT_CHARS = 4000
    HEAD_CHARS = 1500
    TAIL_CHARS = 2500

    @classmethod
    def _truncate_text(cls, text: str) -> Tuple[str, bool]:
        if len(text) <= cls.MAX_OUTPUT_CHARS:
            return text, False
        omitted = len(text) - cls.MAX_OUTPUT_CHARS
        trimmed = (
            text[: cls.HEAD_CHARS]
            + f"\n\n... [TRUNCATED {omitted} CHARACTERS TO PRESERVE AGENT CONTEXT] ...\n\n"
            + text[-cls.TAIL_CHARS :]
        )
        return trimmed, True

    @classmethod
    def format_execution_result(cls, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """Normalizes and truncates `execute_code` responses for LLM consumption."""
        payload = extract_tool_payload(raw_result)

        stdout = payload.get("stdout", "") or ""
        stderr = payload.get("stderr", "") or ""
        expr_result = payload.get("result", "") or ""
        legacy_output = payload.get("output", "") or ""

        exec_error = payload.get("executionError") or payload.get("error")
        server_truncated = bool(payload.get("outputTruncated", False))

        optimized: Dict[str, Any] = {
            "status": "ERROR" if exec_error else "SUCCESS",
            "execution_id": payload.get("executionId") or raw_result.get("execution_id"),
            "session": payload.get("session"),
            "execution_count": payload.get("executionCount"),
        }

        if exec_error and isinstance(exec_error, dict):
            tb = exec_error.get("traceback", [])
            optimized["error"] = {
                "name": exec_error.get("name") or exec_error.get("ename"),
                "value": exec_error.get("value") or exec_error.get("evalue"),
                "traceback": tb[-5:] if isinstance(tb, list) else [str(tb)],
            }

        # Combine stdout / stderr / legacy output for convenient unified reading
        parts = []
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.append(f"[stderr]\n{stderr}")
        if not parts and legacy_output:
            parts.append(legacy_output)
        if expr_result:
            parts.append(f"[result]\n{expr_result}")

        combined = "\n".join(parts)
        if combined:
            trimmed, client_truncated = cls._truncate_text(combined)
            optimized["output"] = trimmed
            optimized["output_truncated"] = server_truncated or client_truncated
        else:
            optimized["output"] = ""
            optimized["output_truncated"] = server_truncated

        if stdout:
            optimized["stdout"], _ = cls._truncate_text(stdout)
        if stderr:
            optimized["stderr"], _ = cls._truncate_text(stderr)
        if expr_result:
            optimized["result"], _ = cls._truncate_text(str(expr_result))

        return optimized
