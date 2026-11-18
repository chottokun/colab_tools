"""Context and output optimizer for AI agents interacting with Colab.

Prevents huge terminal outputs or dataset dumps from overwhelming the LLM's context window.
"""

from typing import Dict, Any


class ContextOptimizer:
    """Formats and trims outputs to maximize AI agent reasoning efficiency."""

    MAX_OUTPUT_CHARS = 4000
    HEAD_CHARS = 1500
    TAIL_CHARS = 2500

    @classmethod
    def format_execution_result(cls, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """Summarizes and truncates code execution outputs for LLM consumption."""
        result = raw_result.get("result", {})
        output = result.get("output", "")
        error = result.get("error", None)

        optimized = {
            "status": "ERROR" if error else "SUCCESS",
            "execution_id": raw_result.get("execution_id"),
        }

        if error:
            optimized["error"] = {
                "ename": error.get("ename"),
                "evalue": error.get("evalue"),
                "traceback": error.get("traceback", [])[-5:]  # Keep last 5 lines of traceback
            }

        if output:
            if len(output) > cls.MAX_OUTPUT_CHARS:
                trimmed = (
                    output[:cls.HEAD_CHARS]
                    + f"\n\n... [TRUNCATED {len(output) - cls.MAX_OUTPUT_CHARS} CHARACTERS TO PRESERVE AGENT CONTEXT] ...\n\n"
                    + output[-cls.TAIL_CHARS:]
                )
                optimized["output"] = trimmed
                optimized["output_truncated"] = True
            else:
                optimized["output"] = output
                optimized["output_truncated"] = False

        return optimized
