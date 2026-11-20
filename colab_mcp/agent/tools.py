"""LLM Tool Calling schemas and adapters for AI Agents.

Provides schemas compatible with:
- OpenAI Function Calling
- Anthropic Claude Tool Use
- LangChain / LlamaIndex / CrewAI
"""

from typing import Any, Dict, List

SUPPORTED_ACCELERATORS = ["T4", "L4", "G4", "A100", "H100", "V5E1", "V6E1", "NONE"]


def get_agent_tool_definitions() -> List[Dict[str, Any]]:
    """Returns canonical definitions of Colab tools exposed to AI agents."""
    return [
        {
            "name": "colab_run_code",
            "description": (
                "Executes Python code on Google Colab (CPU/GPU/TPU) and returns stdout, "
                "stderr, expression result, or error. Kernel state is maintained across calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute on Colab.",
                    },
                    "accelerator": {
                        "type": "string",
                        "enum": SUPPORTED_ACCELERATORS,
                        "default": "T4",
                        "description": (
                            "Hardware accelerator required (e.g. T4/L4/G4 for general ML, "
                            "A100/H100 for heavy LLM training, V5E1/V6E1 for TPU, NONE for CPU)."
                        ),
                    },
                    "shape": {
                        "type": "string",
                        "enum": ["SHAPE_STANDARD", "SHAPE_HIGHMEM"],
                        "default": "SHAPE_STANDARD",
                        "description": "System RAM shape.",
                    },
                },
                "required": ["code"],
            },
        },
        {
            "name": "colab_sync_and_run",
            "description": (
                "Uploads local files/datasets directly to Colab (bypassing LLM tokens), "
                "runs a Python script on GPU/TPU, and downloads output models/artifacts back to local disk."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute.",
                    },
                    "upload_files": {
                        "type": "object",
                        "description": "Key-value mapping of {local_path: remote_path_in_colab (e.g. content/data.csv)}.",
                    },
                    "download_files": {
                        "type": "object",
                        "description": "Key-value mapping of {remote_path_in_colab: local_path}.",
                    },
                    "accelerator": {
                        "type": "string",
                        "enum": SUPPORTED_ACCELERATORS,
                        "default": "T4",
                        "description": "Hardware accelerator.",
                    },
                },
                "required": ["code"],
            },
        },
        {
            "name": "colab_list_hardware_specs",
            "description": "Lists available CPU/GPU/TPU machine types and eligibility for the current Colab subscription.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "colab_list_runtimes",
            "description": "Lists currently active Colab runtimes so existing compute can be reused before allocating new ones.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "colab_teardown",
            "description": "Terminates active Colab runtimes and sessions to stop cloud compute consumption.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    ]


def to_openai_tools() -> List[Dict[str, Any]]:
    """Exports tools formatted for OpenAI / ChatGPT Function Calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in get_agent_tool_definitions()
    ]


def to_anthropic_tools() -> List[Dict[str, Any]]:
    """Exports tools formatted for Anthropic Claude Tool Use."""
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["parameters"],
        }
        for tool in get_agent_tool_definitions()
    ]
