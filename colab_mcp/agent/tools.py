"""LLM Tool Calling schemas and adapters for AI Agents.

Provides schemas compatible with:
- OpenAI Function Calling
- Anthropic Claude Tool Use
- LangChain / LlamaIndex / CrewAI
"""

from typing import Any, Dict, List


def get_agent_tool_definitions() -> List[Dict[str, Any]]:
    """Returns canonical definitions of Colab tools exposed to AI agents."""
    return [
        {
            "name": "colab_run_code",
            "description": "Executes Python code on Google Colab (GPU/TPU) and returns the output or error. State is maintained across calls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute on Colab."
                    },
                    "accelerator": {
                        "type": "string",
                        "enum": ["T4", "A100", "H100", "TPU", "NONE"],
                        "default": "T4",
                        "description": "Hardware accelerator required (e.g. T4 for general ML, A100/H100 for heavy LLM training, NONE for CPU)."
                    }
                },
                "required": ["code"]
            }
        },
        {
            "name": "colab_sync_and_run",
            "description": "Uploads local files/datasets to Colab, runs a Python script on GPU, and downloads output models/artifacts back to local disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute."
                    },
                    "upload_files": {
                        "type": "object",
                        "description": "Key-value mapping of {local_path: remote_path_in_colab}."
                    },
                    "download_files": {
                        "type": "object",
                        "description": "Key-value mapping of {remote_path_in_colab: local_path}."
                    },
                    "accelerator": {
                        "type": "string",
                        "default": "T4",
                        "description": "Hardware accelerator."
                    }
                },
                "required": ["code"]
            }
        },
        {
            "name": "colab_list_hardware_specs",
            "description": "Lists available GPU/TPU machine types and eligibility for the current Colab subscription.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "colab_teardown",
            "description": "Terminates active Colab GPU runtimes and sessions to stop cloud compute consumption.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    ]


def to_openai_tools() -> List[Dict[str, Any]]:
    """Exports tools formatted for OpenAI / ChatGPT Function Calling."""
    tools = []
    for tool in get_agent_tool_definitions():
        tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
        })
    return tools


def to_anthropic_tools() -> List[Dict[str, Any]]:
    """Exports tools formatted for Anthropic Claude Tool Use."""
    tools = []
    for tool in get_agent_tool_definitions():
        tools.append({
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["parameters"]
        })
    return tools
