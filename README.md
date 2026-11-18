# colab-tools: AI Agent Infrastructure for Google Colab

[![CI](https://github.com/chottokun/colab_tools/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/chottokun/colab_tools/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://pypi.org/project/colab-tools/)
[![Security: Bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Security: pip-audit](https://img.shields.io/badge/security-pip--audit-green.svg)](https://github.com/pypa/pip-audit)

A unified infrastructure enabling **AI Agents and LLMs** to autonomously leverage **Google Colab's Compute Infrastructure (GPU/TPU)** via the **Colab REST API**, **Hosted MCP Server**, and **CLI**.

---

## Why this framework for AI Agents?

When AI agents (Claude, GPT-4o, AutoGen, CrewAI, LangChain, etc.) attempt to run deep learning code on local machines, they are frequently constrained by hardware. This platform gives agents a standardized, autonomous gateway to Google Colab:

1. **Autonomous Lifecycle Management (`ColabAgentHarness`)**:
   - Automatically detects, reuses, or provisions GPU (T4 / A100 / H100) and TPU runtimes on demand.
   - Creates isolated Jupyter kernel sessions so variables and models persist across tool calls.
   - Automatic teardown guarantees cloud resources are freed upon task completion to prevent unwanted compute charges.
2. **Context Window Optimization (`ContextOptimizer`)**:
   - Truncates verbose training logs and traceback messages so they never overflow the LLM's context window.
3. **Bandwidth-Efficient File Synchronization (`JupyterProxy`)**:
   - Bypasses LLM prompt tokens: Uploads datasets directly to `/content/...` and downloads output model weights via the Colab Jupyter proxy token (`X-Colab-Runtime-Proxy-Token`).
4. **Standard LLM Tool Schemas**:
   - Ready-to-use function calling schemas for **OpenAI** and **Anthropic Claude**.
5. **Stdio MCP Proxy (`colab-cli proxy`)**:
   - Standard Model Context Protocol bridge for **Claude Desktop**, **Claude Code**, and **Cursor**.

---

## Architecture Overview

```
                  ┌──────────────────────────────────────────────┐
                  │             AI Agent / LLM Host              │
                  │ (Claude Desktop, Claude Code, Cursor, Swarm) │
                  └──────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │           colab-tools Agent Layer             │
                 │   - ColabAgentHarness (Lifecycle & Execution) │
                 │   - ContextOptimizer  (Token Protection)     │
                 │   - Tool Schemas      (OpenAI / Anthropic)   │
                 └───────┬───────────────────────────────┬───────┘
                         │                               │
        MCP (JSON-RPC)   │                               │ Direct Jupyter Proxy
        Bearer OAuth     │                               │ X-Colab-Runtime-Proxy-Token
                         ▼                               ▼
    ┌──────────────────────────────────┐   ┌───────────────────────────────┐
    │  Google Colab Hosted MCP Server  │   │  Colab Runtime Instance       │
    │  https://colaboratory.googleapis │   │  (GPU T4/A100/H100, TPU v6e)  │
    │  .com/mcp                        │   │  /content/...                 │
    └──────────────────────────────────┘   └───────────────────────────────┘
```

---

## Installation

```bash
cd colab_tools
pip install -e .
```

---

## Agent Usage (Python SDK)

### Autonomous Task Execution
```python
from colab_mcp.agent import ColabAgentHarness

# The harness automatically provisions GPU, runs code, and frees resources
with ColabAgentHarness(auto_teardown=True, default_accelerator="T4") as agent:
    result = agent.execute("""
import torch
print("CUDA Available:", torch.cuda.is_available())
""")
    print("Status:", result["status"])
    print("Output:", result["output"])
```

### Dataset Upload & Model Download
```python
with ColabAgentHarness(auto_teardown=True) as agent:
    result = agent.sync_and_run(
        code="""
import pandas as pd
df = pd.read_csv('content/dataset.csv')
summary = df.describe().to_json('content/summary.json')
""",
        upload_files={"./local_data.csv": "content/dataset.csv"},
        download_files={"content/summary.json": "./summary.json"}
    )
```

### Export Tool Definitions to OpenAI / Claude
```python
from colab_mcp.agent import to_openai_tools, to_anthropic_tools

# Pass directly to client.chat.completions.create(..., tools=openai_tools)
openai_tools = to_openai_tools()

# Pass directly to anthropic.messages.create(..., tools=claude_tools)
claude_tools = to_anthropic_tools()
```

---

## CLI for Agents & Developers (`colab-cli`)

Agents operating in a terminal (like Claude Code or Codex) can invoke `colab-cli`:

```bash
# Verify configuration
colab-cli check

# Autonomous agent one-liner
colab-cli agent run -c "import torch; print(torch.cuda.is_available())" --gpu T4 --teardown

# Export LLM tool definitions
colab-cli agent tools --format openai
colab-cli agent tools --format anthropic

# Start stdio MCP bridge (for Claude Desktop / Cursor)
colab-cli proxy
```

## Hybrid Usage with Google Official Skills (`colab-operator`)

This project is built to coexist seamlessly with **Google's Official Colab CLI & Skills ecosystem** ([`googlecolab/google-colab-cli`](https://github.com/googlecolab/google-colab-cli)).

Google provides an official skill: **`skills/colab-operator/SKILL.md`** (*Colab Session Operator*), which is also bundled directly in this repository under [`skills/colab-operator/SKILL.md`](skills/colab-operator/SKILL.md).

### When to Use Which Interface

| Interface | Best For | Description |
| :--- | :--- | :--- |
| **Google Official CLI + `colab-operator`** | **Claude Code, Codex, Antigravity CLI** | Terminal-native agents executing shell commands directly (`colab run --gpu T4 script.py`, `colab exec`). Follows official mental models and session lifecycle rules. |
| **`colab-tools` MCP Server (`colab-cli proxy`)** | **Claude Desktop, Cursor, Chat-based Agents** | Standard Model Context Protocol (MCP) clients that communicate strictly via JSON-RPC tool calls without terminal command execution. |
| **`colab-tools` Python SDK (`ColabAgentHarness`)** | **LangChain, AutoGen, CrewAI, Custom Swarms** | Python-native agent frameworks requiring programmatic auto-provisioning, context optimization, binary file sync, and robust exception retry. |

### Synergistic Agent Setup

You can equip your AI agents with both the official operational guidelines and the unified tools:

#### 1. Claude Code
Install the official skill directly or load it from this repository:
```bash
# Add official skill to Claude Code
npx skills add googlecolab/google-colab-cli --skill colab-operator
```
Claude Code will follow the official safety rules (always stopping sessions, avoiding interactive TTY commands, using `/content` paths) while utilizing `colab-tools` for headless executions.

#### 2. Claude Desktop & Cursor
Add the MCP configuration to `claude_desktop_config.json` or Cursor Settings:
```json
{
  "mcpServers": {
    "colab-tools": {
      "command": "python3",
      "args": ["-m", "colab_mcp.mcp_proxy"],
      "env": {
        "COLAB_API_TOKEN": "<YOUR_TOKEN>"
      }
    }
  }
}
```

#### 3. Python Autonomous Agents
Incorporate `ColabAgentHarness` into your agent loops to benefit from automated token truncation and cloud resource teardown:
```python
from colab_mcp.agent import ColabAgentHarness

with ColabAgentHarness(auto_teardown=True, default_accelerator="T4") as agent:
    res = agent.execute("print('Hello from Agent!')")
```

---

## Verification & Testing

Run all 16 integration, unit, and live endpoint tests:
```bash
python3 run_tests.py
```
