# colab-with-mcp: AI Agent Infrastructure for Google Colab

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
                 │          colab-with-mcp Agent Layer           │
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
cd colab_with_mcp
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

---

## Verification & Testing

Run all 16 integration, unit, and live endpoint tests:
```bash
python3 run_tests.py
```
