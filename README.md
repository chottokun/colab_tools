# colab-with-mcp

Unified Google Colab toolkit integrating the **Colab REST API**, **Hosted MCP Server**, and **Command-Line Interface (CLI)**.

## Architecture & Features

- **Colab REST API Client (`colab_mcp.ColabRESTClient`)**:
  - Direct HTTP client for Colab v1beta endpoints.
  - Manages runtimes and runtime specifications.
- **Hosted MCP Client (`colab_mcp.ColabMCPClient`)**:
  - Interacts directly with Google's hosted MCP endpoint (`https://colaboratory.googleapis.com/mcp`).
  - Supports all 10 MCP tools: runtime provisioning, session management, and remote Python execution.
- **stdio MCP Proxy (`colab_mcp.mcp_proxy`)**:
  - Stdio bridge allowing Claude Desktop, Claude Code, Cursor, and other AI agents to connect to Colab.
- **Jupyter Proxy File Transfer (`colab_mcp.JupyterProxy`)**:
  - Efficient file uploads and downloads using `X-Colab-Runtime-Proxy-Token`.
- **Command-Line Interface (`colab-cli` / `colab-mcp`)**:
  - Full terminal control over subscriptions, GPU runtimes, code execution, and data sync.

## Installation

```bash
cd colab_with_mcp
pip install -e .
```

## CLI Usage

### Configuration
```bash
# Check setup
colab-cli check

# Set access token
export COLAB_API_TOKEN=$(gcloud auth print-access-token)
```

### Runtimes & Hardware
```bash
# View subscription tier
colab-cli subscription

# List available specs (GPU/TPU)
colab-cli specs

# Create a GPU runtime (T4 standard)
colab-cli runtimes create --gpu T4 --shape standard

# List running instances
colab-cli runtimes list

# Delete a runtime
colab-cli runtimes delete runtimes/<RUNTIME_ID>
```

### Sessions & Code Execution
```bash
# Start a Jupyter session
colab-cli sessions create runtimes/<RUNTIME_ID>

# Execute one-liner
colab-cli exec --session runtimes/<RUNTIME_ID>/sessions/<SESSION_ID> -c "import torch; print(torch.cuda.is_available())"

# Execute a Python file
colab-cli exec --session runtimes/<RUNTIME_ID>/sessions/<SESSION_ID> -f train.py
```

### File Transfers
```bash
# Upload local dataset to Colab runtime
colab-cli files upload runtimes/<RUNTIME_ID> ./dataset.csv content/dataset.csv

# Download results
colab-cli files download runtimes/<RUNTIME_ID> content/results.json ./results.json
```

### Running MCP Stdio Proxy (for Claude / Cursor)
```bash
colab-cli proxy
```

## Testing

Run the test suite (includes live tests against the official endpoint):
```bash
python3 run_tests.py
```
