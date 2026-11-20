English | [日本語](README.ja.md)

# colab-tools

[![CI](https://github.com/chottokun/colab_tools/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/chottokun/colab_tools/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
[![Security: Bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Security: pip-audit](https://img.shields.io/badge/security-pip--audit-green.svg)](https://github.com/pypa/pip-audit)

Standard-library Python client, runtime proxy, stdio MCP bridge, and CLI for interacting with the Google Colab Hosted MCP endpoint (`https://colaboratory.googleapis.com/mcp`), REST API (`v1beta` / `v1/operations`), and Jupyter runtime proxy. Requires no third-party runtime dependencies.

---

## Overview

`colab-tools` allows scripts, CLI workflows, and LLM tool-calling agents to allocate Colab runtimes, run Python code in Jupyter sessions, and transfer files directly over the runtime proxy—including in headless environments without a local browser.

```
  Client / Agent / MCP Host
             │
             ├──► colab_mcp.mcp_client / colab_mcp.api / colab_mcp.mcp_proxy
             │         │  HTTPS (Bearer OAuth + x-goog-user-project)
             │         ▼
             │    https://colaboratory.googleapis.com/mcp
             │    https://colaboratory.googleapis.com/v1beta & /v1/operations
             │
             └──► colab_mcp.jupyter_proxy
                       │  HTTPS (X-Colab-Runtime-Proxy-Token)
                       ▼
                  Colab Runtime (/api/contents, /files)
```

---

## Components

- **`ColabMCPClient` (`colab_mcp/mcp_client.py`)**: JSON-RPC 2.0 client over Streamable HTTP (JSON and SSE). Adds `x-goog-user-project` when configured and retries once on HTTP 401 by refreshing ADC credentials.
- **`ColabRESTClient` (`colab_mcp/api.py`)**: REST client for `/v1beta` (`subscription`, `runtimespecs`, `runtimes`) and `/v1/operations` polling (`list_operations`, `get_operation`, `wait_operation`, `create_runtime(..., wait=True)`, `delete_runtime(..., wait=True)`).
- **`JupyterProxy` (`colab_mcp/jupyter_proxy.py`)**: Direct file transfer via `upload_file_path` / `download_file_to_path` (streaming I/O) and `upload_file` / `download_file`. Supports percent-encoded remote paths, automatic parent directory creation, 30 MB chunked uploads, and HTTP `Range` downloads.
- **`ColabAgentHarness` & `ContextOptimizer` (`colab_mcp/agent/`)**: Runtime lifecycle helper supporting CPU (`NONE`), GPU (`T4`, `L4`, `G4`, `A100`, `H100`), and TPU (`V5E1`, `V6E1`). Checks spec eligibility (`validate_spec=True`), matches `shape` before reusing runtimes, tracks runtime ownership (`_owns_runtime`) to avoid deleting shared runtimes on `teardown()`, retries transient failures (`unavailable`, `deadline_exceeded`, `resource_exhausted`, HTTP 429/5xx) with backoff, refreshes expired proxy tokens, and truncates long outputs.
- **Tool Schemas & Dispatcher (`colab_mcp/agent/tools.py`)**: `to_openai_tools()`, `to_anthropic_tools()`, and `execute_agent_tool()`.
- **Stdio MCP Proxy (`colab_mcp/mcp_proxy.py`)**: Forwards JSON-RPC 2.0 over stdin/stdout to `https://colaboratory.googleapis.com/mcp`, skips token lookup for unauthenticated methods (`initialize`, `notifications/initialized`, `tools/list`, `ping`), forwards `Mcp-Session-Id`, retries once on HTTP 401, and emits JSON-RPC `-32700` on invalid JSON input.
- **CLI (`colab_mcp/cli.py`)**: `colab-cli` / `python3 -m colab_mcp.cli`. Redacts sensitive fields (`token`, `access_token`, `refresh_token`, `client_secret`) by default (`--show-secrets` overrides) and supports `--cpu`, `--gpu`, and `--tpu`.

---

## Installation & Authentication

### Installation

```bash
git clone https://github.com/chottokun/colab_tools.git
cd colab_tools
pip install -e .
```

### Enabling the Colab API and Quota Project

Calling `colaboratory.googleapis.com` with user OAuth or ADC credentials requires an enabled Google Cloud project sent via the `x-goog-user-project` header:

```bash
gcloud services enable colaboratory.googleapis.com --project <YOUR_GCP_PROJECT_ID>
export COLAB_QUOTA_PROJECT="<YOUR_GCP_PROJECT_ID>"
```

### Credential Resolution Order

`ColabConfig` (`colab_mcp/config.py`) resolves credentials in the following order:

1. Explicit `access_token` argument passed to a client or harness.
2. Environment variables (`COLAB_API_TOKEN`, `GOOGLE_OAUTH_TOKEN`, `GCLOUD_AUTH_TOKEN`) or `.env` / `.env.colab` / `~/.env.colab`.
3. `gcloud` CLI (`gcloud auth application-default print-access-token`).
4. ADC JSON file (`GOOGLE_APPLICATION_CREDENTIALS` or `~/.config/gcloud/application_default_credentials.json`) with `type == "authorized_user"`, using its `refresh_token` directly over HTTPS (`adc:json`).

ADC tokens are cached in memory until 120 seconds before expiry. If an HTTP 401 response occurs, `ColabConfig` invalidates any expired environment token for the process and switches to refreshed ADC credentials.

### Interactive `gcloud` Setup

```bash
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory

gcloud auth application-default set-quota-project <YOUR_GCP_PROJECT_ID>

colab-cli check
```

### Headless Operation (No Browser / No `gcloud` CLI)

- Avoid storing short-lived `COLAB_API_TOKEN` values (~1 hour TTL) in `.env`.
- Place an `application_default_credentials.json` file (containing `refresh_token` and `quota_project_id`, obtained from another machine) at `~/.config/gcloud/application_default_credentials.json` with `chmod 600`, and set only `COLAB_QUOTA_PROJECT` in `.env` or environment variables.
- `ColabConfig` refreshes access tokens directly against `token_uri` without requiring `gcloud` on the host.

### Environment Variables

| Variable | Purpose | Notes |
| :--- | :--- | :--- |
| `COLAB_QUOTA_PROJECT` | GCP quota project ID | Falls back to `GOOGLE_CLOUD_QUOTA_PROJECT`, `GOOGLE_CLOUD_PROJECT`, `GCLOUD_PROJECT`, or `quota_project_id` in ADC JSON |
| `COLAB_API_TOKEN` | Explicit OAuth access token | Auto-invalidated in-process on 401 fallback (`GOOGLE_OAUTH_TOKEN`, `GCLOUD_AUTH_TOKEN` also checked) |
| `GOOGLE_APPLICATION_CREDENTIALS` | Custom ADC JSON path | Defaults to `~/.config/gcloud/application_default_credentials.json` |
| `COLAB_IGNORE_CWD_DOTENV` | Skip `.env` in working directory | Set to `1` or `true` to load only `~/.env.colab` |
| `COLAB_SKIP_LIVE_TESTS` | Skip live endpoint tests | Set to `1` or `true` to skip `test_live_mcp.py` in `run_tests.py` |

---

## Usage

### Python SDK

```python
from colab_mcp.agent import ColabAgentHarness

with ColabAgentHarness(
    auto_teardown=True,
    default_accelerator="T4",
    validate_spec=True,
) as agent:
    result = agent.execute("import torch; print(torch.cuda.is_available())")
    print(result["status"], result["output"])
```

#### File Sync & Execution

```python
with ColabAgentHarness(auto_teardown=True, default_accelerator="NONE") as agent:
    result = agent.sync_and_run(
        code="""
import pandas as pd
df = pd.read_csv('content/dataset.csv')
df.describe().to_json('content/summary.json')
""",
        upload_files={"./local_data.csv": "content/dataset.csv"},
        download_files={"content/summary.json": "./summary.json"},
    )
```

#### REST API with LRO Polling

```python
from colab_mcp import ColabRESTClient

rest = ColabRESTClient()
rt = rest.create_runtime(
    {"variant": "VARIANT_CPU", "accelerator": "NONE", "shape": "SHAPE_STANDARD"},
    wait=True,
)
try:
    print(rt["name"])
finally:
    rest.delete_runtime(rt["name"], allow_missing=True, wait=True)
```

#### LLM Tool Definitions & Dispatch

```python
from colab_mcp.agent import (
    ColabAgentHarness,
    execute_agent_tool,
    to_anthropic_tools,
    to_openai_tools,
)

openai_tools = to_openai_tools()
claude_tools = to_anthropic_tools()

with ColabAgentHarness(auto_teardown=True) as harness:
    res = execute_agent_tool(
        harness,
        "colab_run_code",
        {"code": "print(1 + 1)", "accelerator": "NONE"},
    )
```

### CLI (`colab-cli` / `python3 -m colab_mcp.cli`)

```bash
# Check configuration
colab-cli check

# Account & hardware specs
colab-cli subscription
colab-cli specs

# Runtimes (connectionInfo.token is masked as ***REDACTED*** by default)
colab-cli runtimes list
colab-cli runtimes create --gpu T4 --shape standard
colab-cli runtimes get <RUNTIME_NAME>          # pass --show-secrets to print raw tokens
colab-cli runtimes delete <RUNTIME_NAME>

# Sessions & execution
colab-cli sessions create <RUNTIME_NAME>
colab-cli sessions list <RUNTIME_NAME>
colab-cli exec --session <SESSION_NAME> -c "print('hello')"
colab-cli sessions delete <SESSION_NAME>

# Streaming file transfer
colab-cli files upload <RUNTIME_NAME> ./local.bin content/remote.bin
colab-cli files download <RUNTIME_NAME> content/remote.bin ./downloaded.bin

# Agent one-liner & schemas
colab-cli agent run --cpu --validate-spec --teardown -c "print('CPU OK')"
colab-cli agent run --gpu T4 --teardown -f train.py
colab-cli agent tools --format openai
colab-cli agent tools --format anthropic

# Stdio MCP proxy
colab-cli proxy
```

### MCP Client Configuration

Example configuration (`examples/mcp_config.json`) using ADC auto-refresh with only `COLAB_QUOTA_PROJECT`:

```json
{
  "mcpServers": {
    "colab": {
      "command": "python3",
      "args": ["-m", "colab_mcp.mcp_proxy"],
      "env": {
        "COLAB_QUOTA_PROJECT": "<YOUR_GCP_PROJECT_ID>"
      }
    }
  }
}
```

### Bundled `colab-operator` Skill

This repository also bundles [`skills/colab-operator/SKILL.md`](skills/colab-operator/SKILL.md), an abridged reference derived from [`googlecolab/google-colab-cli`](https://github.com/googlecolab/google-colab-cli) for terminal agents using the official `colab` CLI.

---

## Security

- **URL validation (`ColabConfig.validate_url`)**: Allows only `https://` URLs and `http://` on loopback hosts (`localhost`, `127.0.0.1`, `::1`).
- **Secret redaction (`ColabConfig.redact_secrets`)**: CLI runtime outputs and `execute_agent_tool("colab_list_runtimes")` mask `token`, `access_token`, `refresh_token`, `client_secret`, and `authorization` values as `***REDACTED***` by default.
- **Working-directory `.env` opt-out**: Set `COLAB_IGNORE_CWD_DOTENV=1` to skip loading `.env` / `.env.colab` from the current working directory.
- **Static analysis**: Checked via `bandit -r colab_mcp` and `pip-audit` in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## Testing

```bash
# Run all tests (unit, integration, and live MCP schema contract tests)
python3 run_tests.py

# Run offline unit tests only
python3 run_tests.py --offline

# Syntax check and SAST
python3 -m compileall -q colab_mcp tests examples
bandit -r colab_mcp
```

---

## License & Attribution

- Original code and documentation in this repository are licensed under the **MIT License** (see [`LICENSE`](LICENSE)).
- [`skills/colab-operator/SKILL.md`](skills/colab-operator/SKILL.md) is an abridged and restructured derivative of `colab_cli/SKILL.md` from [`googlecolab/google-colab-cli`](https://github.com/googlecolab/google-colab-cli) (`Copyright 2026 Google LLC`), distributed under the **Apache License, Version 2.0** (see [`skills/colab-operator/LICENSE`](skills/colab-operator/LICENSE) and [`NOTICE`](NOTICE)).
- **Disclaimer**: This repository is an unofficial, independent project and is not affiliated with, endorsed by, or sponsored by Google LLC. "Google Colab" and "Colaboratory" are trademarks of Google LLC.
