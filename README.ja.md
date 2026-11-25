[English](README.md) | 日本語

# colab-tools

[![CI](https://github.com/chottokun/colab_tools/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/chottokun/colab_tools/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
[![Security: Bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Security: pip-audit](https://img.shields.io/badge/security-pip--audit-green.svg)](https://github.com/pypa/pip-audit)

Google Colab Hosted MCP エンドポイント（`https://colaboratory.googleapis.com/mcp`）、REST API（`v1beta` / `v1/operations`）、およびランタイム Jupyter プロキシを扱うための Python 標準ライブラリ製クライアント・stdio MCP ブリッジ・CLI です。外部ランタイム依存はありません。

---

## 概要

`colab-tools` は、スクリプト・CLI・ツール呼び出しを行うエージェントから Colab ランタイムの割り当て、Jupyter セッション上での Python 実行、およびランタイムプロキシ経由のファイル送受信を行います。ローカルブラウザがないヘッドレス環境にも対応しています。

```
  クライアント / エージェント / MCP ホスト
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
                  Colab ランタイム (/api/contents, /files)
```

---

## コンポーネント

- **`ColabMCPClient` (`colab_mcp/mcp_client.py`)**: Streamable HTTP（JSON / SSE）対応の JSON-RPC 2.0 クライアント。設定されたクオータプロジェクトを `x-goog-user-project` ヘッダとして送信し、HTTP 401 時は ADC からトークンを再取得して 1 回リトライします。
- **`ColabRESTClient` (`colab_mcp/api.py`)**: `/v1beta`（`subscription`, `runtimespecs`, `runtimes`）および `/v1/operations` ポーリング（`list_operations`, `get_operation`, `wait_operation`, `create_runtime(..., wait=True)`, `delete_runtime(..., wait=True)`）に対応した REST クライアント。
- **`JupyterProxy` (`colab_mcp/jupyter_proxy.py`)**: `upload_file_path` / `download_file_to_path`（ストリーミング I/O）および `upload_file` / `download_file` による直接ファイル転送。パスの URL パーセントエンコード、親ディレクトリ自動作成、30 MB チャンク分割アップロード、HTTP `Range` ダウンロードに対応しています。
- **`ColabAgentHarness` & `ContextOptimizer` (`colab_mcp/agent/`)**: CPU (`NONE`) / GPU (`T4`, `L4`, `G4`, `A100`, `H100`) / TPU (`V5E1`, `V6E1`) のライフサイクル管理。`validate_spec=True` による事前適格性確認、`shape` 一致を確認してからの再利用、所有権フラグ（`_owns_runtime`）による共有ランタイムの誤削除防止、一時障害（`unavailable`, `deadline_exceeded`, `resource_exhausted`, HTTP 429/5xx）の指数バックオフリトライ、プロキシトークン自動再取得、長出力トランケーションを行います。
- **ツールスキーマ & ディスパッチャ (`colab_mcp/agent/tools.py`)**: `to_openai_tools()`、`to_anthropic_tools()`、`execute_agent_tool()` を提供します。
- **Stdio MCP プロキシ (`colab_mcp/mcp_proxy.py`)**: 標準入出力上の JSON-RPC 2.0 を `https://colaboratory.googleapis.com/mcp` へ転送します。認証不要メソッド（`initialize`, `notifications/initialized`, `tools/list`, `ping`）ではトークン解決をスキップし、`Mcp-Session-Id` 転送、401 時リフレッシュリトライ、不正 JSON 受信時の `-32700` Parse Error 返却を行います。
- **CLI (`colab_mcp/cli.py`)**: `colab-cli` / `python3 -m colab_mcp.cli`。既定で機密フィールド（`token`, `access_token`, `refresh_token`, `client_secret`）をマスク表示し（`--show-secrets` で解除）、`--cpu` / `--gpu` / `--tpu` 指定に対応しています。

---

## インストールと認証

### インストール

```bash
git clone https://github.com/chottokun/colab_tools.git
cd colab_tools
pip install -e .
```

### Colab API の有効化とクオータプロジェクト設定

ユーザー OAuth または ADC で `colaboratory.googleapis.com` を呼び出すには、API を有効化した Google Cloud プロジェクトを `x-goog-user-project` ヘッダに指定する必要があります。

```bash
gcloud services enable colaboratory.googleapis.com --project <YOUR_GCP_PROJECT_ID>
export COLAB_QUOTA_PROJECT="<YOUR_GCP_PROJECT_ID>"
```

### 認証トークンの解決順序

`ColabConfig`（`colab_mcp/config.py`）は以下の優先順位でトークンを解決します。

1. クライアントまたはハーネスへの明示的な `access_token` 引数。
2. 環境変数（`COLAB_API_TOKEN`, `GOOGLE_OAUTH_TOKEN`, `GCLOUD_AUTH_TOKEN`）または `.env` / `.env.colab` / `~/.env.colab`。
3. `gcloud` CLI（`gcloud auth application-default print-access-token`）。
4. ADC JSON ファイル（`GOOGLE_APPLICATION_CREDENTIALS` または `~/.config/gcloud/application_default_credentials.json`、`type == "authorized_user"`）の `refresh_token` を用いた HTTPS 直接更新（`adc:json`）。

ADC トークンは有効期限の 120 秒前までメモリにキャッシュされます。HTTP 401 受信時は、プロセス内で失効した環境変数トークンを無効化し、更新した ADC トークンへ切り替えます。

### 対話環境での `gcloud` セットアップ

```bash
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory

gcloud auth application-default set-quota-project <YOUR_GCP_PROJECT_ID>

colab-cli check
```

### ヘッドレス運用（ブラウザなし・`gcloud` CLI なし）

- **前提：クオータプロジェクトで Colab API（colaboratory.googleapis.com）が有効化されていること。** ADC 等のユーザー認証では、トークンが有効でもクオータプロジェクト未設定だと API が 403 を返します。有効化はコンソール（https://console.developers.google.com/apis/api/colaboratory.googleapis.com/overview?project=<YOUR_GCP_PROJECT_ID>）から行い、反映まで数分待ちます。`colab-cli check` はクオータプロジェクト未設定を `[ACTION REQUIRED]` として報告します。
- 有効期限の短い `COLAB_API_TOKEN`（約 1 時間）を `.env` に固定する運用は推奨しません。
- 別端末等で取得した `application_default_credentials.json`（`refresh_token` と `quota_project_id` を含む）を `~/.config/gcloud/application_default_credentials.json` に `chmod 600` で配置し、`.env` または環境変数には `COLAB_QUOTA_PROJECT` のみを設定します。
- これにより、`gcloud` CLI が存在しない環境でも `ColabConfig` が `token_uri` に対してアクセストークンを直接再発行します。

### 環境変数一覧

| 変数名 | 用途 | 備考 |
| :--- | :--- | :--- |
| `COLAB_QUOTA_PROJECT` | GCP クオータプロジェクト ID | `GOOGLE_CLOUD_QUOTA_PROJECT`, `GOOGLE_CLOUD_PROJECT`, `GCLOUD_PROJECT`, ADC JSON の `quota_project_id` も参照 |
| `COLAB_API_TOKEN` | 明示的アクセストークン | 401 発生時はプロセス内で無効化し ADC へ切替（`GOOGLE_OAUTH_TOKEN`, `GCLOUD_AUTH_TOKEN` も参照） |
| `GOOGLE_APPLICATION_CREDENTIALS` | ADC JSON のカスタムパス | 未指定時は `~/.config/gcloud/application_default_credentials.json` |
| `COLAB_IGNORE_CWD_DOTENV` | 作業ディレクトリ `.env` の無視 | `1` または `true` で `~/.env.colab` のみ読込 |
| `COLAB_SKIP_LIVE_TESTS` | ライブテストのスキップ | `1` または `true` で `run_tests.py` の `test_live_mcp.py` を除外 |

---

## 使い方

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

#### ファイル同期と実行

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

#### LRO ポーリング付き REST API

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

#### LLM ツール定義と実行ディスパッチ

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
# 設定確認
colab-cli check

# アカウント・ハードウェアスペック確認
colab-cli subscription
colab-cli specs

# ランタイム操作（connectionInfo.token は既定で ***REDACTED*** にマスク表示）
colab-cli runtimes list
colab-cli runtimes create --gpu T4 --shape standard
colab-cli runtimes get <RUNTIME_NAME>          # 生トークンが必要な場合のみ --show-secrets を指定
colab-cli runtimes delete <RUNTIME_NAME>

# セッション操作と実行
colab-cli sessions create <RUNTIME_NAME>
colab-cli sessions list <RUNTIME_NAME>
colab-cli exec --session <SESSION_NAME> -c "print('hello')"
colab-cli sessions delete <SESSION_NAME>

# ストリーミングファイル転送
colab-cli files upload <RUNTIME_NAME> ./local.bin content/remote.bin
colab-cli files download <RUNTIME_NAME> content/remote.bin ./downloaded.bin

# 自律実行ワンライナーとツール定義出力
colab-cli agent run --cpu --validate-spec --teardown -c "print('CPU OK')"
colab-cli agent run --gpu T4 --teardown -f train.py
colab-cli agent tools --format openai
colab-cli agent tools --format anthropic

# Stdio MCP プロキシ
colab-cli proxy
```

### MCP クライアント設定

ADC 自動更新を利用する場合、`COLAB_QUOTA_PROJECT` のみを指定する設定例（[`examples/mcp_config.json`](examples/mcp_config.json)）：

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

### 同梱スキル `colab-operator`

公式 `colab` CLI を使用するターミナルエージェント向けのリファレンスとして、[`googlecolab/google-colab-cli`](https://github.com/googlecolab/google-colab-cli) 由来の抜粋ファイル [`skills/colab-operator/SKILL.md`](skills/colab-operator/SKILL.md) を同梱しています。

---

## セキュリティ

- **URL 検証 (`ColabConfig.validate_url`)**: `https://` およびループバックホスト（`localhost`, `127.0.0.1`, `::1`）上の `http://` のみを許可します。
- **機密値のマスク (`ColabConfig.redact_secrets`)**: CLI のランタイム出力や `execute_agent_tool("colab_list_runtimes")` において、`token`、`access_token`、`refresh_token`、`client_secret`、`authorization` を既定で `***REDACTED***` に置換します。
- **作業ディレクトリ `.env` の無効化**: `COLAB_IGNORE_CWD_DOTENV=1` を指定すると、カレントディレクトリの `.env` / `.env.colab` を読み込みません。
- **静的解析**: [`.github/workflows/ci.yml`](.github/workflows/ci.yml) で `bandit -r colab_mcp` および `pip-audit` を実行しています。

---

## テスト

```bash
# 全テスト実行（単体・統合・ライブ MCP スキーマ契約テスト）
python3 run_tests.py

# オフライン単体テストのみ実行
python3 run_tests.py --offline

# 構文確認・静的セキュリティ解析
python3 -m compileall -q colab_mcp tests examples
bandit -r colab_mcp
```

---

## ライセンスと出典

- 本リポジトリ独自のコードおよびドキュメントは **MIT License**（[`LICENSE`](LICENSE)）の下で提供されます。
- [`skills/colab-operator/SKILL.md`](skills/colab-operator/SKILL.md) は [`googlecolab/google-colab-cli`](https://github.com/googlecolab/google-colab-cli)（`Copyright 2026 Google LLC`）同梱の `colab_cli/SKILL.md` を抜粋・再構成した派生ファイルであり、**Apache License, Version 2.0**（[`skills/colab-operator/LICENSE`](skills/colab-operator/LICENSE) および [`NOTICE`](NOTICE)）に従って配布しています。
- **免責事項**: 本プロジェクトは非公式の独立ツールであり、Google LLC による公式提供・承認・後援を受けたものではありません。「Google Colab」および「Colaboratory」は Google LLC の商標です。
