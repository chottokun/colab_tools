"""Google Colab CLI: Command-Line Interface for Google Colab API & MCP.

Usage:
  colab-cli check
  colab-cli subscription
  colab-cli specs
  colab-cli runtimes list
  colab-cli runtimes create [--gpu ACCELERATOR] [--shape SHAPE] [--id ID]
  colab-cli runtimes get RUNTIME_NAME
  colab-cli runtimes delete RUNTIME_NAME
  colab-cli sessions create RUNTIME_NAME [--id SESSION_ID]
  colab-cli sessions list RUNTIME_NAME
  colab-cli sessions delete SESSION_NAME
  colab-cli exec (--session SESSION | --runtime RUNTIME) (-c CODE | -f FILE)
  colab-cli files upload RUNTIME_NAME LOCAL_FILE REMOTE_PATH
  colab-cli files download RUNTIME_NAME REMOTE_PATH LOCAL_FILE
  colab-cli proxy
"""

import argparse
import json
import os
import sys

from .config import ColabConfig
from .mcp_client import ColabMCPClient
from .jupyter_proxy import JupyterProxy
from .mcp_proxy import run_proxy


def get_client() -> ColabMCPClient:
    token = ColabConfig.get_token()
    if not token:
        sys.stderr.write(
            "Error: COLAB_API_TOKEN is not set and gcloud ADC is unavailable.\n"
            "Configure via one of the following:\n"
            f"  1. {ColabConfig.gcloud_login_command()}\n"
            "  2. export COLAB_API_TOKEN=$(gcloud auth application-default print-access-token)\n"
            "  3. Save COLAB_API_TOKEN=... in ~/.env.colab\n"
        )
        sys.exit(1)
    return ColabMCPClient(access_token=token)


def _print_json(data, show_secrets: bool = False) -> None:
    """既定で機密値（token 等）をマスクした上で JSON を整形出力する。"""
    out = data if show_secrets else ColabConfig.redact_secrets(data)
    print(json.dumps(out, indent=2, ensure_ascii=False))


def cmd_check(args):
    """Checks configuration status."""
    ok, msg = ColabConfig.verify()
    print("=" * 50)
    print(" Colab CLI Configuration Check")
    print("=" * 50)
    print(f"Status:  {'[OK]' if ok else '[ACTION REQUIRED]'}")
    print(f"Details: {msg}")
    if not ok:
        print("\nRecommended setup:")
        print(f"  {ColabConfig.gcloud_login_command()}")
        print("  export COLAB_API_TOKEN=$(gcloud auth application-default print-access-token)")


def cmd_subscription(args):
    """Retrieves subscription tier."""
    client = get_client()
    res = client.get_subscription()
    print(json.dumps(res, indent=2, ensure_ascii=False))


def cmd_specs(args):
    """Lists available runtime specs."""
    client = get_client()
    res = client.list_runtime_specs()
    print(json.dumps(res, indent=2, ensure_ascii=False))


def cmd_runtimes_list(args):
    """Lists active runtimes."""
    client = get_client()
    res = client.list_runtimes()
    _print_json(res, show_secrets=getattr(args, "show_secrets", False))


def cmd_runtimes_get(args):
    """Gets details of a runtime."""
    client = get_client()
    name = args.name if args.name.startswith("runtimes/") else f"runtimes/{args.name}"
    res = client.get_runtime(name)
    _print_json(res, show_secrets=getattr(args, "show_secrets", False))


def cmd_runtimes_create(args):
    """Creates a new runtime."""
    client = get_client()
    variant = "VARIANT_CPU"
    accelerator = "NONE"

    if args.gpu:
        variant = "VARIANT_GPU"
        accelerator = args.gpu.upper()
    elif args.tpu:
        variant = "VARIANT_TPU"
        accelerator = args.tpu.upper()

    shape = f"SHAPE_{args.shape.upper()}" if args.shape else "SHAPE_STANDARD"
    spec = {
        "variant": variant,
        "accelerator": accelerator,
        "shape": shape
    }
    print(f"Provisioning Colab runtime ({variant}, {accelerator}, {shape})...")
    res = client.create_runtime(spec=spec, runtime_id=args.id)
    _print_json(res, show_secrets=getattr(args, "show_secrets", False))


def cmd_runtimes_delete(args):
    """Deletes a runtime."""
    client = get_client()
    name = args.name if args.name.startswith("runtimes/") else f"runtimes/{args.name}"
    res = client.delete_runtime(name, allow_missing=True)
    print(json.dumps(res, indent=2, ensure_ascii=False))


def cmd_sessions_create(args):
    """Creates a Jupyter session on a runtime."""
    client = get_client()
    runtime = args.runtime if args.runtime.startswith("runtimes/") else f"runtimes/{args.runtime}"
    res = client.create_session(runtime, session_id=args.id)
    print(json.dumps(res, indent=2, ensure_ascii=False))


def cmd_sessions_list(args):
    """Lists sessions on a runtime."""
    client = get_client()
    runtime = args.runtime if args.runtime.startswith("runtimes/") else f"runtimes/{args.runtime}"
    res = client.list_sessions(runtime)
    print(json.dumps(res, indent=2, ensure_ascii=False))


def cmd_sessions_delete(args):
    """Deletes a session."""
    client = get_client()
    name = args.name if args.name.startswith("runtimes/") else f"runtimes/{args.name}"
    res = client.delete_session(name)
    print(json.dumps(res, indent=2, ensure_ascii=False))


def cmd_exec(args):
    """Executes Python code on Colab."""
    client = get_client()
    code = ""
    if args.code:
        code = args.code
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            code = f.read()
    else:
        sys.stderr.write("Error: Please provide code via -c CODE or -f FILE\n")
        sys.exit(1)

    session = args.session
    runtime = args.runtime
    if session and not session.startswith("runtimes/"):
        session = f"runtimes/{session}"
    if runtime and not runtime.startswith("runtimes/"):
        runtime = f"runtimes/{runtime}"

    res = client.execute_code(code=code, session=session, runtime=runtime)
    print(json.dumps(res, indent=2, ensure_ascii=False))


def _resolve_runtime_connection(client: ColabMCPClient, runtime_name: str):
    from .mcp_client import extract_tool_payload

    name = runtime_name if runtime_name.startswith("runtimes/") else f"runtimes/{runtime_name}"
    info = client.get_runtime(name)
    runtime_data = extract_tool_payload(info)
    if "runtime" in runtime_data and isinstance(runtime_data["runtime"], dict):
        runtime_data = runtime_data["runtime"]
    conn = runtime_data.get("connectionInfo", {})
    url = conn.get("url")
    token = conn.get("token")
    if not url or not token:
        sys.stderr.write(f"Error: Unable to retrieve connection URL/token for {name}.\n")
        sys.exit(1)
    return url, token


def cmd_files_upload(args):
    """ローカルファイルをストリーミング読み込みして Colab ランタイムへアップロードする。"""
    client = get_client()
    url, token = _resolve_runtime_connection(client, args.runtime)

    res = JupyterProxy.upload_file_path(url, token, args.local_path, args.remote_path)
    print(f"Uploaded {args.local_path} -> {args.remote_path}")
    print(json.dumps(res, indent=2, ensure_ascii=False))


def cmd_files_download(args):
    """Colab ランタイムからローカルディスクへストリーミングダウンロードする。"""
    client = get_client()
    url, token = _resolve_runtime_connection(client, args.runtime)

    size = JupyterProxy.download_file_to_path(url, token, args.remote_path, args.local_path)
    print(f"Downloaded {args.remote_path} -> {args.local_path} ({size} bytes)")


def cmd_agent_run(args):
    """Runs autonomous agent task via ColabAgentHarness."""
    from .agent.runner import ColabAgentHarness
    code = ""
    if args.code:
        code = args.code
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            code = f.read()
    else:
        sys.stderr.write("Error: Please provide code via -c CODE or -f FILE\n")
        sys.exit(1)

    if getattr(args, "cpu", False):
        accelerator = "NONE"
    elif args.tpu:
        accelerator = args.tpu
    elif args.gpu:
        accelerator = args.gpu
    else:
        accelerator = "T4"
    shape = f"SHAPE_{args.shape.upper()}" if getattr(args, "shape", None) else "SHAPE_STANDARD"

    with ColabAgentHarness(
        auto_teardown=args.teardown,
        default_accelerator=accelerator,
        validate_spec=getattr(args, "validate_spec", False),
    ) as harness:
        harness.ensure_runtime(accelerator=accelerator, shape=shape)
        res = harness.execute(code)
        _print_json(res, show_secrets=False)


def cmd_agent_tools(args):
    """Outputs LLM tool calling schema."""
    from .agent.tools import to_openai_tools, to_anthropic_tools, get_agent_tool_definitions
    fmt = (args.format or "openai").lower()
    if fmt == "openai":
        tools = to_openai_tools()
    elif fmt == "anthropic":
        tools = to_anthropic_tools()
    else:
        tools = get_agent_tool_definitions()
    print(json.dumps(tools, indent=2, ensure_ascii=False))


def cmd_proxy(args):
    """Runs stdio MCP server proxy."""
    run_proxy()


def main():
    parser = argparse.ArgumentParser(
        prog="colab-cli",
        description="Google Colab CLI: Manage runtimes, sessions, code execution, and files via Colab API & MCP."
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # check
    p_check = subparsers.add_parser("check", help="Check token and environment configuration")
    p_check.set_defaults(func=cmd_check)

    # subscription
    p_sub = subparsers.add_parser("subscription", help="Get Colab subscription tier")
    p_sub.set_defaults(func=cmd_subscription)

    # specs
    p_specs = subparsers.add_parser("specs", help="List available CPU, GPU, TPU runtime specs")
    p_specs.set_defaults(func=cmd_specs)

    # runtimes
    p_rt = subparsers.add_parser("runtimes", help="Manage Colab runtimes")
    rt_sub = p_rt.add_subparsers(dest="action", required=True)

    rt_list = rt_sub.add_parser("list", help="List active runtimes")
    rt_list.add_argument("--show-secrets", action="store_true", help="接続トークン等の機密値をマスクせず表示する")
    rt_list.set_defaults(func=cmd_runtimes_list)

    rt_get = rt_sub.add_parser("get", help="Get runtime details")
    rt_get.add_argument("name", help="Runtime name (e.g. runtimes/123 or 123)")
    rt_get.add_argument("--show-secrets", action="store_true", help="接続トークン等の機密値をマスクせず表示する")
    rt_get.set_defaults(func=cmd_runtimes_get)

    rt_create = rt_sub.add_parser("create", help="Create a runtime")
    rt_create.add_argument("--gpu", help="GPU accelerator (e.g. T4, A100, H100)")
    rt_create.add_argument("--tpu", help="TPU accelerator (e.g. V5E1, V6E1)")
    rt_create.add_argument("--shape", choices=["standard", "highmem"], default="standard", help="Memory shape")
    rt_create.add_argument("--id", help="Optional custom runtime ID")
    rt_create.add_argument("--show-secrets", action="store_true", help="接続トークン等の機密値をマスクせず表示する")
    rt_create.set_defaults(func=cmd_runtimes_create)

    rt_delete = rt_sub.add_parser("delete", help="Delete a runtime")
    rt_delete.add_argument("name", help="Runtime name")
    rt_delete.set_defaults(func=cmd_runtimes_delete)

    # sessions
    p_sess = subparsers.add_parser("sessions", help="Manage Jupyter kernel sessions")
    sess_sub = p_sess.add_subparsers(dest="action", required=True)

    sess_create = sess_sub.add_parser("create", help="Create a session on a runtime")
    sess_create.add_argument("runtime", help="Runtime name")
    sess_create.add_argument("--id", help="Optional session ID")
    sess_create.set_defaults(func=cmd_sessions_create)

    sess_list = sess_sub.add_parser("list", help="List sessions on a runtime")
    sess_list.add_argument("runtime", help="Runtime name")
    sess_list.set_defaults(func=cmd_sessions_list)

    sess_delete = sess_sub.add_parser("delete", help="Delete a session")
    sess_delete.add_argument("name", help="Session resource name")
    sess_delete.set_defaults(func=cmd_sessions_delete)

    # exec
    p_exec = subparsers.add_parser("exec", help="Execute Python code on Colab")
    exec_target = p_exec.add_mutually_exclusive_group(required=True)
    exec_target.add_argument("--session", help="Session resource name")
    exec_target.add_argument("--runtime", help="Runtime resource name")
    exec_source = p_exec.add_mutually_exclusive_group(required=True)
    exec_source.add_argument("-c", "--code", help="Python code string to execute")
    exec_source.add_argument("-f", "--file", help="Python script file to execute")
    p_exec.set_defaults(func=cmd_exec)

    # files
    p_files = subparsers.add_parser("files", help="Transfer files with Colab runtime")
    files_sub = p_files.add_subparsers(dest="action", required=True)

    f_up = files_sub.add_parser("upload", help="Upload a local file to runtime")
    f_up.add_argument("runtime", help="Runtime resource name")
    f_up.add_argument("local_path", help="Local file path")
    f_up.add_argument("remote_path", help="Remote file path (e.g. content/data.csv)")
    f_up.set_defaults(func=cmd_files_upload)

    f_down = files_sub.add_parser("download", help="Download a file from runtime")
    f_down.add_argument("runtime", help="Runtime resource name")
    f_down.add_argument("remote_path", help="Remote file path")
    f_down.add_argument("local_path", help="Local destination path")
    f_down.set_defaults(func=cmd_files_download)

    # proxy
    p_proxy = subparsers.add_parser("proxy", help="Run stdio MCP Server Proxy for Claude/Cursor")
    p_proxy.set_defaults(func=cmd_proxy)

    # agent
    p_agent = subparsers.add_parser("agent", help="AI Agent autonomous execution & schemas")
    agent_sub = p_agent.add_subparsers(dest="action", required=True)

    ag_run = agent_sub.add_parser("run", help="Run autonomous agent job on Colab")
    ag_run.add_argument("-c", "--code", help="Python code to run")
    ag_run.add_argument("-f", "--file", help="Python script file")
    ag_run.add_argument("--cpu", action="store_true", help="CPU 専用ランタイムを使用する")
    ag_run.add_argument("--gpu", default=None, help="Target GPU accelerator (T4, L4, G4, A100, H100; 既定は T4)")
    ag_run.add_argument("--tpu", help="Target TPU accelerator (e.g. V5E1, V6E1)")
    ag_run.add_argument("--shape", choices=["standard", "highmem"], default="standard", help="Memory shape")
    ag_run.add_argument("--validate-spec", action="store_true", help="Pre-check eligibility via list_runtime_specs")
    ag_run.add_argument("--teardown", action="store_true", help="Auto-teardown runtime after execution")
    ag_run.set_defaults(func=cmd_agent_run)

    ag_tools = agent_sub.add_parser("tools", help="Export LLM tool schemas")
    ag_tools.add_argument("--format", choices=["openai", "anthropic", "raw"], default="openai", help="Schema format")
    ag_tools.set_defaults(func=cmd_agent_tools)

    parsed = parser.parse_args()
    parsed.func(parsed)


if __name__ == "__main__":
    main()
