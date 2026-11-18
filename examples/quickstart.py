"""Quickstart example demonstrating Colab MCP & API workflow.
"""

from colab_mcp import ColabMCPClient, ColabConfig

def main():
    if not ColabConfig.get_token():
        print("Please configure COLAB_API_TOKEN in .env or environment variable.")
        return

    client = ColabMCPClient()

    # 1. Check Subscription Tier
    print("[1] Subscription:", client.get_subscription())

    # 2. Check Available Machine Specs (GPU/TPU)
    print("\n[2] Runtime Specs:", client.list_runtime_specs())

    # 3. Check Active Runtimes
    print("\n[3] Active Runtimes:", client.list_runtimes())

    # 4. Example Code Execution (requires active session)
    # res = client.execute_code("import torch; print(torch.cuda.is_available())", session="...")
    # print("\n[4] Execution Result:", res)

if __name__ == "__main__":
    main()
