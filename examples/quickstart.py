"""Colab MCP & API の基本ワークフローを示すクイックスタートサンプル。"""

from colab_mcp import ColabConfig, ColabMCPClient


def main():
    # 環境変数または ADC（gcloud / ADC JSON）からトークンを確認する
    if not ColabConfig.get_token():
        print("認証トークンが未設定です。COLAB_QUOTA_PROJECT と ADC（または COLAB_API_TOKEN）を設定してください。")
        return

    client = ColabMCPClient()

    # 1. サブスクリプションティアを確認
    print("[1] サブスクリプション:", client.get_subscription())

    # 2. 利用可能なランタイムスペック（CPU / GPU / TPU）を確認
    print("\n[2] ランタイムスペック一覧:", client.list_runtime_specs())

    # 3. 現在稼働中のランタイム一覧を確認
    print("\n[3] 稼働中ランタイム一覧:", client.list_runtimes())

    # 4. コード実行例（アクティブなセッションが必要）
    # res = client.execute_code("import torch; print(torch.cuda.is_available())", session="...")
    # print("\n[4] 実行結果:", res)


if __name__ == "__main__":
    main()
