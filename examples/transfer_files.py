"""Google Colab ランタイムプロキシ経由でファイルを直接送受信するサンプル。
"""

from colab_mcp import ColabMCPClient, JupyterProxy
from colab_mcp.mcp_client import extract_tool_payload


def main():
    client = ColabMCPClient()
    # 公式 MCP レスポンス（structuredContent 等）から稼働中ランタイム一覧を抽出する
    payload = extract_tool_payload(client.list_runtimes())
    runtimes = payload.get("runtimes", [])
    if not runtimes:
        print("アクティブなランタイムが見つかりません。")
        return

    rt = runtimes[0]
    rt_name = rt["name"]
    conn = rt.get("connectionInfo", {})
    url = conn.get("url")
    token = conn.get("token")
    if not url or not token:
        print(f"ランタイム {rt_name} の接続情報が取得できません。")
        return

    print(f"接続先ランタイム: {rt_name}")

    # バイナリデータをアップロード
    data = b"Hello from local machine to Google Colab!"
    upload_res = JupyterProxy.upload_file(url, token, "content/hello.txt", data)
    print("アップロード結果:", upload_res)

    # アップロードしたファイルをダウンロードして検証
    downloaded = JupyterProxy.download_file(url, token, "content/hello.txt")
    print("ダウンロード内容:", downloaded.decode("utf-8"))


if __name__ == "__main__":
    main()
