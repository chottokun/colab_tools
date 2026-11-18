"""Example showing direct file upload and download via Colab runtime proxy.
"""

from colab_mcp import ColabMCPClient, JupyterProxy

def main():
    client = ColabMCPClient()
    runtimes = client.list_runtimes().get("result", {}).get("runtimes", [])
    if not runtimes:
        print("No active runtimes found.")
        return

    rt = runtimes[0]
    rt_name = rt["name"]
    conn = rt.get("connectionInfo", {})
    url = conn.get("url")
    token = conn.get("token")

    print(f"Connecting to runtime: {rt_name}")

    # Upload local file
    data = b"Hello from local machine to Google Colab!"
    upload_res = JupyterProxy.upload_file(url, token, "content/hello.txt", data)
    print("Upload result:", upload_res)

    # Download file back
    downloaded = JupyterProxy.download_file(url, token, "content/hello.txt")
    print("Downloaded content:", downloaded.decode("utf-8"))

if __name__ == "__main__":
    main()
