"""AI エージェントが Google Colab 上で自律的に ML タスクを実行するサンプル。

処理の流れ:
1. ColabAgentHarness を初期化する
2. 指定アクセラレータ（ここでは T4 GPU）のランタイムを再利用または新規作成する
3. エージェント用の独立した Jupyter セッションを作成する
4. PyTorch のミニ学習ループを実行する
5. トークン最適化（長出力の自動トランケーション）された結果を受け取る
6. コンテキストマネージャ終了時に所有ランタイムを自動解放（teardown）する
"""

from colab_mcp.agent import ColabAgentHarness


def main():
    # with 構文と auto_teardown=True により、完了・例外時ともにクラウド資源を確実に解放する
    with ColabAgentHarness(auto_teardown=True, default_accelerator="T4") as agent:
        print("[エージェント] T4 GPU ランタイムを準備しています...")
        rt = agent.ensure_runtime(accelerator="T4")
        print(f"[エージェント] 利用ランタイム: {rt}")

        train_code = """
import torch
import torch.nn as nn

print(f"CUDA 利用可能: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"デバイス名: {torch.cuda.get_device_name(0)}")

# ミニ学習ステップの実行
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(100, 10, device=device)
linear = nn.Linear(10, 1).to(device)
y = linear(x).sum()
y.backward()
print("学習ステップが完了しました。")
"""
        print("[エージェント] 学習タスクを実行中...")
        result = agent.execute(train_code)

        print("\n[エージェント] 実行結果:")
        print("ステータス:", result["status"])
        print("出力:\n", result.get("output"))


if __name__ == "__main__":
    main()
