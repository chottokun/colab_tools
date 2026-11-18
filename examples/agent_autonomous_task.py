"""Example of an AI Agent performing an autonomous ML task on Google Colab.

Workflow:
1. Agent initializes ColabAgentHarness
2. Agent requests GPU execution
3. Harness reuses or provisions a GPU runtime
4. Harness creates an isolated session
5. Agent runs PyTorch training loop
6. Output is token-optimized and returned to the LLM agent
7. Resources are automatically cleaned up
"""

from colab_mcp.agent import ColabAgentHarness

def main():
    # Use context manager for automatic cloud resource teardown
    with ColabAgentHarness(auto_teardown=True, default_accelerator="T4") as agent:
        print("[Agent] Ensuring T4 GPU runtime...")
        rt = agent.ensure_runtime(accelerator="T4")
        print(f"[Agent] Using runtime: {rt}")

        train_code = """
import torch
import torch.nn as nn

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")

# Mini training loop
x = torch.randn(100, 10).cuda() if torch.cuda.is_available() else torch.randn(100, 10)
linear = nn.Linear(10, 1).cuda() if torch.cuda.is_available() else nn.Linear(10, 1)
y = linear(x).sum()
y.backward()
print("Training step completed. Loss backward successful!")
"""
        print("[Agent] Executing training task...")
        result = agent.execute(train_code)

        print("\n[Agent] Output received:")
        print("Status:", result["status"])
        print("Output:\n", result.get("output"))

if __name__ == "__main__":
    main()
