---
name: colab-operator
description: "Operate Google Colab environments via the official colab CLI. Use when asked to create or manage GPU/TPU sessions, run Python/shell on a remote Colab VM, sync files, automate environment setup (packages, auth, Drive), or export session history."
---

# Skill: Colab Session Operator

Operate Google Colab environments via the `colab` CLI: provision GPU/TPU sessions, run Python/shell on the VM, sync files, and capture work as notebooks.

## Installation
If the user does not already have the `colab` tool installed, it can be acquired via:
```bash
uv tool install google-colab-cli
# or
pip install google-colab-cli
```

## Mental Model
1. **A session == a live Jupyter kernel on a rented VM**:
   - `colab new` allocates a billable VM; `colab stop` releases it.
   - Sessions stay alive as long as the kernel is active.
2. **Kernel state PERSISTS across calls in the same session**:
   - `colab exec -s <name>` reattaches to the same kernel.
   - Imports, variables, and defined functions survive between separate exec commands.
   - Build up state incrementally; do not re-import everything each call.
3. **Default working directory is `/content`**:
   - Every `exec` / `run` operates from `/content`. Prefer absolute paths (`/content/...`).
4. **Fire-and-forget**:
   - Each command authenticates, performs the action, and exits.

## Ephemeral One-shot Jobs (`colab run`)
For autonomous scripts without needing manual session teardown:
```bash
# Provision T4 GPU, run script, retrieve output, and auto-terminate VM on completion
colab run --gpu T4 script.py [args...]
```
- Exit codes propagate: uncaught exceptions or `sys.exit(1)` exit non-zero.
- Automatically tears down the VM even if the script crashes (zero wasted compute units).

## Interactive Session Workflow
```bash
# 1. Provision GPU session
colab new -s my-session --gpu T4

# 2. Install dependencies (uses uv in the VM)
colab install -s my-session torch torchvision

# 3. Execute code or files
colab exec -s my-session -c "import torch; print(torch.cuda.is_available())"
colab exec -s my-session -f train.py

# 4. File transfer
colab upload -s my-session local_dataset.csv /content/data.csv
colab download -s my-session /content/model.pt ./model.pt

# 5. ALWAYS stop the session when finished to release billable resources
colab stop -s my-session
```

## Safety & Cost Rules for AI Agents
- **Always `colab stop -s <name>`**: Idle VMs burn compute units continuously.
- **Never invoke interactive commands**:
  - `colab repl`, `colab auth`, and `colab drivemount` expect a TTY and will hang when invoked by an autonomous agent.

---

## License & Attribution
- **Source**: [`googlecolab/google-colab-cli`](https://github.com/googlecolab/google-colab-cli) (`colab_cli/SKILL.md`)
- **Copyright**: Copyright 2026 Google LLC
- **License**: Apache License, Version 2.0 (see [`LICENSE`](LICENSE) in this directory)
- **Modification Notice**: This file is an abridged and restructured version of the original `colab_cli/SKILL.md`.
