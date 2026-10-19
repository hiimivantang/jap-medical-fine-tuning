#!/usr/bin/env bash
# Isolated LLaMA-Factory environment for LoRA training on the DGX Spark (aarch64, CUDA 13.0).
#
# Follows NVIDIA's dgx-spark-playbooks `playbook-llama-factory` pattern (Python venv + PyTorch cu130
# wheels) with two deviations, both for reproducibility:
#   * llamafactory is pinned to the 0.9.5 PyPI release instead of a git clone of main;
#   * torch/torchvision/torchaudio are pinned to a matched cu130 set (torchaudio wheels stop at 2.11,
#     and LLaMA-Factory imports torchaudio at startup, so an unmatched torch 2.14 install breaks).
# It is a separate venv because llamafactory 0.9.5 pins transformers<=5.6, peft<=0.18.1, trl<=0.24,
# which conflict with the main project venv.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
VENV="${LLAMAFACTORY_VENV:-$REPO_ROOT/.venv-llamafactory}"

uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" \
  --index-url https://download.pytorch.org/whl/cu130 \
  "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0"
uv pip install --python "$VENV/bin/python" \
  -r "$REPO_ROOT/experiments/01_finetuning/configs/requirements-llamafactory.txt"

"$VENV/bin/python" - <<'EOF'
import torch, transformers, peft, trl, llamafactory
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__, "peft", peft.__version__, "trl", trl.__version__,
      "llamafactory", llamafactory.__version__)
EOF
