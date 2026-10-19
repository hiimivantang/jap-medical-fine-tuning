#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server in Docker on the DGX Spark (aarch64, GB10).
#
# Usage:
#   serve_vllm.sh <container-name> <port> <hf-model-id> <revision> <served-name> [extra vllm args...]
#
# Examples:
#   serve_vllm.sh jmedft-qwen3 8100 Qwen/Qwen3-8B b968826d9c46dd6066d109eabc6255188de91218 qwen3-8b \
#       --enable-lora --max-lora-rank 16 --lora-modules jmed-lora=/adapters/qwen3-8b-jmed-lora
#
# The server binds to 127.0.0.1 only. Weights come from the host HF cache (offline mode), so the
# pinned revision must already be downloaded. Adapters are mounted read-only from results/adapters.
set -euo pipefail

if [[ $# -lt 5 ]]; then
  sed -n '2,12p' "$0"
  exit 2
fi

NAME="$1"; PORT="$2"; MODEL="$3"; REVISION="$4"; SERVED="$5"; shift 5
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.31.0}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.40}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
ADAPTER_DIR="$REPO_ROOT/experiments/01_finetuning/results/adapters"
mkdir -p "$ADAPTER_DIR"

docker run -d --rm --name "$NAME" --gpus all --ipc=host \
  -p "127.0.0.1:${PORT}:8000" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$ADAPTER_DIR:/adapters:ro" \
  -e HF_HUB_OFFLINE=1 \
  "$IMAGE" \
  "$MODEL" --revision "$REVISION" --served-model-name "$SERVED" \
  --dtype bfloat16 --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL" \
  --seed 0 --max-num-seqs 64 "$@"

echo "Started $NAME on 127.0.0.1:$PORT; waiting for readiness..."
for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo "READY $NAME"
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "Container $NAME exited during startup" >&2
    exit 1
  fi
  sleep 5
done
echo "Timed out waiting for $NAME" >&2
docker logs --tail 50 "$NAME" >&2 || true
exit 1
