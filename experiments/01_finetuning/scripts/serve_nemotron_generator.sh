#!/usr/bin/env bash
# Serve the cross-generator model (test_xgen only) on 127.0.0.1:8300.
# Flags follow the model card's DGX Spark recipe (vLLM v0.27.1, marlin MoE + flashinfer mamba backends);
# speculative decoding is omitted because generation volume is small.
set -euo pipefail
docker run -d --rm --name jmedft-xgen --gpus all --ipc=host -p 127.0.0.1:8300:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -e HF_HUB_OFFLINE=1 \
  vllm/vllm-openai:v0.27.1 \
  nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 --revision 6dbbd757ea75a8ece6e0702872e3ae53f9987728 \
  --served-model-name nemotron-3.5-lightning --moe-backend marlin --mamba-backend flashinfer \
  --kv-cache-dtype fp8 --max-model-len 16384 --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.50}" \
  --reasoning-parser nemotron_v3 --seed 0
for _ in $(seq 1 240); do
  if curl -sf http://127.0.0.1:8300/v1/models >/dev/null; then echo "READY jmedft-xgen"; exit 0; fi
  docker ps --format '{{.Names}}' | grep -qx jmedft-xgen || { echo "container exited" >&2; exit 1; }
  sleep 5
done
echo "timeout" >&2; docker logs --tail 50 jmedft-xgen >&2; exit 1
