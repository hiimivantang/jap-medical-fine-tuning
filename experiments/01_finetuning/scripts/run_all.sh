#!/usr/bin/env bash
# End-to-end reproduction of Experiment 01 on a DGX Spark. Each GPU stage runs alone (shared unified memory).
# Stages can be selected: run_all.sh [synth] [build] [train] [infer] [evaluate]  (default: all)
set -euo pipefail
cd "$(dirname "$0")/../../.."
J=.venv/bin/jmedft
S=experiments/01_finetuning/scripts
stages=("$@"); [[ ${#stages[@]} -eq 0 ]] && stages=(synth build train infer evaluate)
has() { [[ " ${stages[*]} " == *" $1 "* ]]; }

if has synth; then
  VLLM_GPU_UTIL=0.55 VLLM_MAX_MODEL_LEN=16384 bash $S/serve_vllm.sh jmedft-teacher 8200 Qwen/Qwen3.8-27B-FP8 \
    017b9c7af6b5689d5dd426a76e0bc077eb5ca20a qwen3.8-27b --language-model-only
  $J data synth --splits test_synth val train
  docker stop jmedft-teacher
  bash $S/serve_nemotron_generator.sh
  $J data synth --splits test_xgen
  docker stop jmedft-xgen
  $J data revalidate --splits train val test_synth test_xgen
fi

if has build; then
  $J data build
  $J data validate
fi

if has train; then
  $J train --smoke
  $J train
  $J adapter-check
fi

if has infer; then
  $J serve --server qwen3
  $J infer --condition E1_qwen3_8b
  $J infer --condition E3_qwen3_8b_lora
  $J latency --condition E1_qwen3_8b
  $J latency --condition E3_qwen3_8b_lora
  $J serve --server qwen3 --stop
  $J serve --server swallow
  $J infer --condition E2_swallow_8b
  $J latency --condition E2_swallow_8b
  $J infer --condition E2t_swallow_8b_think --splits test_synth test_xgen test_ext --sample 60
  $J serve --server swallow --stop
  $J serve --server swallow_med
  $J infer --condition E2m_medical_swallow_8b
  $J serve --server swallow_med --stop
  $J infer --condition E4_gpt41 --concurrency 16
  $J latency --condition E4_gpt41 --concurrency 16
  $J infer --condition E4b_gpt54mini --concurrency 8 --splits test_synth test_xgen test_ext
  $J latency --condition E4b_gpt54mini --concurrency 16
fi

if has evaluate; then
  $J evaluate
  $J report
fi
