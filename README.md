# Japanese medical LoRA fine-tuning on NVIDIA DGX Spark — Experiment 01

Can a small open-weight model (Qwen3-8B) fine-tuned with LoRA on Japanese medical extraction/classification
tasks match or beat (a) the untuned model, (b) a Japanese-specialised model of the same size
(Qwen3-Swallow-8B) and (c) an OpenAI reference — without losing general Japanese ability?

- Report: [`REPORT.md`](REPORT.md) · customer summary: [`CUSTOMER_SUMMARY.md`](CUSTOMER_SUMMARY.md)
- Pre-registered design: [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md) · data licences & leakage controls:
  [`docs/DATASETS.md`](docs/DATASETS.md) · data quality review: [`docs/DATA_QUALITY.md`](docs/DATA_QUALITY.md)
- Machine-readable results: `experiments/01_finetuning/results/metrics/` · manifest:
  `experiments/01_finetuning/results/manifest.json`

## Layout

```
src/jmedft/
  tasks.py            prompts, label sets, JSON schemas (single source of truth for every model)
  datasets/           synthetic spec sampling + generation + validation, third-party converters, split builder
  training/           LLaMA-Factory LoRA driver, adapter export, Fireworks/vLLM compatibility check
  inference/          async OpenAI-compatible runner (local vLLM or OpenAI), resumable
  evaluation/         parsing, metrics, bootstrap/McNemar/Holm statistics, hypothesis tests
  reporting/          charts, tables, error examples, experiment manifest
  cli.py              `jmedft` command line
experiments/01_finetuning/
  configs/            data.yaml, term_inventory.yaml, conditions.yaml, train_lora.yaml, pricing.yaml
  scripts/            serve_vllm.sh, setup_llamafactory.sh, fetch_external_data.sh, run_all.sh
  results/            adapters/, predictions/, metrics/, figures/, training/, manifest.json
data/
  raw/                third-party data (re-downloadable, git-ignored)
  synthetic/          generated notes with specs and validation logs
  processed/          sft/ and eval/ splits + SHA256SUMS, split_stats.json, leakage_report.json
tests/                unit tests (pytest)
```

## Environment (verified on this machine)

| Item | Value |
|---|---|
| Hardware | NVIDIA DGX Spark, GB10 (sm_121), 128 GB unified memory, aarch64 |
| OS / driver / CUDA | Ubuntu 24.04.4, driver 580.173.02, CUDA 13.0 |
| Main venv (`.venv`, uv) | Python 3.12, torch 2.14.1+cu130, transformers 5.20.1, peft 0.21.2 |
| Training venv (`.venv-llamafactory`) | llamafactory 0.9.5, torch 2.11.0+cu130, transformers 5.6.0, peft 0.18.1, trl 0.24.0 |
| Inference | Docker `vllm/vllm-openai:v0.31.0` (arm64, torch 2.13+cu130) |

## Reproduction

Prerequisites: `uv`, Docker with the NVIDIA runtime, ~60 GB free disk for model weights, `.env` with
`OPENAI_API_KEY` (only for E4/E4b; see `.env.example`). Model revisions are pinned in
`experiments/01_finetuning/configs/conditions.yaml`.

```bash
# 0. environments
uv sync --extra dev                                         # main venv
bash experiments/01_finetuning/scripts/setup_llamafactory.sh  # isolated training venv
.venv/bin/pytest                                            # unit tests

# 1. models (pinned revisions) and third-party data
uvx --from "huggingface_hub[hf_xet]" hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218
uvx --from "huggingface_hub[hf_xet]" hf download tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2 --revision 9218f4843b6f93369a0b0999d8f58d61487ea71c
uvx --from "huggingface_hub[hf_xet]" hf download tokyotech-llm/Medical-Qwen3-Swallow-8B --revision f9a3d4dc38270d39196a8c5b5e14f56088fd0876
uvx --from "huggingface_hub[hf_xet]" hf download Qwen/Qwen3.8-27B-FP8 --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a
uvx --from "huggingface_hub[hf_xet]" hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 --revision 6dbbd757ea75a8ece6e0702872e3ae53f9987728
bash experiments/01_finetuning/scripts/fetch_external_data.sh

# 2. synthetic data (generator servers, then generation)
VLLM_GPU_UTIL=0.55 VLLM_MAX_MODEL_LEN=16384 bash experiments/01_finetuning/scripts/serve_vllm.sh \
  jmedft-teacher 8200 Qwen/Qwen3.8-27B-FP8 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a qwen3.8-27b --language-model-only
.venv/bin/jmedft data synth --splits test_synth val train
docker stop jmedft-teacher
# (Nemotron server command: see experiments/01_finetuning/scripts/run_all.sh)
.venv/bin/jmedft data synth --splits test_xgen

# 3. build + validate splits
.venv/bin/jmedft data build
.venv/bin/jmedft data validate

# 4. fine-tune (smoke test first), export adapter, check Fireworks/vLLM compatibility
.venv/bin/jmedft train --smoke
.venv/bin/jmedft train
.venv/bin/jmedft adapter-check

# 5. inference for every condition (local servers one at a time; OpenAI needs .env)
.venv/bin/jmedft serve --server qwen3
.venv/bin/jmedft infer --condition E1_qwen3_8b
.venv/bin/jmedft infer --condition E3_qwen3_8b_lora
.venv/bin/jmedft serve --server qwen3 --stop
.venv/bin/jmedft serve --server swallow
.venv/bin/jmedft infer --condition E2_swallow_8b
.venv/bin/jmedft infer --condition E2t_swallow_8b_think
.venv/bin/jmedft serve --server swallow --stop
.venv/bin/jmedft infer --condition E4_gpt41 --concurrency 16

# 6. score, test hypotheses, build charts/tables/manifest
.venv/bin/jmedft evaluate
.venv/bin/jmedft report
```

`experiments/01_finetuning/scripts/run_all.sh` chains steps 2–6.

## Safety and scope

All synthetic notes describe fictional patients; a validator rejects dates, phone numbers and postcodes. No real
patient data is used. The system extracts and classifies information stated in text; it does not diagnose or
recommend treatment. API keys are read from `.env` (git-ignored) and redacted from any logged error.
