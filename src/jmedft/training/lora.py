"""LoRA SFT driver around LLaMA-Factory (isolated venv) plus adapter export and compatibility checks."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from jmedft.utils import REPO_ROOT, sha256_file

LF_VENV = REPO_ROOT / ".venv-llamafactory"

DATASET_INFO = {
    "jmed_train": {"file_name": "train.jsonl", "formatting": "sharegpt",
                   "columns": {"messages": "messages"},
                   "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user",
                            "assistant_tag": "assistant", "system_tag": "system"}},
    "jmed_val": {"file_name": "val.jsonl", "formatting": "sharegpt",
                 "columns": {"messages": "messages"},
                 "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user",
                          "assistant_tag": "assistant", "system_tag": "system"}},
}


def render_llamafactory_yaml(cfg: dict[str, Any], smoke: bool) -> tuple[Path, dict[str, Any]]:
    lf = dict(cfg["llamafactory"])
    lf["seed"] = cfg["seed"]
    lf["data_seed"] = cfg["seed"]
    if smoke:
        s = cfg["smoke"]
        lf.update(max_samples=s["max_samples"], max_steps=s["max_steps"], eval_steps=s["eval_steps"],
                  save_steps=s["save_steps"], output_dir=s["output_dir"])
    for key in ("dataset_dir", "output_dir"):
        lf[key] = str(REPO_ROOT / lf[key])
    out_dir = Path(lf["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = out_dir / "llamafactory_config.yaml"
    rendered.write_text(yaml.safe_dump(lf, allow_unicode=True, sort_keys=False))
    info = Path(lf["dataset_dir"]) / "dataset_info.json"
    info.write_text(json.dumps(DATASET_INFO, indent=2))
    return rendered, lf


def run_training(cfg: dict[str, Any], smoke: bool = False) -> Path:
    rendered, lf = render_llamafactory_yaml(cfg, smoke)
    log_path = Path(lf["output_dir"]) / "train_stdout.log"
    cmd = [str(LF_VENV / "bin/llamafactory-cli"), "train", str(rendered)]
    env = {"PATH": f"{LF_VENV / 'bin'}:/usr/bin:/bin", "HOME": str(Path.home()),
           "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "PYTHONHASHSEED": str(cfg["seed"]),
           "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "LLAMAFACTORY_VERBOSITY": "INFO"}
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"training failed (exit {proc.returncode}); see {log_path}")
    return Path(lf["output_dir"])


def export_adapter(train_dir: Path, export_dir: Path) -> dict[str, Any]:
    """Copy the final (best, since load_best_model_at_end) adapter files to the serving directory."""
    export_dir.mkdir(parents=True, exist_ok=True)
    files = ["adapter_config.json", "adapter_model.safetensors"]
    for f in files:
        shutil.copy2(train_dir / f, export_dir / f)
    for f in ["trainer_state.json", "all_results.json", "train_results.json", "eval_results.json"]:
        if (train_dir / f).exists():
            shutil.copy2(train_dir / f, export_dir / f)
    return {f: sha256_file(export_dir / f) for f in files}


def check_adapter(adapter_dir: Path, compat: dict[str, Any]) -> dict[str, Any]:
    """Validate adapter_config.json against Fireworks + vLLM LoRA constraints. Returns a report dict."""
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    targets = cfg.get("target_modules") or []
    if isinstance(targets, str):
        targets = [targets]
    checks = {
        "peft_type_is_lora": cfg.get("peft_type") == "LORA",
        "rank_in_fireworks_range": compat["rank_min"] <= cfg.get("r", 0) <= compat["rank_max"],
        "rank_within_vllm_max_lora_rank": cfg.get("r", 0) <= compat["vllm_max_lora_rank"],
        "targets_allowed": bool(targets) and set(targets) <= set(compat["allowed_target_modules"]),
        "no_modules_to_save": not cfg.get("modules_to_save"),
        "bias_none": cfg.get("bias", "none") == "none",
        "no_dora": not cfg.get("use_dora", False),
        "no_rslora": not cfg.get("use_rslora", False),
        "default_init": cfg.get("init_lora_weights", True) in (True, "gaussian"),
        "weights_safetensors": (adapter_dir / "adapter_model.safetensors").exists(),
    }
    return {"adapter_dir": str(adapter_dir), "r": cfg.get("r"), "lora_alpha": cfg.get("lora_alpha"),
            "target_modules": sorted(targets), "base_model": cfg.get("base_model_name_or_path"),
            "checks": checks, "compatible": all(checks.values())}
