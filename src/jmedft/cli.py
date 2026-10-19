"""`jmedft` command-line interface.

  jmedft data synth    --splits train val test_synth test_xgen   generate synthetic notes (needs generator server)
  jmedft data revalidate --splits ...                            re-apply the current validator to generated notes
  jmedft data build                                              assemble train/val/test files
  jmedft data validate                                           integrity + leakage checks on built data
  jmedft serve   --server qwen3|swallow|swallow_med [--stop]     start/stop a vLLM container
  jmedft infer   --condition E1_qwen3_8b --splits test_synth ... baseline / adapter inference
  jmedft latency --condition E3_qwen3_8b_lora                    single-stream + batched latency profile
  jmedft train   [--smoke]                                       LoRA fine-tuning (LLaMA-Factory)
  jmedft adapter-check                                           Fireworks / vLLM LoRA compatibility
  jmedft evaluate                                                score all predictions, statistics
  jmedft report                                                  charts, report tables, manifest
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

from jmedft.utils import REPO_ROOT, load_env, load_yaml, read_jsonl

CFG_DIR = REPO_ROOT / "experiments/01_finetuning/configs"
RESULTS = REPO_ROOT / "experiments/01_finetuning/results"
EVAL_DIR = REPO_ROOT / "data/processed/eval"


def _cmd_data(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    if args.action == "synth":
        from jmedft.datasets.synth_generate import generate_split

        for split in args.splits:
            out = REPO_ROOT / f"data/synthetic/{split}.raw.jsonl"
            stats = asyncio.run(generate_split(cfg, split, out, retries=args.retries, limit=args.limit))
            print(json.dumps(stats, ensure_ascii=False))
    elif args.action == "build":
        from jmedft.datasets.build import build

        print(json.dumps(build(cfg), ensure_ascii=False, indent=2))
    elif args.action == "revalidate":
        from jmedft.datasets.revalidate import revalidate_file

        for split in args.splits:
            print(json.dumps(revalidate_file(REPO_ROOT / f"data/synthetic/{split}.raw.jsonl", cfg), ensure_ascii=False))
    elif args.action == "validate":
        from jmedft.datasets.validate import validate_all

        report = validate_all(cfg)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["ok"]:
            sys.exit(1)


def _cmd_serve(args: argparse.Namespace) -> None:
    conds = load_yaml(CFG_DIR / "conditions.yaml")
    srv = conds["servers"][args.server]
    if args.stop:
        subprocess.run(["docker", "stop", srv["container"]], check=False)
        return
    script = REPO_ROOT / "experiments/01_finetuning/scripts/serve_vllm.sh"
    env_prefix = {"VLLM_IMAGE": conds["vllm_image"], "VLLM_GPU_UTIL": str(srv["gpu_util"])}
    import os

    subprocess.run(["bash", str(script), srv["container"], str(srv["port"]), srv["hf_id"], srv["revision"],
                    srv["served_name"], *srv["extra_args"]], check=True, env=os.environ | env_prefix)


def _cmd_infer(args: argparse.Namespace) -> None:
    from jmedft.inference.runner import make_client, run_items

    load_env()
    conds = load_yaml(CFG_DIR / "conditions.yaml")
    cond = dict(conds["conditions"][args.condition])
    cond["name"] = args.condition
    if cond["provider"] == "vllm":
        srv = conds["servers"][cond["server"]]
        cond["base_url"] = f"http://127.0.0.1:{srv['port']}/v1"
    client = make_client(cond)
    for split in args.splits:
        for path in sorted((EVAL_DIR / split).glob("*.jsonl")):
            task = path.stem
            if args.tasks and task not in args.tasks:
                continue
            items = read_jsonl(path)
            if args.sample:
                import random

                items = sorted(random.Random(f"{split}:{task}").sample(items, min(args.sample, len(items))),
                               key=lambda it: it["id"])
            if args.limit:
                items = items[: args.limit]
            out = RESULTS / f"predictions/{args.condition}/{split}/{task}.jsonl"
            stats = asyncio.run(run_items(cond, items, out, args.concurrency, client))
            print(json.dumps({"condition": args.condition, "split": split, "task": task} | stats), flush=True)


def _cmd_latency(args: argparse.Namespace) -> None:
    from jmedft.inference.latency import profile

    load_env()
    conds = load_yaml(CFG_DIR / "conditions.yaml")
    cond = dict(conds["conditions"][args.condition])
    cond["name"] = args.condition
    if cond["provider"] == "vllm":
        cond["base_url"] = f"http://127.0.0.1:{conds['servers'][cond['server']]['port']}/v1"
    res = profile(cond, n_single=args.n_single, n_batched=args.n_batched, concurrency=args.concurrency)
    print(json.dumps(res, indent=1))


def _cmd_train(args: argparse.Namespace) -> None:
    from jmedft.training.lora import check_adapter, export_adapter, run_training

    cfg = load_yaml(args.config)
    out_dir = run_training(cfg, smoke=args.smoke)
    print(f"training finished: {out_dir}")
    if not args.smoke:
        export_dir = REPO_ROOT / cfg["export_adapter_dir"]
        hashes = export_adapter(out_dir, export_dir)
        print(json.dumps({"exported": str(export_dir), "sha256": hashes}, indent=2))
        print(json.dumps(check_adapter(export_dir, cfg["compatibility"]), indent=2))


def _cmd_adapter_check(args: argparse.Namespace) -> None:
    from jmedft.training.lora import check_adapter

    cfg = load_yaml(args.config)
    report = check_adapter(Path(args.adapter_dir or REPO_ROOT / cfg["export_adapter_dir"]), cfg["compatibility"])
    out = RESULTS / "adapter_compatibility.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["compatible"]:
        sys.exit(1)


def _cmd_evaluate(args: argparse.Namespace) -> None:
    from jmedft.evaluation.run import evaluate_all

    print(json.dumps(evaluate_all(args.conditions), ensure_ascii=False, indent=2)[:4000])


def _cmd_report(args: argparse.Namespace) -> None:
    from jmedft.reporting.report import generate_report

    print(generate_report())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="jmedft", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="dataset preparation and validation")
    d.add_argument("action", choices=["synth", "revalidate", "build", "validate"])
    d.add_argument("--config", default=str(CFG_DIR / "data.yaml"))
    d.add_argument("--splits", nargs="+", default=["test_synth", "val", "train", "test_xgen"])
    d.add_argument("--retries", type=int, default=2)
    d.add_argument("--limit", type=int, default=None)
    d.set_defaults(func=_cmd_data)

    s = sub.add_parser("serve", help="start/stop a vLLM server container")
    s.add_argument("--server", required=True, choices=["qwen3", "swallow", "swallow_med"])
    s.add_argument("--stop", action="store_true")
    s.set_defaults(func=_cmd_serve)

    i = sub.add_parser("infer", help="run inference for one condition")
    i.add_argument("--condition", required=True)
    i.add_argument("--splits", nargs="+", default=["test_synth", "test_xgen", "test_ext", "test_general"])
    i.add_argument("--tasks", nargs="*", default=None)
    i.add_argument("--concurrency", type=int, default=32)
    i.add_argument("--limit", type=int, default=None)
    i.add_argument("--sample", type=int, default=None, help="seeded random subset of N items per task")
    i.set_defaults(func=_cmd_infer)

    lt = sub.add_parser("latency", help="single-stream and batched latency/throughput profile")
    lt.add_argument("--condition", required=True)
    lt.add_argument("--n-single", type=int, default=30)
    lt.add_argument("--n-batched", type=int, default=150)
    lt.add_argument("--concurrency", type=int, default=32)
    lt.set_defaults(func=_cmd_latency)

    t = sub.add_parser("train", help="LoRA fine-tuning")
    t.add_argument("--config", default=str(CFG_DIR / "train_lora.yaml"))
    t.add_argument("--smoke", action="store_true")
    t.set_defaults(func=_cmd_train)

    a = sub.add_parser("adapter-check", help="LoRA adapter compatibility (Fireworks / vLLM)")
    a.add_argument("--config", default=str(CFG_DIR / "train_lora.yaml"))
    a.add_argument("--adapter-dir", default=None)
    a.set_defaults(func=_cmd_adapter_check)

    e = sub.add_parser("evaluate", help="score predictions and compute statistics")
    e.add_argument("--conditions", nargs="*", default=None)
    e.set_defaults(func=_cmd_evaluate)

    r = sub.add_parser("report", help="generate charts, tables, report and manifest")
    r.set_defaults(func=_cmd_report)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
