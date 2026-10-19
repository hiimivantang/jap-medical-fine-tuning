"""Latency / throughput profile of one condition on a fixed item sample.

Two regimes, measured separately because they answer different questions:
  * single-stream (concurrency 1): per-request latency a user would see on an idle server;
  * batched (concurrency N): aggregate throughput (output tokens/s, requests/s) — the economics regime.
Results go to experiments/01_finetuning/results/metrics/latency/<condition>.json.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import numpy as np

from jmedft.inference.runner import _one, make_client
from jmedft.utils import REPO_ROOT, read_jsonl

OUT = REPO_ROOT / "experiments/01_finetuning/results/metrics/latency"


def _summ(recs: list[dict[str, Any]], wall: float) -> dict[str, Any]:
    ok = [r for r in recs if not r.get("error")]
    lat = [r["latency_s"] for r in ok]
    out_tok = [r["completion_tokens"] or 0 for r in ok]
    in_tok = [r["prompt_tokens"] or 0 for r in ok]
    return {"n": len(recs), "n_errors": len(recs) - len(ok), "wall_s": round(wall, 3),
            "latency_s": {"p50": float(np.percentile(lat, 50)), "p90": float(np.percentile(lat, 90)),
                          "p95": float(np.percentile(lat, 95)), "mean": float(np.mean(lat))} if lat else None,
            "output_tokens_mean": float(np.mean(out_tok)) if out_tok else None,
            "prompt_tokens_mean": float(np.mean(in_tok)) if in_tok else None,
            "requests_per_s": len(ok) / wall if wall else None,
            "output_tokens_per_s": sum(out_tok) / wall if wall else None,
            "input_tokens_per_s": sum(in_tok) / wall if wall else None}


async def _run(cond: dict[str, Any], items: list[dict[str, Any]], concurrency: int) -> dict[str, Any]:
    client = make_client(cond)
    sem = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()
    recs = await asyncio.gather(*[_one(client, cond, it, sem) for it in items])
    return _summ(recs, time.perf_counter() - t0)


def profile(cond: dict[str, Any], n_single: int = 30, n_batched: int = 150, concurrency: int = 32,
            split: str = "test_synth") -> dict[str, Any]:
    """Mixed workload: items cycle through the three synthetic tasks in a fixed order."""
    items: list[dict[str, Any]] = []
    per_task = [read_jsonl(REPO_ROOT / f"data/processed/eval/{split}/{t}.jsonl")
                for t in ("A_entities", "B_doctype", "C_structured")]
    for i in range(max(n_single, n_batched)):
        items.append(per_task[i % 3][i // 3])
    warm = asyncio.run(_run(cond, items[:3], 1))  # warm-up (CUDA graphs, prefix cache), discarded
    single = asyncio.run(_run(cond, items[:n_single], 1))
    batched = asyncio.run(_run(cond, items[:n_batched], concurrency))
    res = {"condition": cond["name"], "model": cond["served_model"], "split": split, "warmup_discarded": warm["n"],
           "single_stream": single, "batched": batched | {"concurrency": concurrency}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{cond['name']}.json").write_text(json.dumps(res, indent=1))
    return res
