"""Inference cost model from MEASURED token counts and throughput (no assumed numbers except list prices).

  * OpenAI: per-token list price × measured mean prompt/completion(+reasoning) tokens per request.
  * Self-hosted (DGX Spark) and Fireworks dedicated GPU: $/hour ÷ measured batched requests/hour.
    The Spark has no list $/hour; we report requests/hour and the break-even $/hour against the OpenAI
    reference instead of inventing a hardware cost. Fireworks dedicated-GPU throughput is NOT measured here
    (different hardware); we only state its list $/hour and the throughput it would need to match.
"""

from __future__ import annotations

import json
from typing import Any

from jmedft.utils import REPO_ROOT, load_yaml

METRICS = REPO_ROOT / "experiments/01_finetuning/results/metrics"


def openai_cost_per_1k_requests(prices: dict[str, float], prompt_tok: float, completion_tok: float) -> float:
    """Uncached list price; completion tokens include any reasoning tokens (billed as output)."""
    return 1000 * (prompt_tok * prices["input_per_mtok"] + completion_tok * prices["output_per_mtok"]) / 1e6


def cost_table() -> dict[str, Any]:
    pricing = load_yaml(REPO_ROOT / "experiments/01_finetuning/configs/pricing.yaml")
    conds = load_yaml(REPO_ROOT / "experiments/01_finetuning/configs/conditions.yaml")["conditions"]
    lat_dir = METRICS / "latency"
    rows: dict[str, Any] = {"price_snapshot": pricing["retrieved"], "rows": {}}
    ref = None
    for path in sorted(lat_dir.glob("*.json")) if lat_dir.exists() else []:
        prof = json.loads(path.read_text())
        name = prof["condition"]
        b = prof["batched"]
        row: dict[str, Any] = {"model": prof["model"],
                               "prompt_tokens_mean": b["prompt_tokens_mean"],
                               "output_tokens_mean": b["output_tokens_mean"],
                               "single_stream_p50_s": prof["single_stream"]["latency_s"]["p50"],
                               "single_stream_p95_s": prof["single_stream"]["latency_s"]["p95"],
                               "batched_requests_per_hour": b["requests_per_s"] * 3600,
                               "batched_output_tokens_per_s": b["output_tokens_per_s"],
                               "batched_concurrency": b["concurrency"]}
        if conds[name]["provider"] == "openai":
            prices = pricing["openai"][prof["model"]]
            row["usd_per_1k_requests"] = openai_cost_per_1k_requests(prices, b["prompt_tokens_mean"],
                                                                     b["output_tokens_mean"])
            if name == "E4_gpt41":
                ref = row
        rows["rows"][name] = row
    if ref:
        for name, row in rows["rows"].items():
            if "usd_per_1k_requests" not in row:
                # Hourly cost at which this self-hosted deployment matches the OpenAI reference per request.
                row["break_even_usd_per_hour_vs_E4"] = ref["usd_per_1k_requests"] * row["batched_requests_per_hour"] / 1000
                fw = pricing["fireworks"]["on_demand_gpu_per_hour"]
                row["fireworks_required_requests_per_hour_to_match_E4"] = {
                    gpu: 1000 * usd / ref["usd_per_1k_requests"] for gpu, usd in fw.items()}
    (METRICS / "costs.json").write_text(json.dumps(rows, indent=1))
    return rows
