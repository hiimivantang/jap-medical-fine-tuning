"""Generate synthetic Japanese clinical notes from sampled specs with a local OpenAI-compatible server.

Each spec is attempted up to `1 + retries` times (different sampling seed per attempt). Notes that fail
deterministic validation on every attempt are dropped and logged with their violation codes; nothing is
repaired by hand. Output: one JSONL record per spec with the accepted note (or the rejection reasons).
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from jmedft.datasets.synth_spec import (
    CATEGORIES,
    NoteSpec,
    clean_generated_text,
    generator_prompt,
    sample_spec,
    split_term_pools,
    validate_note,
)
from jmedft.utils import REPO_ROOT, load_yaml, write_jsonl


def build_specs(cfg: dict[str, Any], split: str) -> list[NoteSpec]:
    inv = load_yaml(REPO_ROOT / cfg["term_inventory"])
    pools = split_term_pools(inv, cfg["heldout_term_fraction"], cfg["seed"])
    scfg = cfg["synthetic"][split]
    rng = random.Random(f"{cfg['seed']}:{split}")
    return [
        sample_spec(f"{split}-{i:05d}", rng, pools, cfg["spec_sampling"], scfg["term_pool"],
                    cfg["synthetic"]["mixed_heldout_prob"], inv["family_relations"])
        for i in range(scfg["n_specs"])
    ]


async def _generate_one(client: AsyncOpenAI, gcfg: dict[str, Any], spec: NoteSpec, styles: dict[str, str],
                        all_terms: set[str], vcfg: dict[str, Any], retries: int, sem: asyncio.Semaphore,
                        base_seed: int) -> dict[str, Any]:
    prompt = generator_prompt(spec, styles)
    attempts: list[dict[str, Any]] = []
    for attempt in range(retries + 1):
        seed = (base_seed * 1000003 + hash_spec(spec.spec_id) + attempt) % (2**31 - 1)
        async with sem:
            t0 = time.perf_counter()
            try:
                resp = await client.chat.completions.create(
                    model=gcfg["name"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=gcfg["temperature"],
                    top_p=gcfg["top_p"],
                    max_tokens=gcfg["max_tokens"],
                    seed=seed,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
            except Exception as exc:  # noqa: BLE001 - network/server errors are recorded, not fatal
                attempts.append({"attempt": attempt, "error": f"{type(exc).__name__}: {exc}"[:300]})
                continue
            latency = time.perf_counter() - t0
        text = clean_generated_text(resp.choices[0].message.content or "")
        errs = validate_note(text, spec, all_terms, vcfg)
        attempts.append({"attempt": attempt, "seed": seed, "errors": errs, "latency_s": round(latency, 3),
                         "completion_tokens": resp.usage.completion_tokens if resp.usage else None})
        if not errs:
            return {"spec": spec.to_dict(), "text": text, "accepted": True, "generator": gcfg["name"],
                    "generator_revision": gcfg["revision"], "attempts": attempts}
    return {"spec": spec.to_dict(), "text": None, "accepted": False, "generator": gcfg["name"],
            "generator_revision": gcfg["revision"], "attempts": attempts}


def hash_spec(spec_id: str) -> int:
    # Stable across processes (str.__hash__ is salted per process).
    h = 0
    for ch in spec_id:
        h = (h * 131 + ord(ch)) % (2**31 - 1)
    return h


async def generate_split(cfg: dict[str, Any], split: str, out_path: Path, retries: int = 2,
                         limit: int | None = None) -> dict[str, Any]:
    scfg = cfg["synthetic"][split]
    gcfg = cfg["generators"][scfg["generator"]]
    retries = scfg.get("retries", retries)
    inv = load_yaml(REPO_ROOT / cfg["term_inventory"])
    styles = inv["doc_type_styles"]
    all_terms = {t for c in CATEGORIES for t in inv[c]}
    specs = build_specs(cfg, split)
    if limit:
        specs = specs[:limit]
    client = AsyncOpenAI(base_url=gcfg["base_url"], api_key="local", timeout=600, max_retries=2)
    sem = asyncio.Semaphore(gcfg["concurrency"])
    t0 = time.perf_counter()
    results = await asyncio.gather(*[
        _generate_one(client, gcfg, s, styles, all_terms, cfg["validation"], retries, sem, cfg["seed"])
        for s in specs
    ])
    elapsed = time.perf_counter() - t0
    write_jsonl(out_path, results)
    n_ok = sum(r["accepted"] for r in results)
    reasons: dict[str, int] = {}
    for r in results:
        if not r["accepted"]:
            for a in r["attempts"][-1:]:
                for e in a.get("errors", [a.get("error", "unknown")]):
                    key = e.split(":")[0]
                    reasons[key] = reasons.get(key, 0) + 1
    stats = {"split": split, "generator": gcfg["name"], "n_specs": len(specs), "n_accepted": n_ok,
             "acceptance_rate": round(n_ok / max(1, len(specs)), 4), "elapsed_s": round(elapsed, 1),
             "final_attempt_rejection_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1]))}
    out_path.with_suffix(".stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats
