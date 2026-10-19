"""Run evaluation items against an OpenAI-compatible endpoint (local vLLM or OpenAI) and record predictions.

One prediction record per item (docs/INTERFACES.md). Requests are issued concurrently for throughput; the
latency of every request is recorded individually. Re-running resumes: items whose prediction already
exists without error are skipped, so an interrupted run never re-bills or re-generates finished items.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from jmedft.evaluation.parsing import strip_reasoning
from jmedft.utils import read_jsonl, redact


def _request_kwargs(cond: dict[str, Any], messages: list[dict[str, str]]) -> dict[str, Any]:
    dec = cond["decoding"]
    kw: dict[str, Any] = {"model": cond["served_model"], "messages": messages}
    if cond["provider"] == "openai":
        if dec.get("max_completion_tokens"):
            kw["max_completion_tokens"] = dec["max_completion_tokens"]
        for k in ("temperature", "top_p", "seed", "reasoning_effort"):
            if dec.get(k) is not None:
                kw[k] = dec[k]
        return kw
    kw.update(temperature=dec["temperature"], top_p=dec.get("top_p", 1.0), max_tokens=dec["max_tokens"],
              seed=dec.get("seed", 0))
    extra: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": bool(dec.get("enable_thinking", False))}}
    for k in ("top_k", "repetition_penalty"):
        if dec.get(k) is not None:
            extra[k] = dec[k]
    kw["extra_body"] = extra
    return kw


async def _one(client: AsyncOpenAI, cond: dict[str, Any], item: dict[str, Any], sem: asyncio.Semaphore,
               max_attempts: int = 4) -> dict[str, Any]:
    rec: dict[str, Any] = {"id": item["id"], "task": item["task"], "condition": cond["name"],
                           "model": cond["served_model"]}
    kwargs = _request_kwargs(cond, item["messages"])
    async with sem:
        for attempt in range(max_attempts):
            t0 = time.perf_counter()
            try:
                resp = await client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - any API/server error is retried, then recorded
                err = redact(f"{type(exc).__name__}: {exc}")[:500]
                if attempt + 1 == max_attempts:
                    return rec | {"output_text": "", "reasoning_text": None, "finish_reason": None,
                                  "latency_s": None, "prompt_tokens": None, "completion_tokens": None,
                                  "error": err}
                await asyncio.sleep(2 ** attempt * 2)
                continue
            latency = time.perf_counter() - t0
            break
    choice = resp.choices[0]
    msg = choice.message
    raw = msg.content or ""
    server_reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
    answer, reasoning, unterminated = strip_reasoning(raw)
    usage = resp.usage
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    return rec | {
        "output_text": answer,
        "raw_text": raw if raw != answer else None,
        "reasoning_text": server_reasoning or reasoning,
        "reasoning_unterminated": unterminated,
        "finish_reason": choice.finish_reason,
        "latency_s": round(latency, 4),
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details else None,
        "error": None,
    }


def request_fingerprint(cond: dict[str, Any], item: dict[str, Any]) -> str:
    """Hash of everything that determines the output: served model, decoding and the exact messages."""
    payload = json.dumps({"model": cond["served_model"], "decoding": cond["decoding"], "messages": item["messages"]},
                         ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


async def run_items(cond: dict[str, Any], items: list[dict[str, Any]], out_path: Path, concurrency: int,
                    client: AsyncOpenAI) -> dict[str, Any]:
    """Resume-safe: an existing row is reused only if it has no error AND its request fingerprint matches the
    current item; rows for items outside `items` (e.g. a --limit run) are preserved, never deleted."""
    existing: dict[str, dict[str, Any]] = {}
    if out_path.exists():
        existing = {r["id"]: r for r in read_jsonl(out_path)}
    fps = {it["id"]: request_fingerprint(cond, it) for it in items}
    todo = [it for it in items
            if not (it["id"] in existing and existing[it["id"]].get("error") is None
                    and existing[it["id"]].get("request_fp") in (fps[it["id"]], None))]
    sem = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()
    new = await asyncio.gather(*[_one(client, cond, it, sem) for it in todo])
    wall = time.perf_counter() - t0
    for r in new:
        r["request_fp"] = fps[r["id"]]
    by_id = existing | {r["id"]: r for r in new}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    order = [it["id"] for it in items] + sorted(set(by_id) - {it["id"] for it in items})
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(by_id[i], ensure_ascii=False) + "\n" for i in order)
    errors = sum(1 for it in items if by_id.get(it["id"], {}).get("error"))
    return {"n_items": len(items), "n_new": len(todo), "n_errors": errors, "wall_s": round(wall, 2),
            "concurrency": concurrency}


def make_client(cond: dict[str, Any]) -> AsyncOpenAI:
    import os

    if cond["provider"] == "openai":
        return AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=300, max_retries=0)
    return AsyncOpenAI(base_url=cond["base_url"], api_key="local", timeout=1800, max_retries=0)


async def measure_single_stream_latency(cond: dict[str, Any], items: list[dict[str, Any]], n: int,
                                        client: AsyncOpenAI) -> list[dict[str, Any]]:
    """Sequential requests (concurrency 1) for an unbatched latency profile."""
    sem = asyncio.Semaphore(1)
    out = []
    for it in items[:n]:
        out.append(await _one(client, cond, it, sem))
    return out
