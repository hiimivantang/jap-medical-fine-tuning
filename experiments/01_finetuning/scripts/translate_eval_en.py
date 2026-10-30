"""Addendum 2026-10-30: translate the Experiment 01 evaluation sets to English with gpt-6.1-sol (post hoc).

Scope: test_synth, test_xgen (A/B/C share one note per spec_id; each note is translated once) and test_ext
(MRNER disease/medicine, CRADE, RRTNM). Each request returns the English document AND an English rendering of
every gold span, so labels and text use the same wording. Validation is mechanical: every English span must
occur verbatim (case-insensitive, NFKC, whitespace-insensitive, as scored) in the English document; an item that
fails is retried once, then excluded and listed — never scored against a broken key.

Cost control: every request is reserved at its upper bound against a $15 ceiling before it is sent (the fail-closed
BudgetGuard from Experiment 02, own ledger, authorisation recorded below) and settled from reported usage. Resume-safe:
translations already in the cache file are reused, so an interrupted run never re-bills.

Writes data/processed/eval_en/<split>/<task>.jsonl (git-ignored like data/processed/eval) and
experiments/01_finetuning/results/translation_en/{report.json, cache.jsonl}.

Run with the Experiment 02 venv (it provides judging.budget):
  ../jap-medical-question-answer/.venv/bin/python experiments/01_finetuning/scripts/translate_eval_en.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

EXP1 = Path(__file__).resolve().parents[3]
# EXP1/src first, so jmedft (and its REPO_ROOT, which is wrong in the installed copy) comes from the Exp1 tree
sys.path.insert(0, str(EXP1 / "src"))
sys.path.insert(1, str(EXP1 / "experiments/01_finetuning/scripts"))
sys.path.insert(2, str(EXP1.parent / "jap-medical-question-answer/src"))

from english_tasks import build_messages, gold_to_english  # noqa: E402
from judging.budget import BudgetError, BudgetGuard, PriceRegistry  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

MODEL, EFFORT, MAX_OUT = "gpt-6.1-sol", "low", 4096
EVAL, OUT = EXP1 / "data/processed/eval", EXP1 / "data/processed/eval_en"
RES = EXP1 / "experiments/01_finetuning/results/translation_en"
CACHE = RES / "cache.jsonl"
AUTH = {"amount_usd": 15, "scope": ["translation"], "granted_by": "user (Claude Code session, 2026-10-30)",
        "granted_utc": "2026-10-30T00:00:00Z"}
SYNTH = ("test_synth", "test_xgen")
SYNTH_TASKS = ("A_entities", "B_doctype", "C_structured")
EXT_TASKS = ("ext_mrner_disease", "ext_mrner_medicine", "ext_crade", "ext_rrtnm")
_WS = re.compile(r"\s+")

TRANSLATOR = """You translate Japanese clinical text into natural, precise clinical English for an evaluation dataset.
Return only a JSON object with the keys described below. Rules:
- Translate faithfully and completely; keep every finding, negation ("no ...", "not observed"), hedge ("suspected",
  "possible"), number, date, unit, laterality and drug name. Do not add, drop, summarise or explain anything.
- Use standard English clinical terminology. Keep Latin-script abbreviations as they are (e.g. CT, NSAIDs).
- "spans" maps each listed Japanese phrase to the exact English wording you used for it in the translated text.
  Every English span MUST appear verbatim in your translated text, with the same spelling. Translate each phrase
  consistently everywhere it occurs. If a phrase occurs inside a longer expression, choose a translation in which
  the phrase's English rendering is still a contiguous substring of the text."""


def norm(s: str) -> str:
    """Scoring normalisation for English: NFKC, case-folded, whitespace removed."""
    return _WS.sub("", unicodedata.normalize("NFKC", s)).casefold()


def units() -> list[dict[str, Any]]:
    """One translation unit per synthetic note and per external item, with the gold spans that must map."""
    out = []
    for split in SYNTH:
        notes: dict[str, dict[str, Any]] = {}
        for task in SYNTH_TASKS:
            for it in map(json.loads, open(EVAL / split / f"{task}.jsonl", encoding="utf-8")):
                n = notes.setdefault(it["meta"]["spec_id"], {"key": f"{split}:{it['meta']['spec_id']}",
                                                             "kind": "note", "text": it["meta"]["source_text"],
                                                             "spans": set(), "items": []})
                n["items"].append(it)
                if task != "B_doctype":  # A and C gold values are spans copied from the note
                    for v in it["gold"].values():
                        for s in (v if isinstance(v, list) else [v]):
                            if isinstance(s, str):
                                n["spans"].add(s)
        out += notes.values()
    for task in EXT_TASKS:
        for it in map(json.loads, open(EVAL / "test_ext" / f"{task}.jsonl", encoding="utf-8")):
            q = it["messages"][1]["content"]
            unit = {"key": f"test_ext:{it['id']}", "kind": task, "text": q, "items": [it],
                    "spans": set(it["gold"].get("entities", []))}
            out.append(unit)
    for u in out:  # closed vocabularies are mapped, never translated
        u["spans"] -= {"男性", "女性"}
        u["spans"] = sorted(u["spans"])
    return out


def request_for(u: dict[str, Any]) -> list[dict[str, str]]:
    if u["kind"] == "note":
        task = ("Translate the clinical document below. Return {\"text\": <English document>, \"spans\": "
                "{<Japanese phrase>: <English phrase>}}.\nPhrases to map:\n" + json.dumps(u["spans"], ensure_ascii=False)
                + "\n\nDocument:\n" + u["text"])
    elif u["kind"] in ("ext_mrner_disease", "ext_mrner_medicine"):
        task = ("The text below is a task prompt: a Japanese question followed by a case report in 「」 and answer "
                "instructions. Translate ONLY the question sentence and the case report (drop the trailing answer "
                "instructions). Return {\"question\": <English question sentence>, \"text\": <English case report>, "
                "\"spans\": {<Japanese phrase>: <English phrase>}}.\nPhrases to map:\n"
                + json.dumps(u["spans"], ensure_ascii=False) + "\n\nTask prompt:\n" + u["text"])
    else:
        task = ("The text below is a multiple-choice task prompt: a Japanese question, a case or radiology report in "
                "「」, and lettered options. Translate the question sentence, the report and every option text "
                "(keep the option order; do not include the letters). Drop the trailing answer instructions. "
                "Return {\"question\": <English question sentence>, \"text\": <English report>, "
                "\"options\": [<English option text>, ...], \"spans\": {}}.\n\nTask prompt:\n" + u["text"])
    return [{"role": "system", "content": TRANSLATOR}, {"role": "user", "content": task}]


def validate(u: dict[str, Any], tr: dict[str, Any]) -> list[str]:
    """Problems with a translation (empty list = usable)."""
    errs = []
    text = tr.get("text")
    if not isinstance(text, str) or not text.strip():
        return ["no text"]
    spans = tr.get("spans") or {}
    for s in u["spans"]:
        en = spans.get(s)
        if not isinstance(en, str) or not en.strip():
            errs.append(f"span not mapped: {s}")
        elif norm(en) not in norm(text):
            errs.append(f"span not in text: {s} -> {en}")
    if u["kind"] in ("ext_crade", "ext_rrtnm"):
        n_opts = len(u["items"][0]["gold"]["labels"])
        if not isinstance(tr.get("options"), list) or len(tr["options"]) != n_opts:
            errs.append(f"options: expected {n_opts}")
    if u["kind"] != "note" and not isinstance(tr.get("question"), str):
        errs.append("no question")
    return errs


def build_items(u: dict[str, Any], tr: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for it in u["items"]:
        task = it["task"]
        if u["kind"] == "note":
            msgs = build_messages(task, text=tr["text"])
        elif task in ("ext_mrner_disease", "ext_mrner_medicine"):
            msgs = build_messages(task, question=f"{tr['question']}\n\"{tr['text']}\"")
        else:
            msgs = build_messages(task, question=f"{tr['question']}\n\"{tr['text']}\"", options=tr["options"])
        gold = gold_to_english(task, it["gold"], tr.get("spans") or {})
        meta = dict(it["meta"]) | {"source_text": tr["text"], "source_text_ja": it["meta"].get("source_text"),
                                   "translated_by": MODEL}
        out.append(it | {"messages": msgs, "gold": gold, "meta": meta, "lang": "en"})
    return out


async def translate(client: AsyncOpenAI, guard: BudgetGuard, prices: PriceRegistry, u: dict[str, Any],
                    sem: asyncio.Semaphore) -> dict[str, Any]:
    from judging.providers import estimate_input_tokens

    msgs = request_for(u)
    async with sem:
        for attempt in (1, 2):
            est = estimate_input_tokens(msgs[0]["content"] + msgs[1]["content"])
            try:
                res = guard.reserve("translation", MODEL, "standard", est, MAX_OUT, purpose=f"translate:{u['key']}")
            except BudgetError as exc:
                return {"key": u["key"], "ok": False, "errors": [f"refused: {exc}"], "attempts": attempt - 1}
            try:
                resp = await client.chat.completions.create(model=MODEL, messages=msgs, reasoning_effort=EFFORT,
                                                            max_completion_tokens=MAX_OUT,
                                                            response_format={"type": "json_object"})
            except Exception as exc:  # noqa: BLE001 - unknown outcome: charged at the bound, item excluded
                guard.fail(res, f"{type(exc).__name__}")
                return {"key": u["key"], "ok": False, "errors": [f"provider error: {type(exc).__name__}"],
                        "attempts": attempt}
            usage = resp.usage
            details = getattr(usage, "prompt_tokens_details", None)
            guard.settle(res, {"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens,
                               "input_tokens_details": {"cached_tokens": getattr(details, "cached_tokens", 0) or 0,
                                                        "cache_write_tokens": 0},
                               "output_tokens_details": {"reasoning_tokens": getattr(
                                   getattr(usage, "completion_tokens_details", None), "reasoning_tokens", 0) or 0}},
                         resp.model, None)
            try:
                tr = json.loads(resp.choices[0].message.content or "")
            except json.JSONDecodeError:
                tr = {}
            errs = validate(u, tr)
            if not errs:
                return {"key": u["key"], "ok": True, "translation": tr, "attempts": attempt}
            if attempt == 1:  # one retry, with the problems spelled out
                msgs = msgs + [{"role": "assistant", "content": resp.choices[0].message.content or ""},
                               {"role": "user", "content": "Fix these problems and return the full JSON again: "
                                                           + "; ".join(errs[:10])}]
        return {"key": u["key"], "ok": False, "errors": errs, "attempts": 2}


async def main() -> None:
    from jmedft.utils import load_env

    load_env()
    RES.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(json.dumps(__import__("yaml").safe_load(
        open(EXP1.parent / "jap-medical-question-answer/experiments/02_medgen/configs/pricing.yaml"))))
    prices = PriceRegistry(cfg)
    guard = BudgetGuard(prices, {"translation": 15, "total": 15}, RES / "ledger.jsonl", authorization=AUTH)
    us = units()
    done = {}
    if CACHE.exists():
        done = {r["key"]: r for r in map(json.loads, open(CACHE, encoding="utf-8")) if r["ok"]}
    todo = [u for u in us if u["key"] not in done]
    print(f"units {len(us)}, cached {len(done)}, to translate {len(todo)}")
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=300, max_retries=0)
    sem = asyncio.Semaphore(16)
    try:
        results = await asyncio.gather(*(translate(client, guard, prices, u, sem) for u in todo))
    finally:
        await client.close()
    with open(CACHE, "a", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in results if r["ok"])
    done |= {r["key"]: r for r in results if r["ok"]}
    failed = [r for r in results if not r["ok"]]
    by_file: dict[tuple[str, str], list[dict[str, Any]]] = {}
    excluded = []
    for u in us:
        if u["key"] not in done:
            excluded += [it["id"] for it in u["items"]]
            continue
        for it in build_items(u, done[u["key"]]["translation"]):
            by_file.setdefault((it["split"], it["task"]), []).append(it)
    for (split, task), rows in by_file.items():
        p = OUT / split / f"{task}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in sorted(rows, key=lambda r: r["id"]))
    report = {"model": MODEL, "effort": EFFORT, "units": len(us), "translated": len(done),
              "failed_units": [{"key": r["key"], "errors": r["errors"][:5]} for r in failed],
              "excluded_item_ids": sorted(excluded),
              "items_per_file": {f"{s}/{t}": len(v) for (s, t), v in sorted(by_file.items())},
              "spend": guard.summary()}
    (RES / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: report[k] for k in ("units", "translated", "items_per_file")}, ensure_ascii=False))
    print("failed units:", len(failed), "| excluded items:", len(excluded))
    print("spend:", report["spend"]["spent_usd"], "| usage_incomplete:", report["spend"]["usage_incomplete"])


if __name__ == "__main__":
    asyncio.run(main())
