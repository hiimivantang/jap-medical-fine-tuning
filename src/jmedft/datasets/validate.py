"""Integrity and leakage validation of built datasets (`jmedft data validate`).

Checks:
  * every eval/SFT record is well-formed (required keys, roles, gold matches the task schema);
  * SFT assistant targets parse as JSON and match the task schema;
  * synthetic gold strings occur verbatim (after normalisation) in the source note;
  * no test document text appears in training (exact prompt and 5-gram near-duplicate checks);
  * no JMED-LLM evaluation document overlaps a training document above the configured threshold;
  * test_synth/test_xgen contain held-out (unseen) terms;
  * SHA256SUMS match the files on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

from jmedft.tasks import TASK_FAMILY, schema_for
from jmedft.utils import REPO_ROOT, char_ngrams, containment, jaccard, normalize_text, read_jsonl, sha256_file


def _gold_as_output(task: str, gold: dict[str, Any]) -> dict[str, Any]:
    if TASK_FAMILY[task] == "mc":
        return {"answer": gold["answer"]}
    return gold


def validate_all(cfg: dict[str, Any]) -> dict[str, Any]:
    out = REPO_ROOT / cfg["output_dir"]
    problems: list[str] = []
    counts: dict[str, int] = {}

    # SHA256SUMS
    sums = (out / "SHA256SUMS").read_text().splitlines()
    for line in sums:
        digest, rel = line.split("  ", 1)
        if sha256_file(out / rel) != digest:
            problems.append(f"checksum mismatch: {rel}")

    # SFT records
    train = read_jsonl(out / "sft/train.jsonl")
    val = read_jsonl(out / "sft/val.jsonl")
    for name, rows in (("train", train), ("val", val)):
        for r in rows:
            roles = [m["role"] for m in r["messages"]]
            if roles != ["system", "user", "assistant"]:
                problems.append(f"sft {name} {r['id']}: roles {roles}")
                continue
            try:
                obj = json.loads(r["messages"][2]["content"])
                jsonschema.validate(obj, schema_for(r["task"]))
            except Exception as exc:  # noqa: BLE001 - report every malformed target
                problems.append(f"sft {name} {r['id']}: bad target ({type(exc).__name__})")
        counts[f"sft_{name}"] = len(rows)

    # Eval records
    eval_items: dict[str, list[dict[str, Any]]] = {}
    for path in sorted((out / "eval").rglob("*.jsonl")):
        split = path.parent.name
        rows = read_jsonl(path)
        eval_items.setdefault(split, []).extend(rows)
        counts[f"eval_{split}_{path.stem}"] = len(rows)
        for it in rows:
            try:
                jsonschema.validate(_gold_as_output(it["task"], it["gold"]), schema_for(it["task"]))
            except jsonschema.ValidationError as exc:
                problems.append(f"eval {it['id']}: gold violates schema ({exc.message[:80]})")
            if it["source"].startswith("synthetic"):
                src = normalize_text(it["meta"]["source_text"])
                vals = []
                for k, v in it["gold"].items():
                    if isinstance(v, list):
                        vals += v
                    elif isinstance(v, str) and k not in ("label", "sex"):
                        vals.append(v)
                missing = [v for v in vals if normalize_text(v) not in src]
                if missing:
                    problems.append(f"eval {it['id']}: gold not in text {missing[:3]}")
            elif TASK_FAMILY[it["task"]] == "entities":
                src = normalize_text(it["meta"]["source_text"])
                missing = [v for v in it["gold"]["entities"] if normalize_text(v) not in src]
                if missing:
                    problems.append(f"eval {it['id']}: external gold not in text {missing[:3]}")

    # Leakage: exact prompts and near-duplicate source documents between training and any test split.
    train_prompts = {r["messages"][1]["content"] for r in train}
    train_docs = sorted({r["messages"][1]["content"].split("文書:\n", 1)[-1] for r in train})
    train_grams = [char_ngrams(t) for t in train_docs]
    thr = cfg["validation"]["near_dup_jaccard"]
    for split, rows in eval_items.items():
        if split == "val":
            continue
        exact = sum(it["messages"][1]["content"] in train_prompts for it in rows)
        if exact:
            problems.append(f"{split}: {exact} eval prompts identical to training prompts")
        docs = {it["meta"]["source_text"] for it in rows}
        worst = 0.0
        n_dup = 0
        for d in docs:
            g = char_ngrams(d)
            if split in ("test_ext",):
                m = max((containment(g, tg) for tg in train_grams), default=0.0)
                limit = cfg["real_sources"]["jmed_llm_containment_threshold"]
            else:
                m = max((jaccard(g, tg) for tg in train_grams), default=0.0)
                limit = thr
            worst = max(worst, m)
            n_dup += m >= limit
        counts[f"{split}_max_overlap_x1000"] = int(worst * 1000)
        if n_dup:
            problems.append(f"{split}: {n_dup} documents overlap training above threshold")

    # Held-out terms present in the synthetic test splits.
    for split in ("test_synth", "test_xgen"):
        rows = [it for it in eval_items.get(split, []) if it["task"] == "A_entities"]
        if rows:
            share = sum(bool(it["meta"]["unseen_terms"]) for it in rows) / len(rows)
            counts[f"{split}_share_docs_with_unseen_terms_x1000"] = int(share * 1000)
            if share < 0.3:
                problems.append(f"{split}: only {share:.0%} of notes contain held-out terms")

    return {"ok": not problems, "n_problems": len(problems), "problems": problems[:50], "counts": counts}


def main() -> None:  # pragma: no cover - convenience entry
    from jmedft.utils import load_yaml

    cfg = load_yaml(Path(REPO_ROOT / "experiments/01_finetuning/configs/data.yaml"))
    print(json.dumps(validate_all(cfg), ensure_ascii=False, indent=2))
