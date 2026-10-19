"""Assemble the final train/val/test splits from generated synthetic notes and third-party sources.

Outputs (under cfg.output_dir = data/processed):
  sft/train.jsonl, sft/val.jsonl            chat-format SFT records (messages + assistant target)
  eval/<split>/<task>.jsonl                 evaluation items (docs/INTERFACES.md)
  leakage_report.json                       removed MedTxt docs, near-duplicate drops, overlap checks
  split_stats.json, SHA256SUMS              counts per split/task and file checksums

The builder never reads JMED-LLM or JGLUE files to create training data except to *exclude* overlapping
documents (enforced by tests/test_build.py).
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from jmedft.datasets.external import load_jglue, load_jmed_llm, medtxt_training_items
from jmedft.datasets.synth_spec import NoteSpec, gold_from_spec
from jmedft.tasks import build_messages, gold_to_target_json
from jmedft.utils import REPO_ROOT, char_ngrams, jaccard, read_jsonl, sha256_file, write_jsonl

SYNTH_TASKS = ["A_entities", "B_doctype", "C_structured"]


def synthetic_items(raw_path: Path, split: str) -> list[dict[str, Any]]:
    """One eval/SFT item per (accepted note, task)."""
    items = []
    for rec in read_jsonl(raw_path):
        if not rec["accepted"]:
            continue
        spec = NoteSpec(**rec["spec"])
        gold = gold_from_spec(spec)
        for task in SYNTH_TASKS:
            items.append({
                "id": f"{spec.spec_id}-{task}", "task": task, "split": split,
                "source": f"synthetic:{rec['generator']}", "messages": build_messages(task, text=rec["text"]),
                "gold": gold[task],
                "meta": {"source_text": rec["text"], "doc_type": spec.doc_type, "spec_id": spec.spec_id,
                         "unseen_terms": spec.unseen_terms, "generator": rec["generator"],
                         "generator_revision": rec["generator_revision"]},
            })
    return items


def to_sft(item: dict[str, Any]) -> dict[str, Any]:
    return {"id": item["id"], "task": item["task"], "source": item["source"],
            "messages": item["messages"] + [{"role": "assistant",
                                             "content": gold_to_target_json(item["task"], item["gold"])}]}


def near_dup_filter(test_items: list[dict[str, Any]], train_texts: list[str], threshold: float
                    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop test notes whose 5-gram Jaccard with any train note >= threshold (all tasks of that note)."""
    train_grams = [char_ngrams(t) for t in train_texts]
    by_doc: dict[str, float] = {}
    for it in test_items:
        doc = it["meta"]["spec_id"]
        if doc not in by_doc:
            g = char_ngrams(it["meta"]["source_text"])
            by_doc[doc] = max((jaccard(g, tg) for tg in train_grams), default=0.0)
    dropped = sorted(d for d, s in by_doc.items() if s >= threshold)
    kept = [it for it in test_items if it["meta"]["spec_id"] not in set(dropped)]
    log = [{"spec_id": d, "max_jaccard": round(by_doc[d], 3)} for d in dropped]
    return kept, log


def build(cfg: dict[str, Any]) -> dict[str, Any]:
    out = REPO_ROOT / cfg["output_dir"]
    syn_dir = REPO_ROOT / "data/synthetic"
    rng = random.Random(cfg["seed"])
    report: dict[str, Any] = {}

    # Synthetic splits.
    syn = {s: synthetic_items(syn_dir / f"{s}.raw.jsonl", s) for s in ["train", "val", "test_synth", "test_xgen"]
           if (syn_dir / f"{s}.raw.jsonl").exists()}
    train_texts = sorted({it["meta"]["source_text"] for it in syn.get("train", []) + syn.get("val", [])})
    thr = cfg["validation"]["near_dup_jaccard"]
    for s in ["test_synth", "test_xgen"]:
        if s in syn:
            syn[s], report[f"near_dup_dropped_{s}"] = near_dup_filter(syn[s], train_texts, thr)

    # Third-party evaluation sets.
    jmed = load_jmed_llm(REPO_ROOT / cfg["external_eval"]["jmed_llm_dir"])
    jglue = load_jglue(REPO_ROOT / cfg["external_eval"]["jglue_dir"])

    # Real-text training items (MedTxt), with JMED-LLM-overlapping documents removed.
    eval_texts = [it["meta"]["source_text"] for t in ["ext_mrner_disease", "ext_mrner_medicine", "ext_crade",
                                                      "ext_rrtnm"] for it in jmed[t]]
    rs = cfg["real_sources"]
    medtxt: list[dict[str, Any]] = []
    for kind, key in [("CR", "medtxt_cr_ja"), ("RR", "medtxt_rr_ja_test")]:
        items, removed = medtxt_training_items(REPO_ROOT / rs[key], kind, eval_texts,
                                               rs["jmed_llm_containment_threshold"])
        medtxt += items
        report[f"medtxt_{kind}_removed_for_overlap"] = removed
    report["medtxt_rr_training_file"] = "excluded entirely (same cases as JMED-LLM radiology items)"

    # SFT files: train = synthetic train + MedTxt; val = synthetic val. Shuffled with the fixed seed.
    sft_train = [to_sft(it) for it in syn.get("train", []) + medtxt]
    sft_val = [to_sft(it) for it in syn.get("val", [])]
    rng.shuffle(sft_train)
    write_jsonl(out / "sft/train.jsonl", sft_train)
    write_jsonl(out / "sft/val.jsonl", sft_val)

    # Evaluation files.
    eval_sets: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for s in ["val", "test_synth", "test_xgen"]:
        if s in syn:
            eval_sets[s] = {t: [it for it in syn[s] if it["task"] == t] for t in SYNTH_TASKS}
    eval_sets["test_ext"] = {t: jmed[t] for t in ["ext_mrner_disease", "ext_mrner_medicine", "ext_crade", "ext_rrtnm"]}
    eval_sets["test_general"] = {"gen_jcqa": jglue["gen_jcqa"], "gen_jnli": jglue["gen_jnli"],
                                 "gen_jmmlu_med": jmed["gen_jmmlu_med"]}
    for s, by_task in eval_sets.items():
        for t, items in by_task.items():
            write_jsonl(out / f"eval/{s}/{t}.jsonl", items)

    # Leakage self-checks on the written files.
    train_ids = {r["id"] for r in sft_train}
    eval_ids = {it["id"] for by_task in eval_sets.values() for items in by_task.values() for it in items
                if it["split"] != "val"}
    report["train_eval_id_overlap"] = sorted(train_ids & eval_ids)
    train_docs = {r["messages"][1]["content"] for r in sft_train}
    report["train_eval_prompt_overlap"] = sum(
        it["messages"][1]["content"] in train_docs
        for by_task in eval_sets.values() for items in by_task.values() for it in items if it["split"] != "val")
    (out / "leakage_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    stats = {
        "seed": cfg["seed"],
        "sft": {"train": len(sft_train), "val": len(sft_val),
                "train_by_task": dict(Counter(r["task"] for r in sft_train)),
                "train_by_source": dict(Counter(r["source"] for r in sft_train))},
        "eval": {s: {t: len(v) for t, v in by_task.items()} for s, by_task in eval_sets.items()},
        "near_dup_dropped": {s: len(report.get(f"near_dup_dropped_{s}", [])) for s in ["test_synth", "test_xgen"]},
        "medtxt_removed_for_overlap": {k: len(report[f"medtxt_{k}_removed_for_overlap"]) for k in ["CR", "RR"]},
        "train_eval_prompt_overlap": report["train_eval_prompt_overlap"],
    }
    (out / "split_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    files = sorted(p for p in out.rglob("*.jsonl"))
    (out / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.relative_to(out)}\n" for p in files))
    return stats
