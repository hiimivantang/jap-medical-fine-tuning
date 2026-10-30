"""Addendum 2026-10-30: base Qwen3-8B (E1 decoding) on the English edition of the evaluation sets (post hoc).

Runs the unchanged inference runner and scorer on data/processed/eval_en (written by translate_eval_en.py), and
writes results/predictions/E1en_qwen3_8b/... plus results/metrics/scores_en_addendum.json and
comparisons_en_addendum.json (English vs Japanese, same items, paired). No original result file is touched.

English-specific scoring adapters (the scorer itself is unchanged):
  * spans are compared case-insensitively: gold, predictions and the source text are case-folded before scoring;
  * closed vocabularies (document type, sex) are mapped back to the Japanese class names the scorer knows, in both
    gold and prediction, so "discharge summary" / "female" are not rejected as invalid.
Requires the vLLM qwen3 server (`jmedft serve --server qwen3`).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

EXP1 = Path(__file__).resolve().parents[3]
# EXP1/src first, so jmedft (and its REPO_ROOT, which is wrong in the installed copy) comes from the Exp1 tree
sys.path.insert(0, str(EXP1 / "src"))
sys.path.insert(1, str(EXP1 / "experiments/01_finetuning/scripts"))

from english_tasks import DOC_TYPES_EN, SEX_EN  # noqa: E402
from jmedft.evaluation import metrics, stats  # noqa: E402
from jmedft.evaluation.parsing import extract_json_object  # noqa: E402
from jmedft.evaluation.run import OUT, _ci, _paired, score_condition  # noqa: E402
from jmedft.inference.runner import make_client, run_items  # noqa: E402
from jmedft.utils import load_env, load_yaml, read_jsonl  # noqa: E402

EVAL_EN = EXP1 / "data/processed/eval_en"
PRED = EXP1 / "experiments/01_finetuning/results/predictions"
COND_JA, COND_EN = "E1_qwen3_8b", "E1en_qwen3_8b"
SPLITS = ("test_synth", "test_xgen", "test_ext")
DOC_JA = {v.casefold(): k for k, v in DOC_TYPES_EN.items()}
SEX_JA = {v.casefold(): k for k, v in SEX_EN.items()}


def _fold(v: Any) -> Any:
    if isinstance(v, str):
        return v.casefold()
    if isinstance(v, list):
        return [_fold(x) for x in v]
    if isinstance(v, dict):
        return {k: _fold(x) for k, x in v.items()}
    return v


def adapt_item(item: dict[str, Any]) -> dict[str, Any]:
    """English item -> what the Japanese scorer expects (closed vocab back to Japanese, spans case-folded)."""
    task, gold = item["task"], dict(item["gold"])
    if task == "B_doctype":
        gold = {"label": DOC_JA[gold["label"].casefold()]}
    elif task in ("ext_crade", "ext_rrtnm"):
        pass
    else:
        sex = gold.get("sex")
        gold = _fold(gold)
        if task == "C_structured":
            gold["sex"] = SEX_JA[sex.casefold()] if isinstance(sex, str) else sex
    meta = dict(item.get("meta") or {})
    meta["source_text"] = (meta.get("source_text") or "").casefold()
    return item | {"gold": gold, "meta": meta}


def adapt_prediction(task: str, pred: dict[str, Any]) -> dict[str, Any]:
    """Map an English answer onto the scorer's vocabulary; unparseable output is passed through unchanged."""
    if pred.get("error") or task in ("ext_crade", "ext_rrtnm"):
        return pred
    obj = extract_json_object(pred.get("output_text") or "")
    if obj is None:
        return pred
    if task == "B_doctype":
        label = obj.get("label")
        if isinstance(label, str):
            obj["label"] = DOC_JA.get(label.strip().casefold(), label)
    else:
        sex = obj.get("sex")
        obj = _fold(obj)
        if task == "C_structured" and isinstance(sex, str):
            obj["sex"] = SEX_JA.get(sex.strip().casefold(), sex)
    return pred | {"output_text": json.dumps(obj, ensure_ascii=False)}


def score_en() -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for split in SPLITS:
        for path in sorted((EVAL_EN / split).glob("*.jsonl")):
            task = path.stem
            preds_p = PRED / COND_EN / split / f"{task}.jsonl"
            if not preds_p.exists():
                continue
            items = [adapt_item(it) for it in read_jsonl(path)]
            preds = [adapt_prediction(task, p) for p in read_jsonl(preds_p)]
            res = metrics.score_task(task, items, preds)
            res["subsampled"] = True  # pair only on the items that survived translation
            point, lo, hi = _ci(task, res)
            res["primary_metric"] = metrics.primary_metric(task)
            res["primary"] = {"value": point, "ci95": [lo, hi]}
            out.setdefault(split, {})[task] = res
    return out


async def infer() -> None:
    load_env()
    conds = load_yaml(EXP1 / "experiments/01_finetuning/configs/conditions.yaml")
    cond = dict(conds["conditions"][COND_JA]) | {"name": COND_EN}
    cond["base_url"] = f"http://127.0.0.1:{conds['servers'][cond['server']]['port']}/v1"
    client = make_client(cond)
    for split in SPLITS:
        for path in sorted((EVAL_EN / split).glob("*.jsonl")):
            stats_ = await run_items(cond, read_jsonl(path), PRED / COND_EN / split / f"{path.stem}.jsonl", 32, client)
            print(json.dumps({"split": split, "task": path.stem} | stats_), flush=True)


def main() -> None:
    if "--score-only" not in sys.argv:
        asyncio.run(infer())
    en, ja = score_en(), score_condition(COND_JA)
    comps: dict[str, Any] = {}
    for split, tasks in en.items():
        for task, res_en in tasks.items():
            comps.setdefault(split, {})[task] = _paired(task, res_en, ja[split][task])
        adj = stats.holm({t: v["p_bootstrap"] for t, v in comps[split].items()})
        for t, v in comps[split].items():
            v["p_holm"] = adj[t]
    strip = {s: {t: {k: v for k, v in r.items() if k != "items"} for t, r in ts.items()} for s, ts in en.items()}
    (OUT / "scores_en_addendum.json").write_text(json.dumps({COND_EN: strip}, ensure_ascii=False, indent=1))
    (OUT / "comparisons_en_addendum.json").write_text(
        json.dumps({f"{COND_EN}__vs__{COND_JA}": comps}, ensure_ascii=False, indent=1))
    for split, tasks in en.items():
        for task, res in tasks.items():
            c = comps[split][task]
            print(f"{split:11s} {task:20s} EN {res['primary']['value']:.3f}  JA {ja[split][task]['primary']['value']:.3f}"
                  f"  EN-JA(paired n={c['n_paired']}) {c['diff']:+.3f} [{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}]"
                  f" holm p={c['p_holm']:.3g}")


if __name__ == "__main__":
    main()
