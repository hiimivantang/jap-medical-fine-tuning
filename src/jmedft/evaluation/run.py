"""Score every condition's predictions, compute CIs and paired comparisons, test the pre-registered
hypotheses and write machine-readable results.

Outputs (experiments/01_finetuning/results/metrics/):
  scores.json         per condition × split × task: full metric dict (+ 95% bootstrap CI of the primary metric)
  scores_long.csv     tidy table: condition, split, task, metric, value, ci_lo, ci_hi, n
  comparisons.json    paired bootstrap differences (+ McNemar for accuracy tasks) for the planned contrasts
  hypotheses.json     H1/H2/H3 decisions with the evidence used
  efficiency.json     latency and token statistics per condition
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from jmedft.evaluation import metrics, stats
from jmedft.tasks import TASK_FAMILY
from jmedft.utils import REPO_ROOT, read_jsonl

RESULTS = REPO_ROOT / "experiments/01_finetuning/results"
EVAL_DIR = REPO_ROOT / "data/processed/eval"
PRED_DIR = RESULTS / "predictions"
OUT = RESULTS / "metrics"

SPLITS = ["test_synth", "test_xgen", "test_ext", "test_general"]
SYNTH_TASKS = ["A_entities", "B_doctype", "C_structured"]
EXT_TASKS = ["ext_mrner_disease", "ext_mrner_medicine", "ext_crade", "ext_rrtnm"]
GEN_TASKS = ["gen_jcqa", "gen_jnli", "gen_jmmlu_med"]
H3_MARGIN = 0.02  # non-inferiority margin (2 accuracy points), pre-registered


def _available_conditions() -> list[str]:
    return sorted(p.name for p in PRED_DIR.iterdir() if p.is_dir()) if PRED_DIR.exists() else []


def _load(split: str, task: str, cond: str) -> tuple[list[dict], list[dict]] | None:
    items_p = EVAL_DIR / split / f"{task}.jsonl"
    pred_p = PRED_DIR / cond / split / f"{task}.jsonl"
    if not items_p.exists() or not pred_p.exists():
        return None
    return read_jsonl(items_p), read_jsonl(pred_p)


def _ci(task: str, result: dict[str, Any]) -> tuple[float, float, float]:
    data = metrics.bootstrap_data(task, result)
    return stats.bootstrap_ci(data, metrics.statistic_fn(task), n_resamples=10000, seed=0)


def score_condition(cond: str) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for split in SPLITS:
        for task_file in sorted((EVAL_DIR / split).glob("*.jsonl")):
            task = task_file.stem
            loaded = _load(split, task, cond)
            if loaded is None:
                continue
            items, preds = loaded
            res = metrics.score_task(task, items, preds)
            point, lo, hi = _ci(task, res)
            res["primary_metric"] = metrics.primary_metric(task)
            res["primary"] = {"value": point, "ci95": [lo, hi]}
            out.setdefault(split, {})[task] = res
    return out


def _paired(task: str, res_a: dict[str, Any], res_b: dict[str, Any]) -> dict[str, Any]:
    da, db = metrics.bootstrap_data(task, res_a), metrics.bootstrap_data(task, res_b)
    da, db = stats.align_by_id(da, db)
    diff, lo, hi, p = stats.paired_bootstrap_diff(da, db, metrics.statistic_fn(task), n_resamples=10000, seed=0)
    row: dict[str, Any] = {"diff": diff, "ci95": [lo, hi], "p_bootstrap": p,
                           "incomplete": (res_a["n_missing_predictions"] + res_b["n_missing_predictions"]) > 0}
    if TASK_FAMILY[task] in ("mc", "classification"):
        ca = [bool(x) for x in metrics.correctness(task, res_a)]
        cb = [bool(x) for x in metrics.correctness(task, res_b)]
        row["mcnemar"] = stats.mcnemar_exact(ca, cb)
    return row


CONTRASTS = [
    ("E3_qwen3_8b_lora", "E1_qwen3_8b"),        # H1
    ("E3_qwen3_8b_lora", "E2_swallow_8b"),      # H2
    ("E2_swallow_8b", "E1_qwen3_8b"),           # general Japanese adaptation effect
    ("E3_qwen3_8b_lora", "E2t_swallow_8b_think"),
    ("E3_qwen3_8b_lora", "E2m_medical_swallow_8b"),
    ("E3_qwen3_8b_lora", "E4_gpt41"),
    ("E3_qwen3_8b_lora", "E4b_gpt54mini"),
]


def compare(scores: dict[str, Any]) -> dict[str, Any]:
    comps: dict[str, Any] = {}
    for a, b in CONTRASTS:
        if a not in scores or b not in scores:
            continue
        key = f"{a}__vs__{b}"
        comps[key] = {}
        for split, tasks in scores[a].items():
            for task, res_a in tasks.items():
                res_b = scores[b].get(split, {}).get(task)
                if res_b is None:
                    continue
                comps[key].setdefault(split, {})[task] = _paired(task, res_a, res_b)
        # Holm correction across the tasks of each split (one family per contrast × split).
        for split, tasks in comps[key].items():
            adj = stats.holm({t: v["p_bootstrap"] for t, v in tasks.items()})
            for t, v in tasks.items():
                v["p_holm"] = adj[t]
    return comps


ALPHA = 0.05


def _sig_positive(c: dict[str, Any]) -> bool:
    """Significant improvement after Holm correction within the hypothesis family (pre-registered)."""
    return c["diff"] > 0 and c["p_holm"] < ALPHA


def _complete(c: dict[str, Any]) -> bool:
    return not c.get("incomplete", False)


def _pooled_contrast(scores: dict[str, Any], a: str, b: str, task: str) -> dict[str, Any] | None:
    """Paired bootstrap of A−B on the concatenation of test_synth and test_xgen items (pre-registered H2)."""
    parts = []
    for split in ("test_synth", "test_xgen"):
        ra, rb = scores.get(a, {}).get(split, {}).get(task), scores.get(b, {}).get(split, {}).get(task)
        if ra is None or rb is None:
            return None
        da, db = stats.align_by_id(metrics.bootstrap_data(task, ra), metrics.bootstrap_data(task, rb))
        parts.append((da, db, ra["n_missing_predictions"] + rb["n_missing_predictions"]))
    da = _concat([p[0] for p in parts])
    db = _concat([p[1] for p in parts])
    diff, lo, hi, p = stats.paired_bootstrap_diff(da, db, metrics.statistic_fn(task), n_resamples=10000, seed=0)
    return {"diff": diff, "ci95": [lo, hi], "p_bootstrap": p, "n": len(da["ids"]),
            "incomplete": sum(x[2] for x in parts) > 0}


def _concat(datas: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"ids": [i for d in datas for i in d["ids"]]}
    for k, v in datas[0].items():
        if k == "ids":
            continue
        out[k] = np.concatenate([d[k] for d in datas]) if isinstance(v, np.ndarray) else v
    return out


def hypotheses(comps: dict[str, Any], scores: dict[str, Any] | None = None) -> dict[str, Any]:
    h: dict[str, Any] = {}
    k1 = "E3_qwen3_8b_lora__vs__E1_qwen3_8b"
    if k1 in comps:
        synth = comps[k1].get("test_synth", {})
        xgen = comps[k1].get("test_xgen", {})
        present = all(t in synth and t in xgen for t in SYNTH_TASKS)
        complete = present and all(_complete(synth[t]) and _complete(xgen[t]) for t in SYNTH_TASKS)
        sig = [t for t in SYNTH_TASKS if t in synth and _sig_positive(synth[t])]
        same_dir = present and bool(sig) and all(xgen[t]["diff"] > 0 for t in sig)
        ext = comps[k1].get("test_ext", {})
        h["H1"] = {
            "rule": "E3-E1 > 0 with Holm-adjusted p < 0.05 (within test_synth's 3 primary endpoints) on >= 2 of 3 "
                    "AND every such endpoint has the same (positive) direction on test_xgen",
            "all_endpoints_present": present,
            "complete_predictions": complete,
            "significant_test_synth": sig,
            "same_direction_test_xgen": same_dir,
            "supported": bool(complete and len(sig) >= 2 and same_dir),
            "external_transfer": {t: ext[t] for t in EXT_TASKS if t in ext},
            "evidence": {"test_synth": synth, "test_xgen": xgen},
        }
    k2 = "E3_qwen3_8b_lora__vs__E2_swallow_8b"
    if k2 in comps and scores is not None:
        pooled = {t: _pooled_contrast(scores, "E3_qwen3_8b_lora", "E2_swallow_8b", t) for t in SYNTH_TASKS}
        present = all(v is not None for v in pooled.values())
        if present:
            adj = stats.holm({t: v["p_bootstrap"] for t, v in pooled.items()})
            for t, v in pooled.items():
                v["p_holm"] = adj[t]
        complete = present and all(_complete(v) for v in pooled.values())
        sig = [t for t, v in pooled.items() if v is not None and _sig_positive(v)]
        h["H2"] = {
            "rule": "E3-E2 > 0 with Holm-adjusted p < 0.05 on >= 2 of 3 primary endpoints, test_synth and test_xgen "
                    "pooled per task (paired bootstrap over the concatenated items)",
            "all_endpoints_present": present,
            "complete_predictions": complete,
            "pooled": pooled,
            "significant_pooled": sig,
            "supported": bool(complete and len(sig) >= 2),
            "per_split_evidence": {s: comps[k2].get(s, {}) for s in ("test_synth", "test_xgen", "test_ext")},
        }
    if k1 in comps:
        gen = comps[k1].get("test_general", {})
        present = all(t in gen for t in GEN_TASKS)
        per_task = {t: {"diff": gen[t]["diff"], "ci95": gen[t]["ci95"], "non_inferior": gen[t]["ci95"][0] > -H3_MARGIN,
                        "complete": _complete(gen[t])} for t in GEN_TASKS if t in gen}
        h["H3"] = {
            "rule": f"for EACH of {GEN_TASKS}, lower 95% CI bound of (E3-E1) > -{H3_MARGIN} (non-inferiority, 2 pp)",
            "all_tasks_present": present,
            "per_task": per_task,
            "supported": bool(present and all(v["non_inferior"] and v["complete"] for v in per_task.values())),
        }
    return h


def efficiency(cond: str) -> dict[str, Any]:
    lat, ptok, ctok, rtok = [], [], [], []
    for p in (PRED_DIR / cond).rglob("*.jsonl"):
        for r in read_jsonl(p):
            if r.get("error"):
                continue
            if r.get("latency_s") is not None:
                lat.append(r["latency_s"])
            if r.get("prompt_tokens") is not None:
                ptok.append(r["prompt_tokens"])
            if r.get("completion_tokens") is not None:
                ctok.append(r["completion_tokens"])
            if r.get("reasoning_tokens"):
                rtok.append(r["reasoning_tokens"])

    def q(a: list[float], x: float) -> float | None:
        return float(np.percentile(a, x)) if a else None

    return {"n_requests": len(lat), "latency_s_concurrent": {"p50": q(lat, 50), "p95": q(lat, 95), "mean": float(np.mean(lat)) if lat else None},
            "prompt_tokens_mean": float(np.mean(ptok)) if ptok else None,
            "completion_tokens": {"mean": float(np.mean(ctok)) if ctok else None, "p95": q(ctok, 95),
                                  "total": int(np.sum(ctok)) if ctok else 0},
            "prompt_tokens_total": int(np.sum(ptok)) if ptok else 0,
            "reasoning_tokens_total": int(np.sum(rtok)) if rtok else 0}


def _long_rows(scores: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for cond, splits in scores.items():
        for split, tasks in splits.items():
            for task, res in tasks.items():
                prim = res["primary"]
                rows.append({"condition": cond, "split": split, "task": task, "metric": res["primary_metric"],
                             "value": prim["value"], "ci_lo": prim["ci95"][0], "ci_hi": prim["ci95"][1], "n": res["n"]})
                for k, v in res.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("n", res["primary_metric"]):
                        rows.append({"condition": cond, "split": split, "task": task, "metric": k, "value": v,
                                     "ci_lo": None, "ci_hi": None, "n": res["n"]})
    return rows


def _strip_items(scores: dict[str, Any]) -> dict[str, Any]:
    """Drop bulky per-item records from the JSON summary (they are written separately)."""
    slim: dict[str, Any] = {}
    for cond, splits in scores.items():
        for split, tasks in splits.items():
            for task, res in tasks.items():
                slim.setdefault(cond, {}).setdefault(split, {})[task] = {k: v for k, v in res.items() if k != "items"}
    return slim


def error_taxonomies(conds: list[str]) -> dict[str, Any]:
    from jmedft.evaluation.errors import taxonomy

    out: dict[str, Any] = {}
    for split in ("test_synth", "test_xgen", "test_ext"):
        c_items = EVAL_DIR / split / "C_structured.jsonl"
        c_gold = {it["meta"]["spec_id"]: it["gold"] for it in read_jsonl(c_items)} if c_items.exists() else None
        for task in ("A_entities", "C_structured", "ext_mrner_disease", "ext_mrner_medicine"):
            items_p = EVAL_DIR / split / f"{task}.jsonl"
            if not items_p.exists():
                continue
            items = read_jsonl(items_p)
            for cond in conds:
                pred_p = PRED_DIR / cond / split / f"{task}.jsonl"
                if pred_p.exists():
                    out.setdefault(cond, {}).setdefault(split, {})[task] = taxonomy(
                        task, items, read_jsonl(pred_p), c_gold if task == "A_entities" else None)
    return out


def evaluate_all(conditions: list[str] | None = None) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    conds = conditions or _available_conditions()
    scores = {c: score_condition(c) for c in conds}
    comps = compare(scores)
    hyp = hypotheses(comps, scores)
    eff = {c: efficiency(c) for c in conds}
    (OUT / "error_taxonomy.json").write_text(json.dumps(error_taxonomies(conds), ensure_ascii=False, indent=1))
    (OUT / "scores.json").write_text(json.dumps(_strip_items(scores), ensure_ascii=False, indent=1))
    (OUT / "comparisons.json").write_text(json.dumps(comps, ensure_ascii=False, indent=1))
    (OUT / "hypotheses.json").write_text(json.dumps(hyp, ensure_ascii=False, indent=1))
    (OUT / "efficiency.json").write_text(json.dumps(eff, ensure_ascii=False, indent=1))
    items_dir = OUT / "per_item"
    for cond, splits in scores.items():
        for split, tasks in splits.items():
            for task, res in tasks.items():
                p = items_dir / cond / split / f"{task}.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(res.get("items", []), ensure_ascii=False))
    rows = _long_rows(scores)
    with open(OUT / "scores_long.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["condition"])
        w.writeheader()
        w.writerows(rows)
    return {"conditions": conds, "hypotheses": {k: v.get("supported") for k, v in hyp.items()}}


def main() -> None:  # pragma: no cover
    print(json.dumps(evaluate_all(), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["Path", "evaluate_all"]
