"""Score parsed predictions per task family (docs/INTERFACES.md, "Scoring").

`score_task` returns aggregate metrics plus per-item records; `bootstrap_data` and `statistic_fn` expose the
per-item quantities and the primary metric in the form `jmedft.evaluation.stats` resamples.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from jmedft import tasks
from jmedft.evaluation.parsing import ParsedPrediction, parse_prediction
from jmedft.utils import normalize_text

INVALID = "INVALID"
N_FIELDS = len(tasks.C_FIELDS)
_PRIMARY = {"entities": "micro_f1", "classification": "macro_f1", "structured": "field_accuracy",
            "mc": "accuracy"}


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, F1; each is 0.0 when undefined."""
    return tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0, _f1(tp, fp, fn)


def _mean(xs: list[Any]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _norm_set(values: Any) -> set[str]:
    """Normalised, de-duplicated non-empty strings of a list (anything else -> empty set)."""
    if not isinstance(values, list):
        return set()
    return {n for v in values if isinstance(v, str) and (n := normalize_text(v))}


def _norm_scalar(value: Any) -> Any:
    """Strings are normalised (empty -> None); ints and None are compared as they are."""
    return (normalize_text(value) or None) if isinstance(value, str) else value


def score_task(task: str, items: list[dict], predictions: list[dict]) -> dict[str, Any]:
    """Parse and score `predictions` against `items`, matched by id. A missing prediction or one with an
    `error` is parsed as "" (invalid/empty answer) and counted in `n_missing_predictions`."""
    family = tasks.TASK_FAMILY[task]
    by_id = {p["id"]: p for p in predictions}
    parsed: list[ParsedPrediction] = []
    n_missing = 0
    for item in items:
        pred = by_id.get(item["id"])
        text = ""
        if pred is None or pred.get("error"):
            n_missing += 1
        else:
            text = pred.get("output_text") or ""
        labels = item["gold"].get("labels") if family == "mc" else None
        parsed.append(parse_prediction(task, text, labels))
    result: dict[str, Any] = {
        "task": task,
        "family": family,
        "n": len(items),
        "n_missing_predictions": n_missing,
        "json_valid_rate": _mean([p.json_valid for p in parsed]),
        "whole_json_rate": _mean([p.whole_json for p in parsed]),
        "schema_valid_rate": _mean([p.schema_valid for p in parsed]),
    }
    scorer = {"entities": _score_entities, "classification": _score_classification,
              "structured": _score_structured, "mc": _score_mc}[family]
    result.update(scorer(task, items, parsed))
    return result


def lenient_match_count(gold: set[str], pred: set[str], min_len: int = 2) -> int:
    """Maximum one-to-one matching where one normalised string contains the other and the shorter one has at
    least `min_len` characters (exact matches always count). Secondary metric: tolerant of span-boundary
    conventions, not of wrong entities."""
    from scipy.optimize import linear_sum_assignment

    exact = gold & pred
    g_rest, p_rest = sorted(gold - exact), sorted(pred - exact)
    if not g_rest or not p_rest:
        return len(exact)
    ok = np.array([[(min(len(g), len(p)) >= min_len and (g in p or p in g)) for p in p_rest] for g in g_rest])
    if not ok.any():
        return len(exact)
    rows, cols = linear_sum_assignment(-ok.astype(float))
    return len(exact) + int(ok[rows, cols].sum())


def _score_entities(task: str, items: list[dict], parsed: list[ParsedPrediction]) -> dict[str, Any]:
    keys = tasks.A_KEYS if task == "A_entities" else ["entities"]
    totals = {k: [0, 0, 0] for k in keys}  # tp, fp, fn per key
    lenient = [0, 0, 0]
    records = []
    for item, pp in zip(items, parsed):
        rec: dict[str, Any] = {"id": item["id"], "tp": 0, "fp": 0, "fn": 0, "per_key": {}, "exact": True}
        for k in keys:
            gold, pred = _norm_set(item["gold"].get(k)), _norm_set(pp.value.get(k))
            fp_values, fn_values = sorted(pred - gold), sorted(gold - pred)
            counts = (len(gold & pred), len(fp_values), len(fn_values))
            rec["per_key"][k] = {"tp": counts[0], "fp": counts[1], "fn": counts[2],
                                 "fp_values": fp_values, "fn_values": fn_values}
            for i, name in enumerate(("tp", "fp", "fn")):
                totals[k][i] += counts[i]
                rec[name] += counts[i]
            rec["exact"] = rec["exact"] and gold == pred
            ltp = lenient_match_count(gold, pred)
            lenient[0] += ltp
            lenient[1] += len(pred) - ltp
            lenient[2] += len(gold) - ltp
        records.append(rec)
    tp, fp, fn = (sum(t[i] for t in totals.values()) for i in range(3))
    precision, recall, f1 = _prf(tp, fp, fn)
    per_key = {}
    for k, (ktp, kfp, kfn) in totals.items():
        kp, kr, kf = _prf(ktp, kfp, kfn)
        per_key[k] = {"precision": kp, "recall": kr, "f1": kf, "support": ktp + kfn}
    lp, lr, lf = _prf(*lenient)
    return {"micro_precision": precision, "micro_recall": recall, "micro_f1": f1, "per_key": per_key,
            "lenient_precision": lp, "lenient_recall": lr, "lenient_f1": lf,
            "exact_match": _mean([r["exact"] for r in records]), "items": records}


def _label_metrics(gold: list[str], pred: list[str], labels: list[str]) -> dict[str, Any]:
    """Macro-F1 and per-class P/R/F1 over `labels` (sklearn, zero_division=0) and the confusion matrix
    (rows = gold labels, cols = labels + INVALID). Invalid answers are INVALID in `pred`, i.e. errors."""
    cols = labels + [INVALID]
    col = {lab: j for j, lab in enumerate(cols)}
    row = {lab: i for i, lab in enumerate(labels)}
    cm = [[0] * len(cols) for _ in labels]
    for g, p in zip(gold, pred):
        if g in row:
            cm[row[g]][col.get(p, col[INVALID])] += 1
    macro_f1 = 0.0
    per_class = {lab: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0} for lab in labels}
    if gold and labels:
        macro_f1 = float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0))
        prec, rec, f1, sup = precision_recall_fscore_support(gold, pred, labels=labels, zero_division=0)
        per_class = {lab: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]),
                           "support": int(sup[i])} for i, lab in enumerate(labels)}
    return {"macro_f1": macro_f1, "per_class": per_class, "confusion_matrix": cm, "labels": list(labels)}


def _score_classification(task: str, items: list[dict], parsed: list[ParsedPrediction]) -> dict[str, Any]:
    labels = list(tasks.DOC_TYPES)
    gold = [item["gold"]["label"] for item in items]
    pred = [pp.value if pp.value in labels else None for pp in parsed]
    records = [{"id": item["id"], "gold": g, "pred": p, "correct": p == g}
               for item, g, p in zip(items, gold, pred)]
    return {"accuracy": _mean([r["correct"] for r in records]),
            **_label_metrics(gold, [p or INVALID for p in pred], labels),
            "invalid_rate": _mean([p is None for p in pred]), "items": records}


def _score_mc(task: str, items: list[dict], parsed: list[ParsedPrediction]) -> dict[str, Any]:
    label_sets = [item["gold"].get("labels") or [] for item in items]
    labels = sorted(set().union(*label_sets, [item["gold"]["answer"] for item in items]))
    records = []
    for item, pp in zip(items, parsed):
        gold = item["gold"]["answer"]
        records.append({"id": item["id"], "gold": gold, "pred": pp.answer_letter,
                        "strict_pred": pp.strict_answer_letter, "correct": pp.answer_letter == gold,
                        "strict_correct": pp.strict_answer_letter == gold})
    return {"accuracy": _mean([r["correct"] for r in records]),
            "strict_accuracy": _mean([r["strict_correct"] for r in records]),
            "invalid_rate": _mean([r["pred"] is None for r in records]),
            **_label_metrics([r["gold"] for r in records], [r["pred"] or INVALID for r in records], labels),
            "items": records}


def _score_structured(task: str, items: list[dict], parsed: list[ParsedPrediction]) -> dict[str, Any]:
    list_totals = {f: [0, 0, 0] for f in tasks.C_LIST_FIELDS}  # tp, fp, fn per list field
    records = []
    for item, pp in zip(items, parsed):
        gold, value = item["gold"], pp.value
        source = normalize_text((item.get("meta") or {}).get("source_text") or "")
        correct = {f: _norm_scalar(gold.get(f)) == _norm_scalar(value.get(f)) for f in tasks.C_SCALAR_FIELDS}
        complaint = _norm_scalar(value.get("chief_complaint"))
        # Values checked against the source text: free-text scalar (sex is a closed vocabulary) + list values.
        checked = [complaint] if complaint is not None else []
        counts = [0, 0, 0]
        for f in tasks.C_LIST_FIELDS:
            g, p = _norm_set(gold.get(f)), _norm_set(value.get(f))
            correct[f] = g == p
            for i, c in enumerate((len(g & p), len(p - g), len(g - p))):
                list_totals[f][i] += c
                counts[i] += c
            checked.extend(sorted(p))
        n_correct = sum(correct.values())
        records.append({"id": item["id"], "fields_correct": n_correct, "field_correct": correct,
                        "list_tp": counts[0], "list_fp": counts[1], "list_fn": counts[2],
                        "missing_fields": list(pp.missing_fields),
                        "unsupported_values": [v for v in checked if v not in source],
                        "n_predicted_values": len(checked), "exact": n_correct == N_FIELDS})
    n = len(records)
    tp, fp, fn = (sum(t[i] for t in list_totals.values()) for i in range(3))
    precision, recall, f1 = _prf(tp, fp, fn)
    missing_total = sum(len(r["missing_fields"]) for r in records)
    unsupported_total = sum(len(r["unsupported_values"]) for r in records)
    n_values = sum(r["n_predicted_values"] for r in records)
    return {
        "field_accuracy": sum(r["fields_correct"] for r in records) / (N_FIELDS * n) if n else 0.0,
        "per_field_accuracy": {f: _mean([r["field_correct"][f] for r in records]) for f in tasks.C_FIELDS},
        "exact_match": _mean([r["exact"] for r in records]),
        "list_micro_precision": precision,
        "list_micro_recall": recall,
        "list_micro_f1": f1,
        "per_list_field_f1": {f: _f1(*t) for f, t in list_totals.items()},
        "missing_fields_total": missing_total,
        "missing_fields_rate": missing_total / (N_FIELDS * n) if n else 0.0,
        "unsupported_total": unsupported_total,
        "unsupported_rate": unsupported_total / max(1, n_values),
        "items": records,
    }


def primary_metric(task: str) -> str:
    return _PRIMARY[tasks.TASK_FAMILY[task]]


def bootstrap_data(task: str, result: dict[str, Any]) -> dict[str, Any]:
    """Per-item arrays (aligned with `ids`) from which `statistic_fn(task)` recomputes the primary metric."""
    family = tasks.TASK_FAMILY[task]
    records = result["items"]
    data: dict[str, Any] = {"ids": [r["id"] for r in records]}
    if family == "entities":
        for k in ("tp", "fp", "fn"):
            data[k] = np.array([r[k] for r in records], dtype=np.int64)
    elif family == "classification":
        index = {lab: i for i, lab in enumerate(tasks.DOC_TYPES)}
        data["gold"] = np.array([index.get(r["gold"], -1) for r in records], dtype=np.int64)
        data["pred"] = np.array([index.get(r["pred"], -1) for r in records], dtype=np.int64)
        data["n_labels"] = len(tasks.DOC_TYPES)
    elif family == "mc":
        data["correct"] = np.array([r["correct"] for r in records], dtype=bool)
    else:
        data["fields_correct"] = np.array([r["fields_correct"] for r in records], dtype=np.int64)
        data["n_fields"] = N_FIELDS
    return data


def _rows(data: dict[str, Any], key: str, idx: np.ndarray | None) -> np.ndarray:
    arr = np.asarray(data[key])
    return arr if idx is None else arr[idx]


def _micro_f1_stat(data: dict[str, Any], idx: np.ndarray | None) -> float:
    tp, fp, fn = (int(_rows(data, k, idx).sum()) for k in ("tp", "fp", "fn"))
    return _f1(tp, fp, fn)


def _macro_f1_stat(data: dict[str, Any], idx: np.ndarray | None) -> float:
    """Macro-F1 over classes 0..n_labels-1; -1 (invalid) is never a class, so it only adds FN."""
    gold, pred = _rows(data, "gold", idx), _rows(data, "pred", idx)
    scores = []
    for c in range(int(data["n_labels"])):
        tp = int(np.sum((gold == c) & (pred == c)))
        fp = int(np.sum((gold != c) & (pred == c)))
        fn = int(np.sum((gold == c) & (pred != c)))
        scores.append(_f1(tp, fp, fn))
    return float(np.mean(scores)) if scores else 0.0


def _accuracy_stat(data: dict[str, Any], idx: np.ndarray | None) -> float:
    correct = _rows(data, "correct", idx)
    return float(correct.mean()) if correct.size else 0.0


def _field_accuracy_stat(data: dict[str, Any], idx: np.ndarray | None) -> float:
    fields_correct = _rows(data, "fields_correct", idx)
    if not fields_correct.size:
        return 0.0
    return float(fields_correct.sum() / (data["n_fields"] * fields_correct.size))


def statistic_fn(task: str) -> Callable[[dict[str, Any], np.ndarray | None], float]:
    """Primary metric over the rows `idx` of `bootstrap_data(task, ...)` (None = all rows)."""
    return {"entities": _micro_f1_stat, "classification": _macro_f1_stat, "mc": _accuracy_stat,
            "structured": _field_accuracy_stat}[tasks.TASK_FAMILY[task]]


def correctness(task: str, result: dict[str, Any]) -> list[bool]:
    """Per-item correctness (lenient for MC) for McNemar; only defined for mc and classification tasks."""
    if tasks.TASK_FAMILY[task] not in ("mc", "classification"):
        raise ValueError(f"correctness is undefined for {task!r} ({tasks.TASK_FAMILY[task]})")
    return [bool(r["correct"]) for r in result["items"]]
