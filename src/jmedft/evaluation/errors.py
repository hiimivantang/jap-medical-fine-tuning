"""Error taxonomy for entity and structured tasks (deterministic, rule-based; no LLM judge).

Per condition × split × task, every false positive / false negative is assigned one category:

  format_invalid_json     output had no parseable JSON object (all gold items become misses)
  format_schema           JSON parsed but required keys missing / wrong types (counted once per item)
  negation_error          predicted as affirmed although the gold marks it negated (A/C: in negated_symptoms)
  wrong_field             value exists in the gold under a different key (e.g. suspected → diagnoses)
  span_boundary           prediction and gold overlap by containment (longer/shorter span of the same entity)
  unsupported             predicted value not present in the source text at all (hallucination)
  unscripted_finding      predicted value IS in the source text but in no gold field (generator-added detail,
                          e.g. an imaging finding; an annotation-scope issue, not necessarily a model error)
  missed                  gold value not predicted and none of the above
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from jmedft import tasks
from jmedft.evaluation.parsing import parse_prediction
from jmedft.utils import normalize_text

ENTITY_KEYS = {"A_entities": tasks.A_KEYS}


def _norm_list(v: Any) -> list[str]:
    if isinstance(v, list):
        return [normalize_text(x) for x in v if isinstance(x, str) and normalize_text(x)]
    if isinstance(v, str) and normalize_text(v):
        return [normalize_text(v)]
    return []


def _all_gold_values(item: dict[str, Any]) -> dict[str, str]:
    """value -> gold field, over every gold field (for A, the C-style detail in meta is not available, so the
    negated/suspected/family context is reconstructed from the C gold of the same note when present)."""
    out: dict[str, str] = {}
    for k, v in item["gold"].items():
        for x in _norm_list(v):
            out.setdefault(x, k)
    for k, v in item.get("meta", {}).get("c_gold", {}).items():
        for x in _norm_list(v):
            out.setdefault(x, k)
    return out


def classify_item(task: str, item: dict[str, Any], output_text: str) -> Counter:
    keys = ENTITY_KEYS.get(task, tasks.C_LIST_FIELDS if task == "C_structured" else ["entities"])
    pp = parse_prediction(task, output_text or "")
    cats: Counter = Counter()
    if not pp.json_valid:
        cats["format_invalid_json"] += 1
    elif not pp.schema_valid:
        cats["format_schema"] += 1
    src = normalize_text(item["meta"]["source_text"])
    all_gold = _all_gold_values(item)
    negated = set(_norm_list(item.get("meta", {}).get("c_gold", {}).get("negated_symptoms")))
    if task == "C_structured":
        negated |= set(_norm_list(item["gold"].get("negated_symptoms")))
    for k in keys:
        gold = set(_norm_list(item["gold"].get(k)))
        pred = set(_norm_list(pp.value.get(k) if isinstance(pp.value, dict) else None))
        fps, fns = pred - gold, gold - pred
        matched_fn: set[str] = set()
        for p in sorted(fps):
            partner = next((g for g in sorted(fns) if g not in matched_fn and (g in p or p in g)), None)
            if partner is not None:
                cats["span_boundary"] += 1
                matched_fn.add(partner)
            elif p in negated and k != "negated_symptoms":
                cats["negation_error"] += 1
            elif p in all_gold and all_gold[p] != k:
                cats["wrong_field"] += 1
            elif p not in src:
                cats["unsupported"] += 1
            else:
                cats["unscripted_finding"] += 1
        for g in sorted(fns - matched_fn):
            if pp.json_valid and pp.schema_valid:
                cats["missed"] += 1
            else:
                cats["missed_due_to_format"] += 1
    return cats


def taxonomy(task: str, items: list[dict[str, Any]], preds: list[dict[str, Any]],
             c_gold_by_spec: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    by_id = {p["id"]: p for p in preds}
    total: Counter = Counter()
    for it in items:
        if c_gold_by_spec and it["meta"].get("spec_id") in c_gold_by_spec:
            it = {**it, "meta": {**it["meta"], "c_gold": c_gold_by_spec[it["meta"]["spec_id"]]}}
        p = by_id.get(it["id"])
        text = "" if p is None or p.get("error") else p.get("output_text", "")
        total += classify_item(task, it, text)
    return {"task": task, "n_items": len(items), "counts": dict(total.most_common())}
