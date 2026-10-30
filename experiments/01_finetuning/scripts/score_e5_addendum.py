"""Addendum 2026-10-30: score E5_gpt61sol and its descriptive contrasts against the LoRA (E3) and GPT-4.1 (E4).

Post hoc, not pre-registered. Uses the evaluation code unchanged (`score_condition`, `_paired`, Holm per split),
but writes ONLY new files, so every original result file stays byte-for-byte:
  results/metrics/scores_e5_addendum.json       E5 scores (same shape as scores.json, per-item lists stripped)
  results/metrics/comparisons_e5_addendum.json  E3 - E5 and E4 - E5 paired contrasts (same shape as comparisons.json)
`jmedft evaluate` is deliberately not used: it rewrites scores/comparisons/hypotheses for every condition.

Run: .venv/bin/python experiments/01_finetuning/scripts/score_e5_addendum.py
"""

from __future__ import annotations

import json

from jmedft.evaluation import stats
from jmedft.evaluation.run import OUT, _paired, _strip_items, score_condition

E5 = "E5_gpt61sol"
CONTRASTS = [("E3_qwen3_8b_lora", E5), ("E4_gpt41", E5)]


def main() -> None:
    scores = {c: score_condition(c) for c in {E5} | {a for a, _ in CONTRASTS}}
    comps: dict = {}
    for a, b in CONTRASTS:
        key = f"{a}__vs__{b}"
        comps[key] = {}
        for split, tasks in scores[b].items():
            for task, res_b in tasks.items():
                res_a = scores[a].get(split, {}).get(task)
                if res_a is not None:
                    comps[key].setdefault(split, {})[task] = _paired(task, res_a, res_b)
        for split, tasks in comps[key].items():
            adj = stats.holm({t: v["p_bootstrap"] for t, v in tasks.items()})
            for t, v in tasks.items():
                v["p_holm"] = adj[t]
    (OUT / "scores_e5_addendum.json").write_text(
        json.dumps(_strip_items({E5: scores[E5]}), ensure_ascii=False, indent=1))
    (OUT / "comparisons_e5_addendum.json").write_text(json.dumps(comps, ensure_ascii=False, indent=1))
    for split, tasks in scores[E5].items():
        for task, res in tasks.items():
            print(f"{split:12s} {task:20s} {res['primary_metric']}={res['primary']['value']:.4f} "
                  f"missing={res['n_missing_predictions']}")


if __name__ == "__main__":
    main()
