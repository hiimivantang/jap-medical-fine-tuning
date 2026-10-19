"""Charts for the experiment report (matplotlib, static PNG).

Colour assignment is fixed per condition (same hue in every chart). Okabe–Ito palette (colour-blind safe).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONDITION_STYLE: dict[str, dict[str, str]] = {
    "E1_qwen3_8b": {"label": "E1 Qwen3-8B", "color": "#999999"},
    "E2_swallow_8b": {"label": "E2 Swallow-8B", "color": "#56B4E9"},
    "E2t_swallow_8b_think": {"label": "E2t Swallow-8B (think)", "color": "#0072B2"},
    "E2m_medical_swallow_8b": {"label": "E2m Medical-Swallow-8B", "color": "#009E73"},
    "E3_qwen3_8b_lora": {"label": "E3 Qwen3-8B + LoRA", "color": "#D55E00"},
    "E4_gpt41": {"label": "E4 GPT-4.1", "color": "#CC79A7"},
    "E4b_gpt54mini": {"label": "E4b GPT-5.4-mini", "color": "#E69F00"},
}

TASK_LABELS = {
    "A_entities": "A: entity F1",
    "B_doctype": "B: doc-type macro-F1",
    "C_structured": "C: field accuracy",
    "ext_mrner_disease": "MRNER-disease F1",
    "ext_mrner_medicine": "MRNER-medicine F1",
    "ext_crade": "CRADE acc.",
    "ext_rrtnm": "RRTNM acc.",
    "gen_jcqa": "JCommonsenseQA acc.",
    "gen_jnli": "JNLI acc.",
    "gen_jmmlu_med": "JMMLU-Med acc.",
}

plt.rcParams.update({
    "font.family": ["Noto Sans CJK JP", "IPAexGothic", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "font.size": 10,
})


def _style(cond: str) -> dict[str, str]:
    return CONDITION_STYLE.get(cond, {"label": cond, "color": "#444444"})


def grouped_bars(scores: dict[str, Any], split: str, tasks: list[str], conditions: list[str], title: str,
                 out: Path) -> Path | None:
    """Primary metric with 95% bootstrap CI per task (groups) × condition (bars)."""
    present = [t for t in tasks if any(t in scores.get(c, {}).get(split, {}) for c in conditions)]
    conds = [c for c in conditions if c in scores and split in scores[c]]
    if not present or not conds:
        return None
    width = 0.8 / len(conds)
    x = np.arange(len(present))
    fig, ax = plt.subplots(figsize=(max(6.5, 2.1 * len(present)), 4.2))
    for i, c in enumerate(conds):
        vals, lo, hi = [], [], []
        for t in present:
            r = scores[c][split].get(t)
            if r is None:
                vals.append(np.nan)
                lo.append(0)
                hi.append(0)
                continue
            v = r["primary"]["value"]
            vals.append(v)
            lo.append(v - r["primary"]["ci95"][0])
            hi.append(r["primary"]["ci95"][1] - v)
        st = _style(c)
        ax.bar(x + (i - (len(conds) - 1) / 2) * width, vals, width * 0.92, yerr=[lo, hi], capsize=2,
               color=st["color"], label=st["label"], error_kw={"elinewidth": 0.8, "ecolor": "#333333"})
    ax.set_xticks(x, [TASK_LABELS.get(t, t) for t in present])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score (95% bootstrap CI)")
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=min(4, len(conds)), loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def diff_forest(comps: dict[str, Any], contrast: str, splits: list[str], title: str, out: Path,
                margin: float | None = None) -> Path | None:
    """Paired differences (A − B) with 95% CI per split × task."""
    rows = []
    for split in splits:
        for task, v in comps.get(contrast, {}).get(split, {}).items():
            rows.append((f"{split}: {TASK_LABELS.get(task, task)}", v["diff"], v["ci95"][0], v["ci95"][1]))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 0.38 * len(rows) + 1.2))
    y = np.arange(len(rows))[::-1]
    for yi, (_, d, lo, hi) in zip(y, rows):
        color = "#D55E00" if lo > 0 else ("#0072B2" if hi < 0 else "#666666")
        ax.plot([lo, hi], [yi, yi], color=color, lw=1.6)
        ax.plot(d, yi, "o", color=color, ms=5)
    ax.axvline(0, color="#333333", lw=0.8)
    if margin is not None:
        ax.axvline(-margin, color="#333333", lw=0.8, ls="--")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax.set_xlabel("difference in primary metric (95% paired bootstrap CI)")
    ax.set_title(title, loc="left", fontsize=11)
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def confusion_matrix_plot(cm: list[list[int]], labels: list[str], pred_labels: list[str], title: str,
                          out: Path) -> Path:
    arr = np.array(cm, dtype=float)
    norm = arr / np.maximum(arr.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.0 + 0.85 * len(pred_labels), 1.0 + 0.7 * len(labels)))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j]:
                ax.text(j, i, int(arr[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.6 else "#222222")
    ax.set_xticks(range(len(pred_labels)), pred_labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def latency_tokens_plot(eff: dict[str, Any], conditions: list[str], out: Path) -> Path | None:
    conds = [c for c in conditions if c in eff and eff[c]["n_requests"]]
    if not conds:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    labels = [_style(c)["label"] for c in conds]
    colors = [_style(c)["color"] for c in conds]
    p50 = [eff[c]["latency_s_concurrent"]["p50"] for c in conds]
    p95 = [eff[c]["latency_s_concurrent"]["p95"] for c in conds]
    y = np.arange(len(conds))
    axes[0].barh(y, p95, color=colors, alpha=0.35, label="p95")
    axes[0].barh(y, p50, color=colors, label="p50")
    axes[0].set_yticks(y, labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("latency per request (s), batched serving")
    axes[0].legend(frameon=False, fontsize=8)
    toks = [eff[c]["completion_tokens"]["mean"] or 0 for c in conds]
    axes[1].barh(y, toks, color=colors)
    axes[1].set_yticks(y, [""] * len(conds))
    axes[1].invert_yaxis()
    axes[1].set_xlabel("mean output tokens per request")
    for ax in axes:
        ax.grid(axis="y", alpha=0)
    fig.suptitle("Latency and output length (all evaluation requests)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def training_curve(log_history: list[dict[str, Any]], out: Path) -> Path | None:
    tr = [(r["step"], r["loss"]) for r in log_history if "loss" in r and "eval_loss" not in r]
    ev = [(r["step"], r["eval_loss"]) for r in log_history if "eval_loss" in r]
    if not tr:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(*zip(*tr), color="#D55E00", lw=1.2, label="train loss")
    if ev:
        ax.plot(*zip(*ev), "o-", color="#0072B2", lw=1.2, ms=4, label="validation loss")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy (assistant tokens)")
    ax.set_title("LoRA SFT loss (Qwen3-8B)", loc="left", fontsize=11)
    ax.legend(frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
