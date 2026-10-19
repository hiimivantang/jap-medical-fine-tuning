"""Generate charts, results tables, error-analysis samples and the experiment manifest from the metrics files.

The narrative report (REPORT.md) and the customer summary are written by hand from these generated
artifacts; this module produces everything machine-derived so the numbers in the documents can be traced.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from typing import Any

from jmedft.reporting import charts
from jmedft.tasks import DOC_TYPES, TASK_FAMILY
from jmedft.utils import REPO_ROOT, load_yaml, read_jsonl, sha256_file

RESULTS = REPO_ROOT / "experiments/01_finetuning/results"
METRICS = RESULTS / "metrics"
FIG = RESULTS / "figures"
CFG_DIR = REPO_ROOT / "experiments/01_finetuning/configs"

ORDER = ["E1_qwen3_8b", "E2_swallow_8b", "E2t_swallow_8b_think", "E2m_medical_swallow_8b", "E3_qwen3_8b_lora",
         "E4_gpt41", "E4b_gpt54mini"]
SYNTH = ["A_entities", "B_doctype", "C_structured"]
EXT = ["ext_mrner_disease", "ext_mrner_medicine", "ext_crade", "ext_rrtnm"]
GEN = ["gen_jcqa", "gen_jnli", "gen_jmmlu_med"]


def _fmt(v: float | None, pct: bool = True) -> str:
    if v is None:
        return "–"
    return f"{100 * v:.1f}" if pct else f"{v:.3f}"


def results_table(scores: dict[str, Any], split: str, tasks: list[str]) -> str:
    conds = [c for c in ORDER if c in scores and split in scores[c]]
    head = "| Condition | " + " | ".join(charts.TASK_LABELS.get(t, t) for t in tasks) + " |"
    sep = "|---|" + "---|" * len(tasks)
    lines = [head, sep]
    for c in conds:
        cells = []
        for t in tasks:
            r = scores[c][split].get(t)
            if r is None:
                cells.append("–")
                continue
            v, (lo, hi) = r["primary"]["value"], r["primary"]["ci95"]
            cells.append(f"{_fmt(v)} [{_fmt(lo)}–{_fmt(hi)}]")
        lines.append(f"| {charts.CONDITION_STYLE.get(c, {'label': c})['label']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def secondary_table(scores: dict[str, Any], split: str, task: str, keys: list[str]) -> str:
    conds = [c for c in ORDER if c in scores and task in scores[c].get(split, {})]
    lines = ["| Condition | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for c in conds:
        r = scores[c][split][task]
        cells = []
        for k in keys:
            v = r.get(k)
            if isinstance(v, float):
                cells.append(_fmt(v) if v <= 1.0 else f"{v:.2f}")
            elif v is None:
                cells.append("–")
            else:
                cells.append(str(v))
        lines.append(f"| {charts.CONDITION_STYLE.get(c, {'label': c})['label']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def comparison_table(comps: dict[str, Any], contrast: str, splits: list[str]) -> str:
    lines = ["| Split | Task | Δ (pp) | 95% CI (pp) | p (bootstrap) | p (Holm) | McNemar p |", "|---|---|---|---|---|---|---|"]
    for split in splits:
        for task, v in comps.get(contrast, {}).get(split, {}).items():
            mc = v.get("mcnemar", {}).get("p_value")
            lines.append(f"| {split} | {charts.TASK_LABELS.get(task, task)} | {100 * v['diff']:+.1f} | "
                         f"[{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}] | {v['p_bootstrap']:.4f} | "
                         f"{v['p_holm']:.4f} | {(f'{mc:.4f}') if mc is not None else '–'} |")
    return "\n".join(lines)


def make_figures(scores: dict[str, Any], comps: dict[str, Any], eff: dict[str, Any]) -> list[str]:
    made: list[str] = []
    for split, tasks, title in [
        ("test_synth", SYNTH, "Synthetic held-out test (Qwen3.8-generated notes)"),
        ("test_xgen", SYNTH, "Cross-generator test (Nemotron-generated notes)"),
        ("test_ext", EXT, "External human-annotated benchmark (JMED-LLM)"),
        ("test_general", GEN, "General Japanese / general medical knowledge (H3)"),
    ]:
        p = charts.grouped_bars(scores, split, tasks, ORDER, title, FIG / f"bars_{split}.png")
        if p:
            made.append(str(p.relative_to(RESULTS)))
    for contrast, title, margin in [
        ("E3_qwen3_8b_lora__vs__E1_qwen3_8b", "E3 (LoRA) − E1 (base): H1 / H3", 0.02),
        ("E3_qwen3_8b_lora__vs__E2_swallow_8b", "E3 (LoRA) − E2 (Swallow): H2", None),
        ("E3_qwen3_8b_lora__vs__E4_gpt41", "E3 (LoRA) − E4 (GPT-4.1)", None),
    ]:
        p = charts.diff_forest(comps, contrast, ["test_synth", "test_xgen", "test_ext", "test_general"], title,
                               FIG / f"diff_{contrast}.png", margin=margin)
        if p:
            made.append(str(p.relative_to(RESULTS)))
    for cond in ORDER:
        for split in ("test_synth", "test_xgen"):
            r = scores.get(cond, {}).get(split, {}).get("B_doctype")
            if r and "confusion_matrix" in r:
                p = charts.confusion_matrix_plot(r["confusion_matrix"], DOC_TYPES, DOC_TYPES + ["INVALID"],
                                                 f"Task B confusion — {charts.CONDITION_STYLE[cond]['label']} ({split})",
                                                 FIG / f"cm_B_{cond}_{split}.png")
                made.append(str(p.relative_to(RESULTS)))
        for task in ("ext_crade", "ext_rrtnm"):
            r = scores.get(cond, {}).get("test_ext", {}).get(task)
            if r and "confusion_matrix" in r:
                labels = r.get("labels") or list("ABCDE")
                p = charts.confusion_matrix_plot(r["confusion_matrix"], labels, labels + ["INVALID"],
                                                 f"{charts.TASK_LABELS[task]} — {charts.CONDITION_STYLE[cond]['label']}",
                                                 FIG / f"cm_{task}_{cond}.png")
                made.append(str(p.relative_to(RESULTS)))
    p = charts.latency_tokens_plot(eff, ORDER, FIG / "latency_tokens.png")
    if p:
        made.append(str(p.relative_to(RESULTS)))
    state = RESULTS / "adapters/qwen3-8b-jmed-lora/trainer_state.json"
    if state.exists():
        p = charts.training_curve(json.loads(state.read_text())["log_history"], FIG / "training_loss.png")
        if p:
            made.append(str(p.relative_to(RESULTS)))
    return made


def error_examples(cond: str, split: str, task: str, k: int = 5) -> list[dict[str, Any]]:
    """Concrete failure cases (deterministic order) for manual error analysis."""
    per_item = METRICS / "per_item" / cond / split / f"{task}.json"
    preds = RESULTS / "predictions" / cond / split / f"{task}.jsonl"
    items = REPO_ROOT / "data/processed/eval" / split / f"{task}.jsonl"
    if not per_item.exists():
        return []
    rec = json.loads(per_item.read_text())
    pred_by_id = {r["id"]: r for r in read_jsonl(preds)}
    item_by_id = {r["id"]: r for r in read_jsonl(items)}
    out = []
    for r in rec:
        wrong = (r.get("fp") or r.get("fn")) if TASK_FAMILY[task] == "entities" else not r.get("correct", r.get("exact", True))
        if not wrong:
            continue
        it = item_by_id[r["id"]]
        out.append({"id": r["id"], "source_excerpt": it["meta"]["source_text"][:240], "gold": it["gold"],
                    "output": pred_by_id.get(r["id"], {}).get("output_text", "")[:400], "score_record": r})
        if len(out) >= k:
            break
    return out


def _pkg_versions(names: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def _lf_versions() -> dict[str, Any]:
    py = REPO_ROOT / ".venv-llamafactory/bin/python"
    if not py.exists():
        return {}
    code = ("import json,importlib.metadata as m;print(json.dumps({n:m.version(n) for n in "
            "['llamafactory','torch','transformers','peft','trl','accelerate','datasets']}))")
    try:
        return json.loads(subprocess.run([str(py), "-c", code], capture_output=True, text=True, check=True).stdout)
    except Exception:  # noqa: BLE001
        return {}


def build_manifest() -> dict[str, Any]:
    data_cfg = load_yaml(CFG_DIR / "data.yaml")
    conds = load_yaml(CFG_DIR / "conditions.yaml")
    train_cfg = load_yaml(CFG_DIR / "train_lora.yaml")
    processed = REPO_ROOT / "data/processed"
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, check=False).stdout.strip()
    manifest: dict[str, Any] = {
        "experiment": "01_finetuning",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "hardware": {"platform": platform.platform(), "machine": platform.machine(), "gpu": smi,
                     "system": "NVIDIA DGX Spark (GB10 Grace Blackwell, 128 GB unified memory)"},
        "software": {"python": platform.python_version(),
                     "main_venv": _pkg_versions(["jmedft", "torch", "transformers", "peft", "accelerate", "datasets",
                                                 "scikit-learn", "numpy", "pandas", "matplotlib", "openai"]),
                     "training_venv": _lf_versions(),
                     "inference_container": conds["vllm_image"]},
        "models": {name: {k: v for k, v in srv.items() if k in ("hf_id", "revision", "served_name")}
                   for name, srv in conds["servers"].items()},
        "openai_models": {c: v["served_model"] for c, v in conds["conditions"].items() if v["provider"] == "openai"},
        "generators": data_cfg["generators"],
        "conditions": conds["conditions"],
        "conditions_executed": sorted(d.name for d in (RESULTS / "predictions").iterdir() if d.is_dir())
        if (RESULTS / "predictions").exists() else [],
        "seeds": {"data": data_cfg["seed"], "training": train_cfg["seed"], "decoding": 0, "bootstrap": 0},
        "training_config": train_cfg,
        "data_config": data_cfg,
        "dataset_versions": {
            "JMED-LLM": {"repo": "sociocom/JMED-LLM", "commit": "3774a34e4a26a7b8a7851f421ecad91a04c2f59b"},
            "JGLUE": {"repo": "yahoojapan/JGLUE", "commit": "6f071c09316baae89c3d083a90985b4b1cb9968c", "version": "v1.3"},
            "Real-MedNLP (MedTxt)": {"download_id": "4864", "license": "CC BY 4.0"},
            "raw_sha256sums": (REPO_ROOT / "data/raw/SHA256SUMS").read_text().count("\n")
            if (REPO_ROOT / "data/raw/SHA256SUMS").exists() else None,
        },
        "processed_data_sha256": {l.split("  ")[1]: l.split("  ")[0]
                                  for l in (processed / "SHA256SUMS").read_text().splitlines()}
        if (processed / "SHA256SUMS").exists() else {},
        "split_stats": json.loads((processed / "split_stats.json").read_text())
        if (processed / "split_stats.json").exists() else {},
    }
    adapter = RESULTS / "adapters/qwen3-8b-jmed-lora"
    if (adapter / "adapter_model.safetensors").exists():
        manifest["adapter"] = {"path": str(adapter.relative_to(REPO_ROOT)),
                               "sha256": {f: sha256_file(adapter / f) for f in ("adapter_config.json",
                                                                                  "adapter_model.safetensors")}}
    return manifest


def generate_report() -> str:
    scores = json.loads((METRICS / "scores.json").read_text())
    comps = json.loads((METRICS / "comparisons.json").read_text())
    eff = json.loads((METRICS / "efficiency.json").read_text())
    figs = make_figures(scores, comps, eff)
    tables = {
        "test_synth": results_table(scores, "test_synth", SYNTH),
        "test_xgen": results_table(scores, "test_xgen", SYNTH),
        "test_ext": results_table(scores, "test_ext", EXT),
        "test_general": results_table(scores, "test_general", GEN),
        "A_secondary_synth": secondary_table(scores, "test_synth", "A_entities",
                                             ["micro_precision", "micro_recall", "micro_f1", "exact_match",
                                              "json_valid_rate", "schema_valid_rate"]),
        "C_secondary_synth": secondary_table(scores, "test_synth", "C_structured",
                                             ["field_accuracy", "exact_match", "list_micro_f1", "missing_fields_total",
                                              "unsupported_rate", "schema_valid_rate"]),
        "B_secondary_synth": secondary_table(scores, "test_synth", "B_doctype", ["accuracy", "macro_f1", "invalid_rate"]),
        "H1": comparison_table(comps, "E3_qwen3_8b_lora__vs__E1_qwen3_8b", ["test_synth", "test_xgen", "test_ext", "test_general"]),
        "H2": comparison_table(comps, "E3_qwen3_8b_lora__vs__E2_swallow_8b", ["test_synth", "test_xgen", "test_ext"]),
        "E4": comparison_table(comps, "E3_qwen3_8b_lora__vs__E4_gpt41", ["test_synth", "test_xgen", "test_ext", "test_general"]),
    }
    (METRICS / "report_tables.md").write_text("\n\n".join(f"### {k}\n\n{v}" for k, v in tables.items()), encoding="utf-8")
    errs = {}
    for cond in ("E1_qwen3_8b", "E2_swallow_8b", "E3_qwen3_8b_lora", "E4_gpt41"):
        for split, task in (("test_synth", "A_entities"), ("test_synth", "C_structured"), ("test_xgen", "B_doctype"),
                            ("test_ext", "ext_mrner_disease"), ("test_ext", "ext_crade")):
            ex = error_examples(cond, split, task)
            if ex:
                errs[f"{cond}/{split}/{task}"] = ex
    (METRICS / "error_examples.json").write_text(json.dumps(errs, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        from jmedft.reporting.costs import cost_table

        cost_table()
    except FileNotFoundError:
        pass
    manifest = build_manifest()
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return json.dumps({"figures": figs, "tables": str(METRICS / "report_tables.md"),
                       "error_examples": str(METRICS / "error_examples.json"),
                       "manifest": str(RESULTS / "manifest.json")}, indent=1)
