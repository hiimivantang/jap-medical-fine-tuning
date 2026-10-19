"""Converters for third-party datasets (licences verified 2026-10-19; see docs/DATASETS.md).

- JMED-LLM (sociocom, CC BY 4.0 / CC BY-SA 4.0 subsets) → external test items (never used for training).
- JGLUE v1.3 (CC BY-SA 4.0) → general-Japanese regression test items (never used for training).
- MedTxt-CR-JA / MedTxt-RR-JA (NTCIR-16 Real-MedNLP, CC BY 4.0) → training items in the MRNER format,
  after removing every document that overlaps a JMED-LLM evaluation document.
"""

from __future__ import annotations

import ast
import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from jmedft.tasks import LETTERS, build_messages
from jmedft.utils import char_ngrams, containment

JMED_NER_FILES = {"ext_mrner_disease": "mrner_disease.csv", "ext_mrner_medicine": "mrner_medicine.csv"}
JMED_MC_FILES = {"ext_crade": "crade.csv", "ext_rrtnm": "rrtnm.csv", "gen_jmmlu_med": "jmmlu_med.csv"}
JMED_LICENSE = {"ext_mrner_disease": "CC BY 4.0", "ext_mrner_medicine": "CC BY 4.0", "ext_crade": "CC BY 4.0",
                "ext_rrtnm": "CC BY 4.0", "gen_jmmlu_med": "CC BY-SA 4.0"}

# JMED-LLM NER question stems (verbatim) keyed by tag; reused for MedTxt training conversions so the
# training prompts follow the same instruction wording as the external test.
MRNER_STEMS = {
    ("d", "CR"): "以下の症例報告において、患者に実際に認められた病変や症状の存在を示す異常などの所見を表す表現は？",
    ("d", "RR"): "以下の読影レポートにおいて、患者に実際に認められた病変や症状の存在を示す異常などの所見を表す表現は？",
    ("m-key", "CR"): "以下の症例報告において、薬品名は？",
    ("m-val", "CR"): "以下の症例報告において、処方量など薬品にかかわる値は？",
}


def document_body(question: str) -> str:
    """Text inside the outermost 「…」 of a JMED-LLM question (the source document)."""
    start, end = question.find("「"), question.rfind("」")
    return question[start + 1 : end] if start != -1 and end > start else question


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_jmed_llm(jmed_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    jmed_dir = Path(jmed_dir)
    out: dict[str, list[dict[str, Any]]] = {}
    for task, fname in JMED_NER_FILES.items():
        items = []
        for i, row in enumerate(_read_csv(jmed_dir / fname)):
            gold = [str(x) for x in ast.literal_eval(row["answer"])]
            q = row["question"].strip()
            items.append({
                "id": f"{task}-{i:03d}", "task": task, "split": "test_ext", "source": f"JMED-LLM/{fname}",
                "messages": build_messages(task, question=q), "gold": {"entities": gold},
                "meta": {"source_text": document_body(q), "tag": row["tag"], "license": JMED_LICENSE[task]},
            })
        out[task] = items
    for task, fname in JMED_MC_FILES.items():
        items = []
        for i, row in enumerate(_read_csv(jmed_dir / fname)):
            opt_keys = [k for k in row if k.startswith("option") and (row[k] or "").strip()]
            options = [row[k].strip() for k in opt_keys]
            labels = list(LETTERS[: len(options)])
            q = row["question"].strip()
            split = "test_general" if task.startswith("gen_") else "test_ext"
            items.append({
                "id": f"{task}-{i:03d}", "task": task, "split": split, "source": f"JMED-LLM/{fname}",
                "messages": build_messages(task, question=q, options=options),
                "gold": {"answer": row["answer"].strip(), "labels": labels},
                "meta": {"source_text": document_body(q), "tag": row["tag"], "license": JMED_LICENSE[task]},
            })
        out[task] = items
    return out


JNLI_LABELS = {"entailment": "含意", "contradiction": "矛盾", "neutral": "中立"}


def load_jglue(jglue_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    jglue_dir = Path(jglue_dir)
    jcqa = []
    for line in (jglue_dir / "jcommonsenseqa-test-v1.3.json").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        options = [r[f"choice{k}"] for k in range(5)]
        jcqa.append({
            "id": f"gen_jcqa-{r['q_id']}", "task": "gen_jcqa", "split": "test_general",
            "source": "JGLUE-v1.3/jcommonsenseqa/test",
            "messages": build_messages("gen_jcqa", question=f"次の質問に答えてください。\n質問: {r['question']}",
                                       options=options),
            "gold": {"answer": LETTERS[int(r["label"])], "labels": list(LETTERS[:5])},
            "meta": {"source_text": r["question"], "license": "CC BY-SA 4.0"},
        })
    jnli = []
    order = ["entailment", "contradiction", "neutral"]
    for line in (jglue_dir / "jnli-test-v1.3.json").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        q = ("前提文と仮説文の関係を答えてください。\n"
             f"前提: {r['sentence1']}\n仮説: {r['sentence2']}")
        options = [f"{JNLI_LABELS[k]}（{k}）" for k in order]
        jnli.append({
            "id": f"gen_jnli-{r['sentence_pair_id']}", "task": "gen_jnli", "split": "test_general",
            "source": "JGLUE-v1.3/jnli/test", "messages": build_messages("gen_jnli", question=q, options=options),
            "gold": {"answer": LETTERS[order.index(r["label"])], "labels": list(LETTERS[:3])},
            "meta": {"source_text": r["sentence1"] + r["sentence2"], "license": "CC BY-SA 4.0"},
        })
    return {"gen_jcqa": jcqa, "gen_jnli": jnli}


def _article_text_and_entities(article: ET.Element) -> tuple[str, dict[str, list[str]]]:
    """Plain text of a MedTxt article plus the JMED-LLM-style gold lists derived from its inline tags."""
    text = "".join(article.itertext())
    ents: dict[str, list[str]] = {"d": [], "m-key": [], "m-val": []}
    for el in article.iter():
        tag = el.tag
        surface = "".join(el.itertext())
        if tag == "d" and el.get("certainty") == "positive":
            ents["d"].append(surface)
        elif tag in ("m-key", "m-val"):
            ents[tag].append(surface)
    # Preserve first-occurrence order, drop duplicates and empty strings.
    for k, v in ents.items():
        seen: list[str] = []
        for s in v:
            s = s.strip()
            if s and s not in seen:
                seen.append(s)
        ents[k] = seen
    return text, ents


def _clean_lines(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def medtxt_training_items(xml_path: str | Path, kind: str, eval_texts: list[str], threshold: float,
                          split_name: str = "train") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert MedTxt articles to MRNER-format items; returns (items, removed_docs_log).

    kind: "CR" (case reports) or "RR" (radiology reports). Any article whose character 5-gram containment
    with any JMED-LLM eval document reaches `threshold` is removed and logged.
    """
    eval_grams = [char_ngrams(t) for t in eval_texts]
    root = ET.parse(xml_path).getroot()
    items, removed = [], []
    for art in root.iter("article"):
        text, ents = _article_text_and_entities(art)
        text = _clean_lines(text)
        g = char_ngrams(text)
        overlap = max((containment(g, e) for e in eval_grams), default=0.0)
        doc_id = f"MedTxt-{kind}-{art.get('id')}"
        if overlap >= threshold:
            removed.append({"doc_id": doc_id, "max_containment": round(overlap, 3)})
            continue
        tasks = [("ext_mrner_disease", "d")] + ([("ext_mrner_medicine", "m-key"), ("ext_mrner_medicine", "m-val")]
                                                if kind == "CR" else [])
        for task, tag in tasks:
            gold = ents[tag]
            if not gold:
                continue
            question = f"{MRNER_STEMS[(tag, kind)]}\n「{text}」"
            items.append({
                "id": f"{doc_id}-{tag}", "task": task, "split": split_name, "source": f"MedTxt-{kind}-JA",
                "messages": build_messages(task, question=question), "gold": {"entities": gold},
                "meta": {"source_text": text, "tag": tag, "doc_id": doc_id, "license": "CC BY 4.0",
                         "max_containment_vs_jmed_eval": round(overlap, 3)},
            })
    return items, removed


def strip_question_prefix(question: str) -> str:
    return re.sub(r"^以下の.*?[？?]\s*", "", question, count=1, flags=re.DOTALL)
