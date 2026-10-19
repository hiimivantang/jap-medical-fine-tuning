"""Leakage and integrity invariants of the dataset builder (run on small fixtures, no GPU)."""

import json

import pytest

from jmedft.datasets import build as build_mod
from jmedft.datasets.build import near_dup_filter, synthetic_items, to_sft
from jmedft.datasets.external import document_body, load_jglue, load_jmed_llm, medtxt_training_items
from jmedft.utils import REPO_ROOT, normalize_text

RAW = REPO_ROOT / "data/raw"
needs_raw = pytest.mark.skipif(not (RAW / "jmed_llm/mrner_disease.csv").exists(),
                               reason="external data not fetched (run scripts/fetch_external_data.sh)")


def _raw_record(spec_id: str, text: str, accepted: bool = True) -> dict:
    spec = {"spec_id": spec_id, "doc_type": "紹介状", "age": 70, "sex": "女性", "chief_complaint": "頭痛", "symptoms": ["頭痛"],
                "negated_symptoms": [], "past_history": ["高血圧症"], "family_history": [], "medications": ["アムロジピン"],
                "allergies": [], "diagnoses": [], "suspected_diagnosis": [], "unseen_terms": []}
    return {"spec": spec, "text": text if accepted else None, "accepted": accepted, "generator": "g",
            "generator_revision": "r", "attempts": []}


def test_synthetic_items_one_per_task_and_rejected_dropped(tmp_path):
    p = tmp_path / "x.raw.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [
        _raw_record("s-1", "70歳女性。頭痛を主訴に受診。高血圧症でアムロジピン内服中。"),
        _raw_record("s-2", "", accepted=False)]))
    items = synthetic_items(p, "train")
    assert [i["task"] for i in items] == ["A_entities", "B_doctype", "C_structured"]
    assert all(i["meta"]["spec_id"] == "s-1" for i in items)
    sft = to_sft(items[0])
    assert [m["role"] for m in sft["messages"]] == ["system", "user", "assistant"]
    assert json.loads(sft["messages"][-1]["content"]) == {"symptoms": ["頭痛"], "diseases": ["高血圧症"],
                                                          "medications": ["アムロジピン"]}


def test_near_dup_filter_drops_all_tasks_of_duplicate_note():
    text = "70歳女性。頭痛を主訴に受診。高血圧症でアムロジピン内服中。経過観察とした。"
    items = [{"meta": {"spec_id": "t-1", "source_text": text}, "task": t} for t in ("A", "B", "C")]
    items += [{"meta": {"spec_id": "t-2", "source_text": "全く異なる内容の文書である。腹痛と下痢が続いている。"}, "task": "A"}]
    kept, dropped = near_dup_filter(items, [text], 0.6)
    assert {i["meta"]["spec_id"] for i in kept} == {"t-2"}
    assert [d["spec_id"] for d in dropped] == ["t-1"]


@needs_raw
def test_jmed_llm_loader_counts_and_gold_in_text():
    jm = load_jmed_llm(RAW / "jmed_llm")
    assert {k: len(v) for k, v in jm.items()} == {"ext_mrner_disease": 100, "ext_mrner_medicine": 100,
                                                  "ext_crade": 100, "ext_rrtnm": 100, "gen_jmmlu_med": 100}
    for it in jm["ext_mrner_disease"] + jm["ext_mrner_medicine"]:
        src = normalize_text(it["meta"]["source_text"])
        assert all(normalize_text(e) in src for e in it["gold"]["entities"])
    for it in jm["ext_crade"]:
        assert it["gold"]["labels"] == ["A", "B", "C", "D"]


@needs_raw
def test_jglue_loader_counts():
    jg = load_jglue(RAW / "jglue")
    assert len(jg["gen_jcqa"]) == 1118 and len(jg["gen_jnli"]) == 2508
    assert {it["gold"]["answer"] for it in jg["gen_jnli"]} == {"A", "B", "C"}


@needs_raw
def test_medtxt_training_docs_do_not_overlap_jmed_eval():
    jm = load_jmed_llm(RAW / "jmed_llm")
    eval_texts = [it["meta"]["source_text"] for t in ("ext_mrner_disease", "ext_mrner_medicine", "ext_crade",
                                                      "ext_rrtnm") for it in jm[t]]
    items, removed = medtxt_training_items(RAW / "real_mednlp/extracted/MedTxt-CR-JA-training-pub.xml", "CR",
                                           eval_texts, 0.2)
    assert len(removed) >= 100  # the 107 JMED-LLM source articles must be removed
    assert all(it["meta"]["max_containment_vs_jmed_eval"] < 0.2 for it in items)
    eval_bodies = {normalize_text(t) for t in eval_texts}
    assert not any(normalize_text(it["meta"]["source_text"]) in eval_bodies for it in items)


def test_builder_never_reads_test_split_for_training():
    """Training SFT is built only from synthetic train + MedTxt; the source must not feed test splits into it."""
    src = (REPO_ROOT / "src/jmedft/datasets/build.py").read_text()
    sft_line = next(line for line in src.splitlines() if line.strip().startswith("sft_train ="))
    assert "test" not in sft_line and "jmed" not in sft_line and "jglue" not in sft_line


def test_document_body_extracts_outer_quotes():
    assert document_body("以下の症例報告において、薬品名は？\n「本文「入れ子」終わり」") == "本文「入れ子」終わり"


def test_build_module_exposes_build():
    assert callable(build_mod.build)
