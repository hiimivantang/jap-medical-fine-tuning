import random

import pytest

from jmedft.datasets.synth_spec import (
    CATEGORIES,
    NoteSpec,
    gold_from_spec,
    sample_spec,
    split_term_pools,
    validate_note,
)
from jmedft.tasks import C_FIELDS
from jmedft.utils import REPO_ROOT, load_yaml, normalize_text

CFG = load_yaml(REPO_ROOT / "experiments/01_finetuning/configs/data.yaml")
INV = load_yaml(REPO_ROOT / CFG["term_inventory"])
ALL_TERMS = {t for c in CATEGORIES for t in INV[c]}
VCFG = CFG["validation"]


def _spec(**kw) -> NoteSpec:
    base = dict(spec_id="t", doc_type="外来診療録", age=67, sex="男性", chief_complaint=None, symptoms=[],
                negated_symptoms=[], past_history=[], family_history=[], medications=[], allergies=[],
                diagnoses=[], suspected_diagnosis=[])
    base.update(kw)
    return NoteSpec(**base)


PAD = "本日の外来受診時の記録である。全身状態は概ね安定しており、経過を慎重に観察していく方針とした。次回は二週間後に再診予定とする。"


def test_inventory_has_no_substring_collisions():
    terms = [t for c in CATEGORIES for t in INV[c]]
    assert len(terms) == len(set(terms)), "duplicate surface strings across categories"
    norm = [normalize_text(t) for t in terms]
    collisions = [(a, b) for a in norm for b in norm if a != b and a in b]
    assert collisions == []


def test_term_pools_are_disjoint_and_deterministic():
    p1 = split_term_pools(INV, 0.25, 123)
    p2 = split_term_pools(INV, 0.25, 123)
    assert p1 == p2
    for cat in CATEGORIES:
        assert not set(p1[cat]["heldout"]) & set(p1[cat]["train"])
        assert set(p1[cat]["heldout"]) | set(p1[cat]["train"]) == set(INV[cat])


def test_train_specs_never_use_heldout_terms():
    pools = split_term_pools(INV, 0.25, 7)
    heldout = {t for c in CATEGORIES for t in pools[c]["heldout"]}
    rng = random.Random(0)
    for i in range(300):
        s = sample_spec(f"s{i}", rng, pools, CFG["spec_sampling"], "train", 0.5, INV["family_relations"])
        used = set(s.symptoms + s.negated_symptoms + s.past_history + s.diagnoses + s.suspected_diagnosis
                   + s.medications + s.allergies + [x["disease"] for x in s.family_history])
        assert not used & heldout
        assert s.unseen_terms == []


def test_spec_terms_are_unique_within_a_note():
    pools = split_term_pools(INV, 0.25, 7)
    rng = random.Random(1)
    for i in range(300):
        s = sample_spec(f"s{i}", rng, pools, CFG["spec_sampling"], "mixed", 0.5, INV["family_relations"])
        used = (s.symptoms + s.negated_symptoms + s.past_history + s.diagnoses + s.suspected_diagnosis
                + s.medications + s.allergies + [x["disease"] for x in s.family_history])
        assert len(used) == len(set(used))


def test_gold_from_spec_field_order_and_task_a_diseases():
    s = _spec(past_history=["高血圧症"], diagnoses=["肺炎"], suspected_diagnosis=["胃癌"],
              family_history=[{"relation": "父", "disease": "大腸癌"}])
    g = gold_from_spec(s)
    assert list(g["C_structured"]) == C_FIELDS
    assert g["A_entities"]["diseases"] == ["高血圧症", "肺炎"]  # suspected and family excluded
    assert g["C_structured"]["family_history"] == ["大腸癌"]


def test_valid_note_passes():
    s = _spec(symptoms=["頭痛", "発熱"], negated_symptoms=["胸痛"], medications=["アムロジピン"],
              diagnoses=["肺炎"], suspected_diagnosis=["胃癌"])
    text = f"67歳男性。頭痛と発熱を訴える。胸痛は認めない。アムロジピン内服中。肺炎と診断した。胃癌の疑いがある。{PAD}"
    assert validate_note(text, s, ALL_TERMS, VCFG) == []


@pytest.mark.parametrize(
    "text, code",
    [
        (f"67歳男性。発熱を訴える。{PAD}", "missing:頭痛"),
        (f"67歳男性。頭痛と発熱を訴える。胸痛がある。{PAD}", "negation:胸痛"),
        (f"67歳男性。頭痛と発熱を訴える。胸痛は認めない。嘔吐もある。{PAD}", "extra:嘔吐"),
        (f"男性。頭痛と発熱を訴える。胸痛は認めない。{PAD}", "age_missing"),
        (f"67歳男性。現在の診断は不明。頭痛と発熱を訴える。胸痛は認めない。{PAD}", "label_echo"),
        (f"外来診療録\n67歳男性。頭痛と発熱を訴える。胸痛は認めない。{PAD}", "doctype_title"),
    ],
)
def test_violations_detected(text, code):
    s = _spec(symptoms=["頭痛", "発熱"], negated_symptoms=["胸痛"])
    assert code in validate_note(text, s, ALL_TERMS, VCFG)


def test_hedge_on_neighbouring_term_does_not_flag_confirmed_diagnosis():
    s = _spec(symptoms=["頭痛"], diagnoses=["バセドウ病", "大腸癌"], suspected_diagnosis=["緑内障"])
    text = f"67歳男性。頭痛あり。A:バセドウ病、大腸癌。緑内障の可能性。{PAD}"
    assert validate_note(text, s, ALL_TERMS, VCFG) == []


def test_confirmed_diagnosis_written_as_suspected_is_flagged():
    s = _spec(symptoms=["頭痛"], diagnoses=["肺炎"])
    text = f"67歳男性。頭痛あり。肺炎の疑い。{PAD}"
    assert "confirmed_hedged:肺炎" in validate_note(text, s, ALL_TERMS, VCFG)


def test_suspected_diagnosis_must_be_hedged_in_its_own_clause():
    s = _spec(symptoms=["頭痛"], diagnoses=["大腸癌"], suspected_diagnosis=["緑内障"])
    text = f"67歳男性。頭痛あり。緑内障、大腸癌の疑い。{PAD}"
    errs = validate_note(text, s, ALL_TERMS, VCFG)
    assert "suspected_unhedged:緑内障" in errs
    assert "confirmed_hedged:大腸癌" in errs


def test_current_diagnosis_only_in_past_history_is_flagged():
    s = _spec(symptoms=["頭痛"], diagnoses=["肝硬変"])
    text = f"67歳男性。頭痛あり。既往歴として肝硬変がある。{PAD}"
    assert "dx_as_past_history:肝硬変" in validate_note(text, s, ALL_TERMS, VCFG)


def test_current_diagnosis_only_under_past_history_heading_is_flagged():
    s = _spec(symptoms=["頭痛"], diagnoses=["脳出血"])
    text = f"67歳男性。\n主訴\n頭痛\n\n既往歴\n脳出血\n\n経過\n{PAD}"
    assert "dx_as_past_history:脳出血" in validate_note(text, s, ALL_TERMS, VCFG)


def test_current_diagnosis_repeated_in_assessment_section_is_accepted():
    s = _spec(symptoms=["頭痛"], diagnoses=["脳出血"])
    text = f"67歳男性。頭痛あり。\n既往歴\n特記事項なし\n\n評価\n脳出血と診断した。\n{PAD}"
    assert validate_note(text, s, ALL_TERMS, VCFG) == []


def test_past_history_must_be_marked_as_past():
    s = _spec(symptoms=["頭痛"], past_history=["高血圧症"])
    ok = f"67歳男性。頭痛あり。\n既往歴：高血圧症\n{PAD}"
    bad = f"67歳男性。頭痛あり。高血圧症と診断した。\n{PAD}"
    assert validate_note(ok, s, ALL_TERMS, VCFG) == []
    assert "past_history_not_marked:高血圧症" in validate_note(bad, s, ALL_TERMS, VCFG)


@pytest.mark.parametrize("phrase", ["高血圧症の既往あり。", "高血圧症、痛風の既往を有する。", "患者は以前より高血圧症の既往がある。",
                                    "過去の病歴として高血圧症の診断を受けたことがある。"])
def test_post_positioned_past_history_cue_is_accepted(phrase):
    s = _spec(symptoms=["頭痛"], past_history=["高血圧症"])
    text = f"67歳男性。頭痛あり。{phrase}{PAD}"
    errs = validate_note(text, s, ALL_TERMS, VCFG)
    assert not [e for e in errs if e.startswith("past_history_not_marked")]


@pytest.mark.parametrize("assessment", [
    "\n入院時の評価\n本症例では、うつ病と診断した。",
    "\n診察の結果、うつ病と診断された。",
    "\n総合的な評価の結果、うつ病と診断された。",
])
def test_diagnosis_after_past_history_block_is_current(assessment):
    """Regression: a 既往歴 block must not swallow later assessment lines (observed false positives)."""
    s = _spec(symptoms=["頭痛"], past_history=["高血圧症"], diagnoses=["うつ病"])
    text = f"67歳男性。頭痛あり。\n\n既往歴\n高血圧症\n{assessment}\n{PAD}"
    errs = validate_note(text, s, ALL_TERMS, VCFG)
    assert not [e for e in errs if e.startswith(("dx_as_past_history", "past_history_not_marked"))], errs


def test_pii_patterns_flagged():
    s = _spec(symptoms=["頭痛"])
    text = f"67歳男性。2024年3月5日より頭痛あり。連絡先 03-1234-5678。{PAD}"
    assert "pii_pattern" in validate_note(text, s, ALL_TERMS, VCFG)


def test_chief_complaint_stated_but_gold_null_is_flagged():
    s = _spec(symptoms=["頭痛", "発熱"])  # chief_complaint None
    text = f"67歳男性。\n主訴：頭痛\n頭痛と発熱を認める。{PAD}"
    assert "cc_stated_but_unspecified" in validate_note(text, s, ALL_TERMS, VCFG)


def test_chief_complaint_listing_other_symptoms_is_flagged():
    s = _spec(symptoms=["頭痛", "発熱"], chief_complaint="頭痛")
    text = f"67歳男性。\n主訴：頭痛、発熱\n頭痛と発熱を認める。{PAD}"
    assert "cc_multiple_symptoms" in validate_note(text, s, ALL_TERMS, VCFG)


def test_chief_complaint_consistent_is_accepted():
    s = _spec(symptoms=["頭痛", "発熱"], chief_complaint="頭痛")
    for text in (f"67歳男性。\n主訴：頭痛\n頭痛と発熱を認める。{PAD}", f"67歳男性。頭痛を主訴に来院。発熱もある。{PAD}",
                 f"67歳男性。\n【主訴】\n頭痛\n発熱もある。{PAD}"):
        assert validate_note(text, s, ALL_TERMS, VCFG) == [], text
