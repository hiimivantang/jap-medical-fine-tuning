"""Scoring tests with hand-computed expectations (never derived by calling the code under test)."""

import json

import numpy as np
import pytest
from sklearn.metrics import f1_score

from jmedft.evaluation import metrics, stats
from jmedft.tasks import DOC_TYPES


def _pred(item_id: str, obj: dict | None = None, text: str | None = None, error: str | None = None) -> dict:
    output = json.dumps(obj, ensure_ascii=False) if obj is not None else text
    return {"id": item_id, "output_text": output, "error": error}


def _a_case() -> tuple[str, list[dict], list[dict]]:
    items = [
        {"id": "a1", "gold": {"symptoms": ["発熱", "咳嗽"], "diseases": ["高血圧"],
                              "medications": ["２０㎎"]}},
        {"id": "a2", "gold": {"symptoms": ["腹痛"], "diseases": [], "medications": []}},
        {"id": "a3", "gold": {"symptoms": ["嘔気"], "diseases": ["糖尿病"], "medications": []}},
    ]
    preds = [  # a1: 発熱 three times (one with a space) counts once; "20mg" == NFKC("２０㎎")
        _pred("a1", {"symptoms": ["発熱", "発熱", "発 熱", "頭痛"], "diseases": ["高血圧", "胃炎"],
                     "medications": ["20mg"]}),
        _pred("a2", {"symptoms": ["腹痛"], "diseases": [], "medications": []}),
    ]  # a3 has no prediction record at all -> all FN
    return "A_entities", items, preds


B_GOLD = ["外来診療録", "外来診療録", "入院時記録", "紹介状", "看護記録", "退院時サマリー"]
B_PRED = ["外来診療録", "紹介状", "入院時記録", "紹介状", None, "外来診療録"]  # None -> unparseable output


def _b_case() -> tuple[str, list[dict], list[dict]]:
    items = [{"id": f"b{i}", "gold": {"label": g}} for i, g in enumerate(B_GOLD)]
    preds = [_pred(f"b{i}", {"label": p}) if p else _pred(f"b{i}", text="判定できません")
             for i, p in enumerate(B_PRED)]
    return "B_doctype", items, preds


def _mc_case() -> tuple[str, list[dict], list[dict]]:
    items = [{"id": f"m{i}", "gold": {"answer": a, "labels": ["A", "B", "C", "D"]}}
             for i, a in enumerate("ABCD")]
    preds = [
        _pred("m0", {"answer": "A"}),
        _pred("m1", text="答えはB"),  # no JSON: strict None, lenient B
        _pred("m2", {"answer": "D"}),
        _pred("m3", {"answer": "D"}, error="timeout"),  # errored request is scored as "" despite the text
    ]
    return "ext_crade", items, preds


C_SRC_1 = "67歳男性。主訴は胸痛。高血圧の既往あり。アムロジピン内服中。発熱なし。ペニシリンアレルギーあり。"
C_SRC_2 = "女性。主訴は頭痛。頭痛と嘔気を認め、片頭痛と診断した。"
C_GOLD_1 = {"age": 67, "sex": "男性", "chief_complaint": "胸痛", "symptoms": ["胸痛"],
            "negated_symptoms": ["発熱"], "past_history": ["高血圧"], "family_history": [],
            "medications": ["アムロジピン"], "allergies": ["ペニシリン"], "diagnoses": [],
            "suspected_diagnosis": []}
C_GOLD_2 = {"age": None, "sex": "女性", "chief_complaint": "頭痛", "symptoms": ["頭痛", "嘔気"],
            "negated_symptoms": [], "past_history": [], "family_history": [], "medications": [],
            "allergies": [], "diagnoses": ["片頭痛"], "suspected_diagnosis": []}


def _c_case() -> tuple[str, list[dict], list[dict]]:
    items = [{"id": "c1", "gold": C_GOLD_1, "meta": {"source_text": C_SRC_1}},
             {"id": "c2", "gold": C_GOLD_2, "meta": {"source_text": C_SRC_2}}]
    # c1: age "67歳", "allergies" key missing, medication not in the source -> wrong: medications, allergies
    pred1 = {k: v for k, v in C_GOLD_1.items() if k != "allergies"}
    pred1.update(age="67歳", medications=["アムロジピン", "ロキソプロフェン"])
    # c2: schema-valid; misses 嘔気 and files 片頭痛 as suspected -> wrong: symptoms, diagnoses, suspected
    pred2 = {**C_GOLD_2, "symptoms": ["頭痛"], "diagnoses": [], "suspected_diagnosis": ["片頭痛"]}
    return "C_structured", items, [_pred("c1", pred1), _pred("c2", pred2)]


def test_entities_micro_counts_dedup_nfkc_and_missing_prediction():
    task, items, preds = _a_case()
    res = metrics.score_task(task, items, preds)
    assert (res["task"], res["family"], res["n"], res["n_missing_predictions"]) == (task, "entities", 3, 1)
    assert res["micro_precision"] == pytest.approx(4 / 6)
    assert res["micro_recall"] == pytest.approx(4 / 7)
    assert res["micro_f1"] == pytest.approx(8 / 13)
    assert res["exact_match"] == pytest.approx(1 / 3)
    assert (res["json_valid_rate"], res["schema_valid_rate"]) == pytest.approx((2 / 3, 2 / 3))
    per_key = {
        "symptoms": {"precision": 2 / 3, "recall": 1 / 2, "f1": 4 / 7, "support": 4},
        "diseases": {"precision": 1 / 2, "recall": 1 / 2, "f1": 1 / 2, "support": 2},
        "medications": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1},
    }
    assert list(res["per_key"]) == list(per_key)
    for key, expected in per_key.items():
        assert res["per_key"][key] == pytest.approx(expected)
    assert [r["id"] for r in res["items"]] == ["a1", "a2", "a3"]
    a1, a2, a3 = res["items"]
    assert (a1["tp"], a1["fp"], a1["fn"], a1["exact"]) == (3, 2, 1, False)
    assert a1["per_key"]["symptoms"] == {"tp": 1, "fp": 1, "fn": 1, "fp_values": ["頭痛"],
                                         "fn_values": ["咳嗽"]}
    assert a1["per_key"]["medications"] == {"tp": 1, "fp": 0, "fn": 0, "fp_values": [], "fn_values": []}
    assert (a2["tp"], a2["fp"], a2["fn"], a2["exact"]) == (1, 0, 0, True)
    assert (a3["tp"], a3["fp"], a3["fn"], a3["exact"]) == (0, 0, 2, False)
    assert a3["per_key"]["diseases"]["fn_values"] == ["糖尿病"]


def test_entities_bootstrap_data_and_resampled_statistic():
    task, items, preds = _a_case()
    data = metrics.bootstrap_data(task, metrics.score_task(task, items, preds))
    assert data["ids"] == ["a1", "a2", "a3"]
    assert data["tp"].tolist() == [3, 1, 0]
    assert data["fp"].tolist() == [2, 0, 0]
    assert data["fn"].tolist() == [1, 0, 2]
    fn = metrics.statistic_fn(task)
    assert fn(data, np.array([0, 0])) == pytest.approx(12 / 18)  # tp=6, fp=4, fn=2
    assert fn(data, np.array([2])) == 0.0


def test_ext_ner_uses_entities_key():
    items = [{"id": "e1", "gold": {"entities": ["肺炎", "COPD"]}}]
    res = metrics.score_task("ext_mrner_disease", items, [_pred("e1", {"entities": ["ＣＯＰＤ"]})])
    assert list(res["per_key"]) == ["entities"]
    assert (res["items"][0]["tp"], res["items"][0]["fp"], res["items"][0]["fn"]) == (1, 0, 1)
    assert res["micro_precision"] == pytest.approx(1.0)
    assert res["micro_recall"] == pytest.approx(1 / 2)
    assert res["micro_f1"] == pytest.approx(2 / 3)


def test_classification_invalid_output_lands_in_invalid_column():
    task, items, preds = _b_case()
    res = metrics.score_task(task, items, preds)
    assert res["labels"] == DOC_TYPES
    assert res["accuracy"] == pytest.approx(3 / 6)
    assert res["invalid_rate"] == pytest.approx(1 / 6)
    assert res["json_valid_rate"] == pytest.approx(5 / 6)
    assert res["macro_f1"] == pytest.approx(13 / 36)  # (1/2 + 1 + 0 + 2/3 + 0 + 0) / 6
    assert res["confusion_matrix"] == [
        [1, 0, 0, 1, 0, 0, 0],  # 外来診療録: one right, one predicted 紹介状
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],  # 退院時サマリー predicted as 外来診療録
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],  # 看護記録: unparseable output -> INVALID column
        [0, 0, 0, 0, 0, 0, 0],
    ]
    per_class = res["per_class"]
    assert per_class["外来診療録"] == pytest.approx({"precision": 0.5, "recall": 0.5, "f1": 0.5,
                                                     "support": 2})
    assert per_class["紹介状"] == pytest.approx({"precision": 0.5, "recall": 1.0, "f1": 2 / 3, "support": 1})
    assert per_class["看護記録"] == pytest.approx({"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1})
    assert res["items"][4] == {"id": "b4", "gold": "看護記録", "pred": None, "correct": False}
    assert metrics.correctness(task, res) == [True, False, True, True, False, False]
    fixed = [*preds[:4], _pred("b4", {"label": "看護記録"}), preds[5]]
    assert metrics.score_task(task, items, fixed)["accuracy"] == pytest.approx(4 / 6)


def test_classification_numpy_macro_f1_equals_sklearn():
    task, items, preds = _b_case()
    res = metrics.score_task(task, items, preds)
    data = metrics.bootstrap_data(task, res)
    assert (data["gold"].tolist(), data["pred"].tolist()) == ([0, 0, 1, 3, 4, 2], [0, 3, 1, 3, -1, 0])
    assert data["n_labels"] == 6
    fn = metrics.statistic_fn(task)
    pred = [p or "INVALID" for p in B_PRED]
    ref = f1_score(B_GOLD, pred, labels=DOC_TYPES, average="macro", zero_division=0)
    assert fn(data, None) == pytest.approx(ref)
    assert fn(data, None) == pytest.approx(res["macro_f1"])
    assert fn(data, np.array([0, 2])) == pytest.approx(2 / 6)  # two perfect classes out of six
    rng = np.random.default_rng(0)
    gold, pred = rng.integers(0, 6, 300), rng.integers(-1, 6, 300)
    rand = {"ids": [str(i) for i in range(300)], "gold": gold, "pred": pred, "n_labels": 6}
    ref = f1_score(gold, pred, labels=list(range(6)), average="macro", zero_division=0)
    assert fn(rand, None) == pytest.approx(ref)


def test_mc_lenient_vs_strict():
    task, items, preds = _mc_case()
    res = metrics.score_task(task, items, preds)
    assert (res["family"], res["n_missing_predictions"]) == ("mc", 1)
    assert res["accuracy"] == pytest.approx(2 / 4)
    assert res["strict_accuracy"] == pytest.approx(1 / 4)
    assert res["invalid_rate"] == pytest.approx(1 / 4)
    assert res["json_valid_rate"] == pytest.approx(2 / 4)
    assert res["items"][1] == {"id": "m1", "gold": "B", "pred": "B", "strict_pred": None, "correct": True,
                               "strict_correct": False}
    assert res["items"][3]["pred"] is None
    assert res["labels"] == ["A", "B", "C", "D"]
    assert res["confusion_matrix"] == [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
    assert res["macro_f1"] == pytest.approx(2 / 4)  # A and B perfect; C and D have F1 0
    assert metrics.correctness(task, res) == [True, True, False, False]
    assert metrics.bootstrap_data(task, res)["correct"].tolist() == [True, True, False, False]


def test_structured_fields_missing_and_unsupported():
    task, items, preds = _c_case()
    res = metrics.score_task(task, items, preds)
    c1, c2 = res["items"]
    assert c1["field_correct"]["age"] is True  # "67歳" -> 67
    wrong1 = [f for f, ok in c1["field_correct"].items() if not ok]
    wrong2 = [f for f, ok in c2["field_correct"].items() if not ok]
    assert wrong1 == ["medications", "allergies"]
    assert wrong2 == ["symptoms", "diagnoses", "suspected_diagnosis"]
    assert (c1["fields_correct"], c2["fields_correct"], c1["exact"]) == (9, 8, False)
    assert (c1["missing_fields"], c2["missing_fields"]) == (["allergies"], [])
    assert (c1["unsupported_values"], c2["unsupported_values"]) == (["ロキソプロフェン"], [])
    assert (c1["n_predicted_values"], c2["n_predicted_values"]) == (6, 3)
    assert (c1["list_tp"], c1["list_fp"], c1["list_fn"]) == (4, 1, 1)
    assert (c2["list_tp"], c2["list_fp"], c2["list_fn"]) == (1, 1, 2)
    assert res["field_accuracy"] == pytest.approx(17 / 22)
    assert res["exact_match"] == 0.0
    assert res["per_field_accuracy"] == pytest.approx({
        "age": 1.0, "sex": 1.0, "chief_complaint": 1.0, "symptoms": 0.5, "negated_symptoms": 1.0,
        "past_history": 1.0, "family_history": 1.0, "medications": 0.5, "allergies": 0.5, "diagnoses": 0.5,
        "suspected_diagnosis": 0.5,
    })
    assert (res["list_micro_precision"], res["list_micro_recall"]) == pytest.approx((5 / 7, 5 / 8))
    assert res["list_micro_f1"] == pytest.approx(2 / 3)
    assert res["per_list_field_f1"] == pytest.approx({
        "symptoms": 4 / 5, "negated_symptoms": 1.0, "past_history": 1.0, "family_history": 0.0,
        "medications": 2 / 3, "allergies": 0.0, "diagnoses": 0.0, "suspected_diagnosis": 0.0,
    })
    assert (res["missing_fields_total"], res["unsupported_total"]) == (1, 1)
    assert res["missing_fields_rate"] == pytest.approx(1 / 22)
    assert res["unsupported_rate"] == pytest.approx(1 / 9)
    assert (res["json_valid_rate"], res["schema_valid_rate"]) == (1.0, 0.5)
    data = metrics.bootstrap_data(task, res)
    assert (data["ids"], data["fields_correct"].tolist(), data["n_fields"]) == (["c1", "c2"], [9, 8], 11)


@pytest.mark.parametrize(("case", "name", "expected"), [
    (_a_case, "micro_f1", 8 / 13),
    (_b_case, "macro_f1", 13 / 36),
    (_mc_case, "accuracy", 2 / 4),
    (_c_case, "field_accuracy", 17 / 22),
])
def test_statistic_on_all_rows_equals_reported_primary_metric(case, name, expected):
    task, items, preds = case()
    res = metrics.score_task(task, items, preds)
    assert metrics.primary_metric(task) == name
    assert res[name] == pytest.approx(expected)
    assert metrics.statistic_fn(task)(metrics.bootstrap_data(task, res), None) == pytest.approx(expected)


@pytest.mark.parametrize("task", ["A_entities", "ext_mrner_medicine", "C_structured"])
def test_correctness_undefined_for_set_and_structured_tasks(task):
    with pytest.raises(ValueError):
        metrics.correctness(task, {"items": []})


def test_bootstrap_data_survives_stats_alignment_and_pairing():
    task, items, preds = _b_case()
    da = metrics.bootstrap_data(task, metrics.score_task(task, items, preds))
    db = metrics.bootstrap_data(task, metrics.score_task(task, items[::-1], preds))
    da, db = stats.align_by_id(da, db)
    assert da["ids"] == db["ids"] == [f"b{i}" for i in range(6)]
    assert (da["n_labels"], db["pred"].tolist()) == (6, [0, 3, 1, 3, -1, 0])
    diff, lo, hi, _ = stats.paired_bootstrap_diff(da, db, metrics.statistic_fn(task), n_resamples=50, seed=0)
    assert (diff, lo, hi) == (0.0, 0.0, 0.0)


def test_lenient_match_count_one_to_one_containment():
    from jmedft.evaluation.metrics import lenient_match_count

    gold = {"胃癌", "走行異常", "発熱"}
    pred = {"胃体上部小彎の胃癌", "胃癌の再発", "走行異常", "咳嗽"}
    # exact: 走行異常 (1); containment: 胃癌 matches ONE of the two longer spans (one-to-one) → 2 total.
    assert lenient_match_count(gold, pred) == 2
    assert lenient_match_count(set(), {"a"}) == 0
    assert lenient_match_count({"頭痛"}, {"頭痛"}) == 1


def test_lenient_match_is_maximal_and_ignores_single_chars():
    from jmedft.evaluation.metrics import lenient_match_count

    # Greedy would pair AB↔ABC first and leave ABCDE unmatched; maximal matching finds 2 pairs.
    assert lenient_match_count({"AB", "ABCDE"}, {"ABC", "ABCDEFG"}) == 2
    assert lenient_match_count({"胃癌"}, {"癌"}) == 0  # single character never matches by containment
