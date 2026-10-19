import json

from jmedft.evaluation.errors import classify_item


def _item(gold, text, c_gold=None):
    return {"id": "x", "task": "A_entities", "gold": gold,
            "meta": {"source_text": text, "c_gold": c_gold or {}}}


def test_taxonomy_categories_hand_labelled():
    text = "頭痛と発熱を認める。胸痛は認めない。胸部X線で斑状陰影を認める。胃癌の疑い。アムロジピン内服中。"
    gold = {"symptoms": ["頭痛", "発熱"], "diseases": [], "medications": ["アムロジピン"]}
    c_gold = {"negated_symptoms": ["胸痛"], "suspected_diagnosis": ["胃癌"]}
    out = json.dumps({"symptoms": ["頭痛", "胸痛", "斑状陰影", "咳嗽"], "diseases": ["胃癌"],
                      "medications": ["アムロジピン内服"]}, ensure_ascii=False)
    c = classify_item("A_entities", _item(gold, text, c_gold), out)
    assert c["negation_error"] == 1          # 胸痛 is negated in the text
    assert c["unscripted_finding"] == 1      # 斑状陰影 is in the text but in no gold field
    assert c["unsupported"] == 1             # 咳嗽 is not in the text
    assert c["wrong_field"] == 1             # 胃癌 is a suspected diagnosis, predicted as disease
    assert c["span_boundary"] == 1           # アムロジピン内服 vs アムロジピン
    assert c["missed"] == 1                  # 発熱 not predicted
    assert "format_invalid_json" not in c


def test_invalid_json_counts_misses_as_format():
    c = classify_item("A_entities", _item({"symptoms": ["頭痛"], "diseases": [], "medications": []}, "頭痛あり"),
                      "not json")
    assert c["format_invalid_json"] == 1 and c["missed_due_to_format"] == 1
