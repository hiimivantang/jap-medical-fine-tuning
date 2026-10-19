"""Tests for jmedft.evaluation.parsing. Every expected value is written out by hand."""

import pytest

from jmedft.evaluation.parsing import ParsedPrediction, extract_json_object, parse_prediction, strip_reasoning
from jmedft.utils import normalize_text

EMPTY_A = {"symptoms": [], "diseases": [], "medications": []}
EMPTY_C = {
    "age": None, "sex": None, "chief_complaint": None, "symptoms": [], "negated_symptoms": [],
    "past_history": [], "family_history": [], "medications": [], "allergies": [], "diagnoses": [],
    "suspected_diagnosis": [],
}
ALL_C_FIELDS = ["age", "sex", "chief_complaint", "symptoms", "negated_symptoms", "past_history",
                "family_history", "medications", "allergies", "diagnoses", "suspected_diagnosis"]
ABCDE = ["A", "B", "C", "D", "E"]


# --- reasoning blocks --------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<think>R</think>A", ("A", "R", False)),
        ('<think>\n検討中\n</think>\n\n{"answer": "B"}\n', ('{"answer": "B"}', "検討中", False)),
        ("<think>途中まで考え", ("", "途中まで考え", True)),
        ("推論のみ</think>答え", ("答え", "推論のみ", False)),
        ("a</think>b</think> c ", ("c", "a</think>b", False)),
        ("前置き<think>R</think>A", ("A", "前置きR", False)),
        ("  そのまま  ", ("そのまま", None, False)),
    ],
)
def test_strip_reasoning(raw, expected):
    assert strip_reasoning(raw) == expected


def test_strip_reasoning_empty_think_block():
    answer, reasoning, unterminated = strip_reasoning("<think></think>A")
    assert answer == "A"
    assert reasoning in ("", None)
    assert unterminated is False


# --- JSON extraction ---------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('```json\n{"label": "紹介状"}\n```', {"label": "紹介状"}),
        ('```\n{"answer": "B"}\n```', {"answer": "B"}),
        ('{"a": "x}y", "b": 1}', {"a": "x}y", "b": 1}),
        ('{"a": "{[(", "b": "}"}', {"a": "{[(", "b": "}"}),
        ('{"s": "q\\"}"}', {"s": 'q"}'}),
        ('回答は以下の通りです。\n{"answer": "C"}\n以上。', {"answer": "C"}),
        ('{"a": {"b": [1, 2]}} {"c": 3}', {"a": {"b": [1, 2]}}),
        ('{これはJSONではない} {"k": [1]}', {"k": [1]}),
        ('{ {"a": 1}', {"a": 1}),
    ],
)
def test_extract_json_object(text, expected):
    assert extract_json_object(text) == expected


@pytest.mark.parametrize("text", ["", "JSONなし", "[1, 2]", '{"a": 1', "{'a': 1}"])
def test_extract_json_object_none(text):
    assert extract_json_object(text) is None


def test_normalize_text_nfkc_equivalence():
    assert normalize_text("２０㎎") == "20mg"
    assert normalize_text(" 看護　記録 ") == "看護記録"


# --- entity sets -------------------------------------------------------------------------------------------

def test_entities_valid():
    text = '{"symptoms": ["発熱", "咳嗽"], "diseases": [], "medications": ["アセトアミノフェン"]}'
    expected = {"symptoms": ["発熱", "咳嗽"], "diseases": [], "medications": ["アセトアミノフェン"]}
    assert parse_prediction("A_entities", text) == ParsedPrediction(True, True, expected, whole_json=True)


@pytest.mark.parametrize("text", ['{"symptoms": ["発熱"]', "症状: 発熱、咳嗽", ""])
def test_entities_invalid_json_is_empty_prediction(text):
    p = parse_prediction("A_entities", text)
    assert (p.json_valid, p.schema_valid) == (False, False)
    assert p.value == EMPTY_A


def test_entities_single_string_and_item_coercion():
    p = parse_prediction("ext_mrner_disease", '{"entities": "肺炎"}')
    assert (p.json_valid, p.schema_valid, p.value) == (True, False, {"entities": ["肺炎"]})
    p = parse_prediction("ext_mrner_medicine", '{"entities": [" ロキソニン ", "", 3, null, "アムロジピン"]}')
    assert (p.json_valid, p.schema_valid, p.value) == (True, False, {"entities": ["ロキソニン", "アムロジピン"]})


def test_entities_missing_key_and_wrong_type():
    p = parse_prediction("A_entities", '{"symptoms": ["頭痛"], "diseases": {"x": 1}}')
    assert (p.json_valid, p.schema_valid) == (True, False)
    assert p.value == {"symptoms": ["頭痛"], "diseases": [], "medications": []}


# --- classification (B_doctype) ----------------------------------------------------------------------------

def test_doctype_from_json_label():
    p = parse_prediction("B_doctype", '{"label": "紹介状"}')
    assert (p.json_valid, p.schema_valid, p.value) == (True, True, "紹介状")


def test_doctype_json_label_matched_after_normalisation():
    # half-width katakana middle dot (U+FF65) and an ideographic space normalise to the canonical name
    p = parse_prediction("B_doctype", '{"label": "検査･画像所見　報告書"}')
    assert (p.json_valid, p.schema_valid, p.value) == (True, False, "検査・画像所見報告書")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("この文書は看護記録です。", "看護記録"),
        ("紹介状ではなく看護記録と判断します", "紹介状"),  # earliest occurrence wins, not the longer name
        ("種類: 外来 診療録", "外来診療録"),  # whitespace is removed before matching
        ('{"label": "診療情報提供書（紹介状）"}', "紹介状"),  # invalid JSON label -> text search
    ],
)
def test_doctype_text_fallback(text, expected):
    assert parse_prediction("B_doctype", text).value == expected


@pytest.mark.parametrize("text", ["判断できません", '{"label": "診断書"}', ""])
def test_doctype_none_when_no_doc_type(text):
    assert parse_prediction("B_doctype", text).value is None


# --- structured (C_structured) -----------------------------------------------------------------------------

def test_structured_full_valid():
    text = ('{"age": 67, "sex": "男性", "chief_complaint": "胸痛", "symptoms": ["胸痛", "冷汗"], '
            '"negated_symptoms": ["発熱"], "past_history": ["高血圧症"], "family_history": [], '
            '"medications": ["アムロジピン"], "allergies": [], "diagnoses": ["急性心筋梗塞"], '
            '"suspected_diagnosis": []}')
    p = parse_prediction("C_structured", text)
    assert (p.json_valid, p.schema_valid, p.missing_fields) == (True, True, [])
    assert p.value == {
        "age": 67, "sex": "男性", "chief_complaint": "胸痛", "symptoms": ["胸痛", "冷汗"],
        "negated_symptoms": ["発熱"], "past_history": ["高血圧症"], "family_history": [],
        "medications": ["アムロジピン"], "allergies": [], "diagnoses": ["急性心筋梗塞"],
        "suspected_diagnosis": [],
    }


def test_structured_partial_coercion_and_missing_fields():
    text = '{"age": "67歳", "sex": "男", "chief_complaint": "  頭痛 ", "symptoms": "嘔気"}'
    p = parse_prediction("C_structured", text)
    assert (p.json_valid, p.schema_valid) == (True, False)
    assert p.missing_fields == ["negated_symptoms", "past_history", "family_history", "medications",
                                "allergies", "diagnoses", "suspected_diagnosis"]
    assert p.value == {**EMPTY_C, "age": 67, "chief_complaint": "頭痛", "symptoms": ["嘔気"]}


def test_structured_no_json_all_fields_missing():
    p = parse_prediction("C_structured", "抽出できませんでした")
    assert (p.json_valid, p.schema_valid) == (False, False)
    assert p.missing_fields == ALL_C_FIELDS
    assert p.value == EMPTY_C


@pytest.mark.parametrize(
    ("field", "raw", "expected"),
    [
        ("age", "67", 67),
        ("age", '"67歳"', 67),
        ("age", '"６７歳"', 67),
        ("age", '"約60〜65歳"', 60),
        ("age", "67.0", 67),
        ("age", "67.5", None),
        ("age", "true", None),
        ("age", '"不明"', None),
        ("age", "null", None),
        ("sex", '"女性"', "女性"),
        ("sex", '" 女性 "', "女性"),
        ("sex", '"女"', None),
        ("sex", '"F"', None),
        ("chief_complaint", '""', None),
        ("chief_complaint", "12", None),
    ],
)
def test_structured_scalar_coercion(field, raw, expected):
    assert parse_prediction("C_structured", f'{{"{field}": {raw}}}').value[field] == expected


# --- multiple choice ---------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "labels", "strict", "lenient"),
    [
        ('{"answer": "C"}', ABCDE, "C", "C"),
        ('{"answer": "C."}', ABCDE, "C", "C"),
        ('{"answer": "B: 肺炎"}', ABCDE, "B", "B"),
        ('{"answer": "C）"}', ABCDE, "C", "C"),
        ('{"answer": "Ｄ"}', ABCDE, "D", "D"),
        ('{"answer": " b "}', ABCDE, "B", "B"),
        ('{"answer": "Bです"}', ABCDE, "B", "B"),  # only ASCII letters make a letter non-standalone
        ('{"answer": "選択肢A"}', ABCDE, "A", "A"),
        ('{"answer": "Cardiology"}', ABCDE, None, None),
        ('{"answer": "E"}', ["A", "B", "C", "D"], None, None),
        ("ADE", ABCDE, None, None),
        ("ADEの所見からBが正しい", ABCDE, None, "B"),
        ("答えはＢです", ABCDE, None, "B"),
        ("D ではなく B", ["A", "B", "C"], None, "B"),
        ('{"answer": "H"}', None, "H", "H"),
        ('{"answer": "I"}', None, None, None),
        ("", ABCDE, None, None),
    ],
)
def test_mc_answer_letters(text, labels, strict, lenient):
    p = parse_prediction("gen_jcqa", text, labels)
    assert (p.strict_answer_letter, p.answer_letter, p.value) == (strict, lenient, lenient)


def test_mc_validity_flags():
    labels = ["A", "B", "C", "D"]
    p = parse_prediction("ext_crade", '{"answer": "C."}', labels)
    assert (p.json_valid, p.schema_valid) == (True, True)
    p = parse_prediction("ext_crade", '{"answer": 3}', labels)
    assert (p.json_valid, p.schema_valid, p.strict_answer_letter, p.answer_letter) == (True, False, None, None)
    p = parse_prediction("ext_crade", "答えはBです", labels)
    assert (p.json_valid, p.schema_valid, p.strict_answer_letter, p.answer_letter) == (False, False, None, "B")


def test_mc_reviewer_regressions_no_wrong_letter():
    from jmedft.evaluation.parsing import parse_prediction

    labels = ["A", "B", "C", "D", "E"]
    # Valid JSON: only the answer field counts; an unusable answer yields None (never a guessed "A").
    assert parse_prediction("ext_crade", '{"answer": "N/A"}', labels).answer_letter is None
    assert parse_prediction("ext_crade", '{"reason": "Aは誤り", "Answer": "C"}', labels).answer_letter is None
    # Prose without JSON: cued letter wins; ranges such as A〜D are ignored; otherwise the last lone letter.
    assert parse_prediction("ext_crade", "選択肢A〜Dのうち、最も適切なのはCです。", labels).answer_letter == "C"
    assert parse_prediction("ext_rrtnm", "A 72 mm mass is present, so the answer is E", labels).answer_letter == "E"
    assert parse_prediction("ext_crade", "答えはB", labels).answer_letter == "B"


def test_doctype_text_fallback_only_without_json():
    from jmedft.evaluation.parsing import parse_prediction

    echoed = '{"label": "不明", "options": "外来診療録, 入院時記録"}'
    assert parse_prediction("B_doctype", echoed).value is None
    assert parse_prediction("B_doctype", '{"label": "診療情報提供書（紹介状）"}').value == "紹介状"
    assert parse_prediction("B_doctype", "この文書は退院時サマリーです").value == "退院時サマリー"
