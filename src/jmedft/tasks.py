"""Task definitions: prompt templates, label sets and JSON schemas shared by data building, inference,
training and evaluation. Every model sees byte-identical prompts for the same item."""

from __future__ import annotations

import json
from typing import Any

DOC_TYPES: list[str] = ["外来診療録", "入院時記録", "退院時サマリー", "紹介状", "看護記録", "検査・画像所見報告書"]

A_KEYS = ["symptoms", "diseases", "medications"]
C_LIST_FIELDS = [
    "symptoms",
    "negated_symptoms",
    "past_history",
    "family_history",
    "medications",
    "allergies",
    "diagnoses",
    "suspected_diagnosis",
]
C_SCALAR_FIELDS = ["age", "sex", "chief_complaint"]
C_FIELDS = C_SCALAR_FIELDS + C_LIST_FIELDS
SEX_VALUES = ["男性", "女性"]

_COMMON_RULES = "出力はJSONオブジェクトのみとし、説明文やコードブロック記号は付けないでください。"

SYSTEM_PROMPT = (
    "あなたは日本語の医療文書を扱う情報抽出アシスタントです。指示に従い、文書に書かれている内容だけを根拠に回答してください。"
    "診断や治療の推奨は行わないでください。"
)

A_INSTRUCTION = (
    "次の医療文書から、患者本人に現在認められる「症状」「疾患」「薬剤」を抽出してください。\n"
    "規則:\n"
    "- 文書中の表記をそのまま抜き出す（言い換え・正規化をしない）。\n"
    "- symptoms: 患者本人に認められる症状・所見（「〜なし」「〜は認めない」など否定されたものは含めない）。\n"
    "- diseases: 患者本人が罹患している（既往を含む）診断済みの疾患。疑い病名・家族歴の疾患は含めない。\n"
    "- medications: 患者が使用中または使用した薬剤名（用量は含めない）。\n"
    "- 該当がなければ空リストにする。\n"
    f"{_COMMON_RULES}\n"
    '形式: {"symptoms": [...], "diseases": [...], "medications": [...]}'
)

B_INSTRUCTION = (
    "次の医療文書の種類を、以下の選択肢から1つ選んでください。\n"
    f"選択肢: {', '.join(DOC_TYPES)}\n"
    f"{_COMMON_RULES}\n"
    '形式: {"label": "<選択肢のいずれか>"}'
)

C_INSTRUCTION = (
    "次の医療文書から、以下の項目を構造化して抽出してください。\n"
    "規則:\n"
    "- age: 年齢（整数）。記載がなければ null。\n"
    '- sex: "男性" または "女性"。記載がなければ null。\n'
    "- chief_complaint: 主訴（文書中の表記）。記載がなければ null。\n"
    "- symptoms: 患者本人に認められる症状（否定されたものは含めない）。\n"
    "- negated_symptoms: 「〜なし」「〜は認めない」など否定されている症状。\n"
    "- past_history: 既往歴の疾患。\n"
    "- family_history: 家族歴の疾患（続柄は含めない）。\n"
    "- medications: 使用中の薬剤名（用量は含めない）。\n"
    "- allergies: アレルギーの原因物質・薬剤。\n"
    "- diagnoses: 現在の確定診断名。\n"
    "- suspected_diagnosis: 疑い・可能性として記載されている病名。\n"
    "- リスト項目は文書中の表記をそのまま抜き出し、該当がなければ空リストにする。\n"
    f"{_COMMON_RULES}\n"
    "形式: {" + ", ".join(f'"{k}": ...' for k in C_FIELDS) + "}"
)

EXT_NER_INSTRUCTION = (
    "{question}\n"
    "文書中の表記をそのまま抜き出し、該当する表現をすべて列挙してください。\n"
    f"{_COMMON_RULES}\n"
    '形式: {{"entities": [...]}}'
)

MC_INSTRUCTION = (
    "{question}\n\n{options}\n\n"
    "最も適切な選択肢を1つ選び、その記号で答えてください。\n"
    f"{_COMMON_RULES}\n"
    '形式: {{"answer": "<記号>"}}'
)

LETTERS = "ABCDEFGH"


def _doc_block(text: str) -> str:
    return f"文書:\n「{text}」"


def build_messages(task: str, *, text: str | None = None, question: str | None = None,
                   options: list[str] | None = None) -> list[dict[str, str]]:
    """Return chat messages for a task. `text` is the source document; MC tasks pass question/options."""
    if task == "A_entities":
        user = f"{A_INSTRUCTION}\n\n{_doc_block(text)}"
    elif task == "B_doctype":
        user = f"{B_INSTRUCTION}\n\n{_doc_block(text)}"
    elif task == "C_structured":
        user = f"{C_INSTRUCTION}\n\n{_doc_block(text)}"
    elif task in ("ext_mrner_disease", "ext_mrner_medicine"):
        user = EXT_NER_INSTRUCTION.format(question=question)
    elif task in ("ext_crade", "ext_rrtnm", "gen_jcqa", "gen_jnli", "gen_jmmlu_med"):
        opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options or []))
        user = MC_INSTRUCTION.format(question=question, options=opts)
    else:
        raise ValueError(f"unknown task {task}")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _str_list() -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}}


SCHEMAS: dict[str, dict[str, Any]] = {
    "A_entities": {
        "type": "object",
        "required": A_KEYS,
        "properties": {k: _str_list() for k in A_KEYS},
    },
    "B_doctype": {
        "type": "object",
        "required": ["label"],
        "properties": {"label": {"type": "string", "enum": DOC_TYPES}},
    },
    "C_structured": {
        "type": "object",
        "required": C_FIELDS,
        "properties": {
            "age": {"type": ["integer", "null"]},
            "sex": {"type": ["string", "null"], "enum": SEX_VALUES + [None]},
            "chief_complaint": {"type": ["string", "null"]},
            **{k: _str_list() for k in C_LIST_FIELDS},
        },
    },
    "entities": {"type": "object", "required": ["entities"], "properties": {"entities": _str_list()}},
    "mc": {"type": "object", "required": ["answer"], "properties": {"answer": {"type": "string"}}},
}

TASK_FAMILY = {
    "A_entities": "entities",
    "B_doctype": "classification",
    "C_structured": "structured",
    "ext_mrner_disease": "entities",
    "ext_mrner_medicine": "entities",
    "ext_crade": "mc",
    "ext_rrtnm": "mc",
    "gen_jcqa": "mc",
    "gen_jnli": "mc",
    "gen_jmmlu_med": "mc",
}


def schema_for(task: str) -> dict[str, Any]:
    if task in SCHEMAS:
        return SCHEMAS[task]
    return SCHEMAS["entities"] if TASK_FAMILY[task] == "entities" else SCHEMAS["mc"]


def gold_to_target_json(task: str, gold: dict[str, Any]) -> str:
    """Canonical assistant target used for SFT (same JSON shape the evaluator expects)."""
    if task == "A_entities":
        obj = {k: gold[k] for k in A_KEYS}
    elif task == "B_doctype":
        obj = {"label": gold["label"]}
    elif task == "C_structured":
        obj = {k: gold[k] for k in C_FIELDS}
    elif TASK_FAMILY[task] == "entities":
        obj = {"entities": gold["entities"]}
    else:
        obj = {"answer": gold["answer"]}
    return json.dumps(obj, ensure_ascii=False)
