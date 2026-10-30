"""English edition of the Experiment 01 task prompts (addendum 2026-10-30, post hoc).

Literal translations of jmedft.tasks: every rule, field and output format is kept. The JSON keys and output
schemas are unchanged, so the same parser and scorers apply. Fixed vocabularies (document types, sex) map to
fixed English terms; gold labels are converted with these maps, never re-translated.
"""

from __future__ import annotations

from typing import Any

import jmedft.tasks as ja

SYSTEM_PROMPT = ("You are an information-extraction assistant for Japanese clinical documents written in English. "
                 "Follow the instructions and answer only on the basis of what is written in the document. "
                 "Do not give diagnoses or treatment recommendations.")
_JSON_ONLY = "Output only the JSON object, with no explanation and no code-block markers."

DOC_TYPES_EN = {"外来診療録": "outpatient record", "入院時記録": "admission record", "退院時サマリー": "discharge summary",
                "紹介状": "referral letter", "看護記録": "nursing record", "検査・画像所見報告書": "test and imaging report"}
SEX_EN = {"男性": "male", "女性": "female"}
assert list(DOC_TYPES_EN) == list(ja.DOC_TYPES) and list(SEX_EN) == list(ja.SEX_VALUES)

A_INSTRUCTION = (
    "Extract the patient's own current \"symptoms\", \"diseases\" and \"medications\" from the following clinical "
    "document.\nRules:\n"
    "- Copy the wording exactly as it appears in the document (do not paraphrase or normalise).\n"
    "- symptoms: symptoms and findings present in the patient (exclude negated ones such as \"no ...\" or "
    "\"... not observed\").\n"
    "- diseases: diagnosed diseases the patient has (including past history). Exclude suspected diagnoses and "
    "family-history diseases.\n"
    "- medications: names of drugs the patient is using or has used (without doses).\n"
    "- If there are none, use an empty list.\n"
    f"{_JSON_ONLY}\n"
    'Format: {"symptoms": [...], "diseases": [...], "medications": [...]}'
)

B_INSTRUCTION = (
    "Choose the type of the following clinical document from the options below.\n"
    "Options: " + ", ".join(DOC_TYPES_EN.values()) + "\n"
    f"{_JSON_ONLY}\n"
    'Format: {"label": "<one of the options>"}'
)

C_INSTRUCTION = (
    "Extract the following items from the clinical document as structured data.\nRules:\n"
    "- age: age (integer). null if not stated.\n"
    "- sex: \"male\" or \"female\". null if not stated.\n"
    "- chief_complaint: chief complaint (as written in the document). null if not stated.\n"
    "- symptoms: symptoms present in the patient (exclude negated ones).\n"
    "- negated_symptoms: symptoms that are negated, such as \"no ...\" or \"... not observed\".\n"
    "- past_history: diseases in the past medical history.\n"
    "- family_history: diseases in the family history (without the relationship).\n"
    "- medications: names of drugs in use (without doses).\n"
    "- allergies: allergens or drugs the patient is allergic to.\n"
    "- diagnoses: current confirmed diagnoses.\n"
    "- suspected_diagnosis: diagnoses described as suspected or possible.\n"
    "- For list items, copy the wording exactly as it appears in the document; if there are none, use an empty list.\n"
    f"{_JSON_ONLY}\n"
    'Format: {"age": ..., "sex": ..., "chief_complaint": ..., "symptoms": ..., "negated_symptoms": ..., '
    '"past_history": ..., "family_history": ..., "medications": ..., "allergies": ..., "diagnoses": ..., '
    '"suspected_diagnosis": ...}'
)

EXT_NER_INSTRUCTION = (
    "{question}\n"
    "Copy the wording exactly as it appears in the document and list every matching expression.\n"
    f"{_JSON_ONLY}\n"
    'Format: {{"entities": [...]}}'
)

MC_INSTRUCTION = (
    "{question}\n\n{options}\n\n"
    "Choose the single most appropriate option and answer with its letter.\n"
    f"{_JSON_ONLY}\n"
    'Format: {{"answer": "<letter>"}}'
)


def _doc_block(text: str) -> str:
    return f"Document:\n\"{text}\""


def build_messages(task: str, *, text: str | None = None, question: str | None = None,
                   options: list[str] | None = None) -> list[dict[str, str]]:
    """English counterpart of jmedft.tasks.build_messages (same task names, same output schemas)."""
    if task == "A_entities":
        user = f"{A_INSTRUCTION}\n\n{_doc_block(text)}"
    elif task == "B_doctype":
        user = f"{B_INSTRUCTION}\n\n{_doc_block(text)}"
    elif task == "C_structured":
        user = f"{C_INSTRUCTION}\n\n{_doc_block(text)}"
    elif task in ("ext_mrner_disease", "ext_mrner_medicine"):
        user = EXT_NER_INSTRUCTION.format(question=question)
    elif task in ("ext_crade", "ext_rrtnm"):
        opts = "\n".join(f"{ja.LETTERS[i]}. {o}" for i, o in enumerate(options or []))
        user = MC_INSTRUCTION.format(question=question, options=opts)
    else:
        raise ValueError(f"unknown task {task}")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def gold_to_english(task: str, gold: dict[str, Any], span_map: dict[str, str]) -> dict[str, Any]:
    """Map a Japanese gold label to English: closed vocabularies via the fixed maps, spans via `span_map`
    (Japanese span -> English span as it appears in the translated document). Raises KeyError on a missing span."""
    if task == "B_doctype":
        return {"label": DOC_TYPES_EN[gold["label"]]}
    if task in ("ext_crade", "ext_rrtnm"):
        return dict(gold)  # answer letters are language-independent
    if task in ("ext_mrner_disease", "ext_mrner_medicine"):
        return {"entities": [span_map[s] for s in gold["entities"]]}
    out: dict[str, Any] = {}
    for k, v in gold.items():
        if k == "sex":
            out[k] = SEX_EN[v] if v is not None else None
        elif k == "age" or v is None:
            out[k] = v
        elif isinstance(v, list):
            out[k] = [span_map[s] for s in v]
        else:
            out[k] = span_map[v]
    return out
