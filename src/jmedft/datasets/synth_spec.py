"""Synthetic note specifications: sampling (gold labels come from the spec, not from the text), generator
prompt construction, and deterministic validation of generated notes against their spec."""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from jmedft.tasks import C_FIELDS, DOC_TYPES
from jmedft.utils import normalize_text

CATEGORIES = ["symptoms", "diseases", "medications", "allergens"]


@dataclass
class NoteSpec:
    spec_id: str
    doc_type: str
    age: int | None
    sex: str | None
    chief_complaint: str | None
    symptoms: list[str]
    negated_symptoms: list[str]
    past_history: list[str]
    family_history: list[dict[str, str]]  # [{"relation": "父", "disease": "胃癌"}]
    medications: list[str]
    allergies: list[str]
    diagnoses: list[str]
    suspected_diagnosis: list[str]
    unseen_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_term_pools(inventory: dict[str, list[str]], heldout_fraction: float, seed: int) -> dict[str, dict]:
    """Deterministically split each category into a train pool and a held-out (test-only) pool."""
    pools: dict[str, dict] = {}
    for cat in CATEGORIES:
        terms = sorted(set(inventory[cat]))
        rng = random.Random(f"{seed}:{cat}")
        rng.shuffle(terms)
        k = max(1, round(len(terms) * heldout_fraction))
        pools[cat] = {"heldout": sorted(terms[:k]), "train": sorted(terms[k:])}
    return pools


def _rand_n(rng: random.Random, bounds: list[int]) -> int:
    return rng.randint(bounds[0], bounds[1])


def sample_spec(spec_id: str, rng: random.Random, pools: dict[str, dict], cfg: dict[str, Any],
                term_pool: str, mixed_heldout_prob: float, relations: list[str]) -> NoteSpec:
    """Sample one note spec. term_pool: 'train' (train pools only) or 'mixed' (per category, the held-out
    pool is used with probability mixed_heldout_prob)."""

    def pool(cat: str) -> tuple[list[str], bool]:
        if term_pool == "train":
            return pools[cat]["train"], False
        use_heldout = rng.random() < mixed_heldout_prob
        return (pools[cat]["heldout"] if use_heldout else pools[cat]["train"]), use_heldout

    used: set[str] = set()
    unseen: list[str] = []

    def draw(cat: str, n: int) -> list[str]:
        terms, heldout = pool(cat)
        avail = [t for t in terms if t not in used]
        out = rng.sample(avail, min(n, len(avail)))
        used.update(out)
        if heldout:
            unseen.extend(out)
        return out

    sc = cfg
    symptoms = draw("symptoms", _rand_n(rng, sc["n_symptoms"]))
    negated = draw("symptoms", _rand_n(rng, sc["n_negated_symptoms"]))
    past = draw("diseases", _rand_n(rng, sc["n_past_history"]))
    diagnoses = draw("diseases", _rand_n(rng, sc["n_diagnoses"]))
    suspected = draw("diseases", _rand_n(rng, sc["n_suspected"]))
    fam_diseases = draw("diseases", _rand_n(rng, sc["n_family_history"]))
    family = [{"relation": rng.choice(relations), "disease": d} for d in fam_diseases]
    meds = draw("medications", _rand_n(rng, sc["n_medications"]))
    allergies = draw("allergens", _rand_n(rng, sc["n_allergies"]))
    age = None if rng.random() < sc["p_age_missing"] else rng.randint(*sc["age_range"])
    sex = None if rng.random() < sc["p_sex_missing"] else rng.choice(["男性", "女性"])
    chief = symptoms[0] if symptoms and rng.random() < sc["p_chief_complaint"] else None
    return NoteSpec(
        spec_id=spec_id,
        doc_type=rng.choice(DOC_TYPES),
        age=age,
        sex=sex,
        chief_complaint=chief,
        symptoms=symptoms,
        negated_symptoms=negated,
        past_history=past,
        family_history=family,
        medications=meds,
        allergies=allergies,
        diagnoses=diagnoses,
        suspected_diagnosis=suspected,
        unseen_terms=sorted(set(unseen)),
    )


GENERATOR_RULES = """以下の仕様に厳密に従い、架空の患者についての日本語の医療文書を1つ作成してください。

文書の種類: {doc_type}
文体・構成: {style}

必ず守る規則:
1. 仕様の各用語は、文書中に一字一句そのままの表記で記載する（言い換え・略語化・送り仮名の変更をしない）。
2. 「症状」は患者に現在認められる症状として記載する。
3. 「否定された症状」は「〜は認めない」「〜なし」など、否定であることが明確な形で記載する。
4. 「既往歴」は過去に診断された疾患として記載する。
5. 「家族歴」は続柄とともに家族の病気として記載し、患者本人の病気として書かない。
6. 「現在の診断」は確定した診断名として記載する。
7. 「疑い病名」は「〜の疑い」「〜の可能性」として記載し、確定させない。
8. 「内服薬」は現在使用中の薬剤として薬剤名を記載する（用量を付けてもよい）。
9. 「アレルギー」はアレルギーの原因として記載する。
10. 仕様にない症状・疾患・薬剤・アレルギーは一切書かない。空の項目は書かない（「特記事項なし」も不要）。
11. 年齢・性別は仕様どおりに記載し、仕様で「記載しない」となっている場合は文書に書かない。
12. 患者氏名、医療機関名、医師名、住所、電話番号、具体的な日付など、個人や施設を特定できる情報は書かない。
13. 文書本文のみを出力する（前置き・説明・見出し記号「#」は不要）。
14. 文書の種類名（「{doc_type}」「診療情報提供書」など）をタイトルや本文に書かない。文書の種類は文体と構成だけで表現する。
15. 仕様の項目名（「現在の診断」「疑い病名」「否定された症状」「確定診断」など）をそのまま文中に書かない。実際の臨床文書らしい自然な表現にする。
16. 「現在の診断」の病名は、既往歴ではなく今回の診断・評価として記載する（「既往歴」欄に書かない）。「既往歴」の病名は過去の病気として記載する。

仕様:
{spec_json}
"""

_FIELD_LABELS = {
    "age": "年齢",
    "sex": "性別",
    "chief_complaint": "主訴",
    "symptoms": "症状",
    "negated_symptoms": "否定された症状",
    "past_history": "既往歴",
    "family_history": "家族歴",
    "medications": "内服薬",
    "allergies": "アレルギー",
    "diagnoses": "現在の診断",
    "suspected_diagnosis": "疑い病名",
}


def generator_prompt(spec: NoteSpec, styles: dict[str, str]) -> str:
    view: dict[str, Any] = {}
    for key, label in _FIELD_LABELS.items():
        val = getattr(spec, key)
        if key == "age":
            view[label] = f"{val}歳" if val is not None else "記載しない"
        elif key == "sex":
            view[label] = val if val is not None else "記載しない"
        elif key == "chief_complaint":
            if val is not None:
                view[label] = val
        elif key == "family_history":
            if val:
                view[label] = [f"{x['relation']}: {x['disease']}" for x in val]
        elif val:
            view[label] = val
    return GENERATOR_RULES.format(doc_type=spec.doc_type, style=styles[spec.doc_type],
                                  spec_json=json.dumps(view, ensure_ascii=False, indent=1))


def gold_from_spec(spec: NoteSpec) -> dict[str, dict[str, Any]]:
    """Gold labels for tasks A/B/C derived from the spec (task A 'diseases' = past history + diagnoses)."""
    c_gold = {
        "age": spec.age,
        "sex": spec.sex,
        "chief_complaint": spec.chief_complaint,
        "symptoms": list(spec.symptoms),
        "negated_symptoms": list(spec.negated_symptoms),
        "past_history": list(spec.past_history),
        "family_history": [x["disease"] for x in spec.family_history],
        "medications": list(spec.medications),
        "allergies": list(spec.allergies),
        "diagnoses": list(spec.diagnoses),
        "suspected_diagnosis": list(spec.suspected_diagnosis),
    }
    assert list(c_gold) == C_FIELDS
    return {
        "A_entities": {
            "symptoms": list(spec.symptoms),
            "diseases": list(spec.past_history) + list(spec.diagnoses),
            "medications": list(spec.medications),
        },
        "B_doctype": {"label": spec.doc_type},
        "C_structured": c_gold,
    }


_AGE_RE = re.compile(r"(\d{1,3})\s*(?:歳|才)")
_PII_RES = [
    re.compile(r"\d{2,4}-\d{2,4}-\d{3,4}"),  # phone numbers
    re.compile(r"(?:19|20)\d{2}年\d{1,2}月\d{1,2}日"),  # full calendar dates
    re.compile(r"(?:令和|平成|昭和)\d{1,2}年\d{1,2}月\d{1,2}日"),
    re.compile(r"〒\s?\d{3}-?\d{4}"),
]


def clean_generated_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text).strip()
    return text


def validate_note(text: str, spec: NoteSpec, all_terms: set[str], vcfg: dict[str, Any]) -> list[str]:
    """Return a list of violation codes (empty = valid). Pure string checks, no model involved."""
    errs: list[str] = []
    norm = normalize_text(text)
    if not (vcfg["min_chars"] <= len(text) <= vcfg["max_chars"]):
        errs.append("length")
    required = (spec.symptoms + spec.negated_symptoms + spec.past_history + spec.medications + spec.allergies
                + spec.diagnoses + spec.suspected_diagnosis + [x["disease"] for x in spec.family_history])
    for term in required:
        if normalize_text(term) not in norm:
            errs.append(f"missing:{term}")
    # Negated symptoms: some negation cue must follow a mention within the window.
    for term in spec.negated_symptoms:
        t = normalize_text(term)
        ok = False
        for m in re.finditer(re.escape(t), norm):
            window = norm[m.end(): m.end() + vcfg["negation_window"]]
            if any(normalize_text(cue) in window for cue in vcfg["negation_cues"]):
                ok = True
                break
        if not ok:
            errs.append(f"negation:{term}")
    # Positive symptoms must not be immediately negated.
    for term in spec.symptoms:
        t = normalize_text(term)
        mentions = [m for m in re.finditer(re.escape(t), norm)]
        if mentions and all(
            any(normalize_text(c) in norm[m.end(): m.end() + 6] for c in ("なし", "認めない", "認めず", "無し"))
            for m in mentions
        ):
            errs.append(f"positive_negated:{term}")
    # Suspected diagnoses must carry a hedge cue nearby; confirmed ones must have an unhedged mention.
    hedges = ("疑", "可能性", "否定できない", "考えられ", "示唆")
    other_terms = [normalize_text(t) for t in all_terms]

    def hedged(m: re.Match) -> bool:
        # The hedge must belong to this mention: stop at the first clause boundary or the next term.
        window = re.split(r"[。、，,；;）)]", norm[m.end(): m.end() + 12])[0]
        cut = min((window.find(t) for t in other_terms if t and t in window), default=len(window))
        return any(cue in window[:cut] for cue in hedges)

    for term in spec.suspected_diagnosis:
        if not any(hedged(m) for m in re.finditer(re.escape(normalize_text(term)), norm)):
            errs.append(f"suspected_unhedged:{term}")
    for term in spec.diagnoses + spec.past_history:
        mentions = list(re.finditer(re.escape(normalize_text(term)), norm))
        if mentions and all(hedged(m) for m in mentions):
            errs.append(f"confirmed_hedged:{term}")
    # Spec-external terms from the inventory must not appear (prevents silent extra gold entities).
    spec_terms = {normalize_text(t) for t in required}
    for term in all_terms:
        nt = normalize_text(term)
        if nt in spec_terms or any(nt in st for st in spec_terms):
            continue
        if nt in norm:
            errs.append(f"extra:{term}")
    # Age / sex consistency.
    ages = {int(a) for a in _AGE_RE.findall(norm)}
    if spec.age is None and ages:
        errs.append("age_present_but_unspecified")
    if spec.age is not None and spec.age not in ages:
        errs.append("age_missing")
    if spec.sex is None and ("男性" in norm or "女性" in norm):
        errs.append("sex_present_but_unspecified")
    if spec.sex is not None and spec.sex not in norm:
        errs.append("sex_missing")
    if spec.sex == "男性" and "女性" in norm or spec.sex == "女性" and "男性" in norm:
        errs.append("sex_conflict")
    for rx in _PII_RES:
        if rx.search(text):
            errs.append("pii_pattern")
            break
    # Leaks that would make tasks trivial or echo the prompt's internal labels.
    if any(normalize_text(lbl) in norm for lbl in _LABEL_ECHOES):
        errs.append("label_echo")
    if any(normalize_text(name) in norm for name in _DOC_TYPE_TITLES):
        errs.append("doctype_title")
    # Chief complaint: gold null ⇒ the note must not state a 主訴; gold set ⇒ every stated 主訴 contains the gold
    # term and no other spec symptom (otherwise the field's gold would disagree with the text).
    cc_texts = [m.group(1) for m in _CC_RE.finditer(text)]
    if spec.chief_complaint is None:
        if cc_texts or "主訴" in text:
            errs.append("cc_stated_but_unspecified")
    else:
        if not cc_texts and "主訴" not in text:
            errs.append("cc_missing")
        others = [normalize_text(s) for s in spec.symptoms if s != spec.chief_complaint]
        for cc in cc_texts:
            ncc = normalize_text(cc)
            if normalize_text(spec.chief_complaint) not in ncc:
                errs.append("cc_mismatch")
            elif any(o in ncc for o in others):
                errs.append("cc_multiple_symptoms")
    # A current diagnosis must be mentioned at least once outside a past-history context (inline cue such as
    # 「既往歴として」 or a 既往歴 section heading); a past-history disease must appear in such a context.
    contexts = _mention_contexts(text)
    for term in spec.diagnoses:
        ctx = contexts(term)
        if ctx and all(c == "past" for c in ctx):
            errs.append(f"dx_as_past_history:{term}")
    for term in spec.past_history:
        ctx = contexts(term)
        if ctx and "past" not in ctx:
            errs.append(f"past_history_not_marked:{term}")
    return errs


_INLINE_PAST_CUES = ("既往", "過去の病歴", "過去に", "以前に診断", "以前より")
# A heading-only line, e.g. "既往歴", "【既往歴】", "■ 既往歴：", "A" (SOAP), "入院時の評価". Heading words come
# from an explicit vocabulary so that a one-word content line such as "脳出血" is never taken for a heading.
_HEADING_WORDS = ("既往", "主訴", "現病歴", "家族歴", "内服", "処方", "アレルギー", "診断", "評価", "印象", "所見",
                  "現症", "経過", "方針", "計画", "検査", "治療", "状態", "概要", "目的", "観察", "バイタル", "看護")
_HEADING_ONLY_RE = re.compile(
    r"^[\s【\[■●#]*((?:[^\s:：。、】\]]{0,6}(?:" + "|".join(_HEADING_WORDS) + r")[^\s:：。、】\]]{0,6})|[SOAP])"
    r"[】\]]?\s*[:：]?\s*$")


def _mention_contexts(text: str):
    """Return f(term) -> list of 'past'/'other' per mention.

    A heading-only line containing 既往 opens a past-history block that ends at the next blank line or the
    next heading-only line; an inline "既往…" cue anywhere in the same sentence also marks the mention past.
    """
    tagged: list[tuple[str, str]] = []  # (context, normalised line)
    section = "other"
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            section = "other"
            continue
        m = _HEADING_ONLY_RE.match(line)
        if m:
            section = "past" if any(cue in m.group(1) for cue in _INLINE_PAST_CUES) else "other"
            continue
        tagged.append((section, normalize_text(line)))

    def contexts(term: str) -> list[str]:
        # A mention is "past" inside a 既往 section, or when its sentence carries a 既往 cue before it
        # (「既往歴として高血圧症」) or after it (「高血圧症の既往あり」「…、高血圧症の既往を有する」).
        t = normalize_text(term)
        out = []
        for ctx, line in tagged:
            for clause in re.split(r"[。]", line):
                start = clause.find(t)
                while start != -1:
                    inline_past = any(cue in clause for cue in _INLINE_PAST_CUES)
                    out.append("past" if ctx == "past" or inline_past else "other")
                    start = clause.find(t, start + 1)
        return out

    return contexts


# Text stated in a chief-complaint field: "主訴：X", "【主訴】⏎X", "主訴は X". (Inline "Xを主訴に" is not captured;
# it still counts as a stated 主訴 for the gold-null check, which looks for "主訴" anywhere.)
_CC_RE = re.compile(r"主訴\s*[】\]]?\s*(?:[:：]|は|\n)\s*([^\n。]+)")
_LABEL_ECHOES = ["現在の診断", "疑い病名", "否定された症状", "確定診断である", "仕様"]
# Document-type names (and the formal name of a referral letter) must not appear as titles/text.
_DOC_TYPE_TITLES = DOC_TYPES + ["診療情報提供書", "退院サマリー", "外来記録"]
