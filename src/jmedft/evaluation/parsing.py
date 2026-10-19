"""Parse raw model outputs into task predictions (docs/INTERFACES.md, "Parsing rules").

The same rules apply to every model: strip a markdown fence, take the first balanced JSON object, validate it
against the task's JSON Schema and coerce it into the value the scorer consumes.
"""

from __future__ import annotations

import functools
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator

from jmedft import tasks
from jmedft.utils import normalize_text

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_DIGIT_RUN = re.compile(r"[0-9]+")
_STANDALONE_LETTER = re.compile(r"(?<![A-Za-z])[A-Z](?![A-Za-z])")


def strip_reasoning(text: str) -> tuple[str, str | None, bool]:
    """Split a completion into (answer_text, reasoning_text, unterminated), all stripped.

    `<think>R</think>A` -> ("A", "R", False); a `</think>` without an opening tag splits at its last
    occurrence; a `<think>` that is never closed -> ("", R, True); no tags -> (text, None, False).
    """
    close = text.rfind(_THINK_CLOSE)
    if close != -1:
        reasoning = text[:close].replace(_THINK_OPEN, "", 1)
        return text[close + len(_THINK_CLOSE):].strip(), reasoning.strip(), False
    opened = text.find(_THINK_OPEN)
    if opened != -1:
        return "", text[opened + len(_THINK_OPEN):].strip(), True
    return text.strip(), None, False


def _strip_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        newline = s.find("\n")
        s = s[newline + 1:] if newline != -1 else s[3:]
    return s.removesuffix("```").strip()


def _match_braces(s: str, start: int, ends: dict[int, int | None]) -> None:
    """Record in `ends` the `}` closing the `{` at `start` (None if it never closes).

    Braces inside JSON string literals do not count. Every other `{` met on the way is recorded as well: a
    scan starting there would see the same string state, so it would find the same match. This keeps
    `extract_json_object` linear on degenerate outputs such as thousands of unclosed `{`.
    """
    stack: list[int] = []
    in_string = escaped = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            stack.append(i)
        elif ch == "}":
            ends[stack.pop()] = i
            if not stack:
                return
    ends.update(dict.fromkeys(stack))


def extract_json_object(text: str) -> dict[str, Any] | None:
    """First balanced `{...}` in `text` (code fence stripped) that parses to a JSON object, else None."""
    s = _strip_fence(text)
    ends: dict[int, int | None] = {}
    start = s.find("{")
    while start != -1:
        if start not in ends:
            _match_braces(s, start, ends)
        end = ends[start]
        if end is not None:
            try:
                obj = json.loads(s[start:end + 1])
            except (ValueError, RecursionError):
                obj = None
            if isinstance(obj, dict):
                return obj
        start = s.find("{", start + 1)
    return None


@dataclass
class ParsedPrediction:
    json_valid: bool
    schema_valid: bool
    value: Any
    missing_fields: list[str] = field(default_factory=list)
    answer_letter: str | None = None  # lenient (primary) MC answer
    strict_answer_letter: str | None = None  # MC answer taken from the JSON `answer` field only
    whole_json: bool = False  # the entire output (fence stripped) is exactly one JSON object


@functools.cache
def _validator(task: str) -> Draft202012Validator:
    return Draft202012Validator(tasks.schema_for(task))


def _str_list(value: Any) -> list[str]:
    """Coerce to a list of stripped non-empty strings; a lone string becomes a one-item list."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [x.strip() for x in value if isinstance(x, str) and x.strip()]


def _doc_type(label: str) -> str | None:
    norm = normalize_text(label)
    return next((d for d in tasks.DOC_TYPES if normalize_text(d) == norm), None)


def _doc_type_in_text(text: str) -> str | None:
    """Doc-type name occurring earliest in `text`; at equal positions the longer name wins."""
    norm = normalize_text(text)
    best: tuple[int, str] | None = None
    for name in sorted(tasks.DOC_TYPES, key=lambda d: len(normalize_text(d)), reverse=True):
        pos = norm.find(normalize_text(name))
        if pos != -1 and (best is None or pos < best[0]):
            best = (pos, name)
    return best[1] if best else None


def _age(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():  # JSON Schema counts 67.0 as an integer
        return int(value)
    if isinstance(value, str):
        m = _DIGIT_RUN.search(unicodedata.normalize("NFKC", value))
        return int(m.group()) if m else None
    return None


def _sex(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    norm = normalize_text(value)
    return next((s for s in tasks.SEX_VALUES if normalize_text(s) == norm), None)


def _strict_letter(answer: Any, allowed: set[str]) -> str | None:
    """Letter from the JSON `answer`: a lone option letter or one followed by a non-letter ("A.", "C）")."""
    if not isinstance(answer, str):
        return None
    s = unicodedata.normalize("NFKC", answer).strip().upper()
    s = re.sub(r"^(?:選択肢|OPTION|CHOICE|答え|正解)\s*[:：]?\s*", "", s)  # "選択肢A", "Option B"
    if not s or s[0] not in allowed:
        return None
    return s[0] if len(s) == 1 or not (s[1].isascii() and s[1].isalpha()) else None


_ANSWER_CUE = re.compile(r"(?:答え|正解|回答|解答|answer)\s*(?:は|:|：|=|is)?\s*[「『\"'(（]?\s*([A-Z])(?![A-Za-z])",
                         re.IGNORECASE)
_RANGE_NEIGHBOURS = set("/〜~-ー－")


def _lenient_letter(text: str, allowed: set[str]) -> str | None:
    """Fallback for outputs WITHOUT a usable JSON answer: an explicitly cued letter (答えはB / answer: B) wins;
    otherwise the LAST standalone option letter, ignoring letters inside ranges such as "A〜D" or "A/B"."""
    norm = unicodedata.normalize("NFKC", text)
    cued = [m.group(1).upper() for m in _ANSWER_CUE.finditer(norm) if m.group(1).upper() in allowed]
    if cued:
        return cued[-1]
    found = None
    for m in _STANDALONE_LETTER.finditer(norm):
        before = norm[m.start() - 1] if m.start() > 0 else ""
        after = norm[m.end()] if m.end() < len(norm) else ""
        if m.group() in allowed and before not in _RANGE_NEIGHBOURS and after not in _RANGE_NEIGHBOURS:
            found = m.group()
    return found


def is_whole_json_object(text: str) -> bool:
    try:
        return isinstance(json.loads(_strip_fence(text or "").strip()), dict)
    except (ValueError, TypeError):
        return False


def parse_prediction(task: str, output_text: str, labels: list[str] | None = None) -> ParsedPrediction:
    pp = _parse_prediction(task, output_text, labels)
    pp.whole_json = is_whole_json_object(output_text)
    return pp


def _parse_prediction(task: str, output_text: str, labels: list[str] | None = None) -> ParsedPrediction:
    """Parse one model output for `task`. `labels` are the MC option letters (default "A".."H")."""
    text = output_text or ""
    obj = extract_json_object(text)
    json_valid = obj is not None
    schema_valid = json_valid and _validator(task).is_valid(obj)
    src: dict[str, Any] = obj or {}
    family = tasks.TASK_FAMILY[task]
    if family == "entities":
        keys = tasks.A_KEYS if task == "A_entities" else ["entities"]
        return ParsedPrediction(json_valid, schema_valid, {k: _str_list(src.get(k)) for k in keys})
    if family == "classification":
        label = src.get("label")
        value = _doc_type(label) if isinstance(label, str) else None
        if value is None and isinstance(label, str):
            value = _doc_type_in_text(label)  # e.g. "診療情報提供書（紹介状）" → 紹介状
        if value is None and not json_valid:
            value = _doc_type_in_text(text)  # free-text fallback only when no JSON was produced
        return ParsedPrediction(json_valid, schema_valid, value)
    if family == "structured":
        complaint = src.get("chief_complaint")
        value = {
            "age": _age(src.get("age")),
            "sex": _sex(src.get("sex")),
            "chief_complaint": complaint.strip() if isinstance(complaint, str) and complaint.strip() else None,
            **{k: _str_list(src.get(k)) for k in tasks.C_LIST_FIELDS},
        }
        missing = [k for k in tasks.C_FIELDS if k not in src]
        return ParsedPrediction(json_valid, schema_valid, value, missing_fields=missing)
    if family != "mc":
        raise ValueError(f"unknown task family {family!r} for task {task!r}")
    allowed = set(labels) if labels is not None else set(tasks.LETTERS)
    strict = _strict_letter(src.get("answer"), allowed)
    # With valid JSON only the `answer` field counts (contract rule 5); prose fallback only without JSON.
    lenient = strict if json_valid else _lenient_letter(text, allowed)
    return ParsedPrediction(json_valid, schema_valid, lenient, answer_letter=lenient,
                            strict_answer_letter=strict)
