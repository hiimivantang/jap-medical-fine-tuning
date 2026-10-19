"""Shared helpers: text normalisation, JSONL I/O, hashing, config loading, secrets-safe env loading."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """NFKC-normalise and remove all whitespace (full-width digits/letters become ASCII)."""
    return _WS.sub("", unicodedata.normalize("NFKC", s))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_env() -> None:
    """Load REPO_ROOT/.env into os.environ without overriding existing values. Values are never printed."""
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)


def redact(text: str) -> str:
    """Remove any known secret values from a string before it is logged or written to a report."""
    for key in ("OPENAI_API_KEY", "FIREWORKS_API_KEY", "HUGGINGFACE_KEY", "HF_TOKEN"):
        val = os.environ.get(key)
        if val and len(val) >= 8:
            text = text.replace(val, f"<{key}:redacted>")
    return text


def char_ngrams(s: str, n: int = 5) -> set[str]:
    s = normalize_text(s)
    return {s[i : i + n] for i in range(max(0, len(s) - n + 1))}


def containment(a: set[str], b: set[str]) -> float:
    """|a ∩ b| / min(|a|, |b|): 1.0 when one text is (nearly) contained in the other."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
