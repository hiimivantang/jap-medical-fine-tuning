"""Re-run the (current) deterministic validator over already-generated notes.

Used when the validator is strengthened after generation: notes that now fail are marked rejected with the
new violation codes (`revalidation_errors`); nothing is edited by hand. The original record is kept, so the
change is auditable, and a summary is written next to the file.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from jmedft.datasets.synth_spec import CATEGORIES, NoteSpec, validate_note
from jmedft.utils import REPO_ROOT, load_yaml, read_jsonl, write_jsonl


def revalidate_file(path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    inv = load_yaml(REPO_ROOT / cfg["term_inventory"])
    all_terms = {t for c in CATEGORIES for t in inv[c]}
    rows = read_jsonl(path)
    newly_rejected: Counter[str] = Counter()
    n_before = sum(r["accepted"] for r in rows)
    for r in rows:
        if not r["accepted"]:
            continue
        errs = validate_note(r["text"], NoteSpec(**r["spec"]), all_terms, cfg["validation"])
        if errs:
            r["accepted"] = False
            r["revalidation_errors"] = errs
            for e in errs:
                newly_rejected[e.split(":")[0]] += 1
    write_jsonl(path, rows)
    summary = {"file": str(path.resolve().relative_to(REPO_ROOT)), "accepted_before": n_before,
               "accepted_after": sum(r["accepted"] for r in rows), "violation_counts": dict(newly_rejected)}
    path.with_suffix(".revalidation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
