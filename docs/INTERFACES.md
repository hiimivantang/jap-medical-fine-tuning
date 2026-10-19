# Internal interfaces (data contracts between pipeline stages)

All files are UTF-8 JSON Lines. Strings are compared after `jmedft.utils.normalize_text` (NFKC + strip all
whitespace) unless stated otherwise.

## Eval item (`data/processed/eval/<split>/<task>.jsonl`)

```json
{"id": "synth-test-000123", "task": "A_entities", "split": "test_synth", "source": "synthetic:qwen3.8-27b",
 "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
 "gold": {...}, "meta": {"doc_type": "外来診療録", "source_text": "...", "unseen_terms": ["..."]}}
```

`meta.source_text` is the raw document the model reads (used for "unsupported extraction" checks).

## Tasks and gold formats

| task | family | gold | expected model output (JSON object) |
|---|---|---|---|
| `A_entities` | entity sets | `{"symptoms": [..], "diseases": [..], "medications": [..]}` | same keys |
| `B_doctype` | classification | `{"label": "<one of 6 doc types>"}` | `{"label": "..."}` |
| `C_structured` | structured | 11 fields (below) | same keys |
| `ext_mrner_disease` | entity sets | `{"entities": [..]}` | `{"entities": [..]}` |
| `ext_mrner_medicine` | entity sets | `{"entities": [..]}` | `{"entities": [..]}` |
| `ext_crade` | multiple choice | `{"answer": "A", "labels": ["A","B","C","D"]}` | `{"answer": "A"}` |
| `ext_rrtnm` | multiple choice | `{"answer": "B", "labels": ["A",..]}` | `{"answer": "B"}` |
| `gen_jcqa` | multiple choice | `{"answer": "C", "labels": ["A".."E"]}` | `{"answer": "C"}` |
| `gen_jnli` | multiple choice | `{"answer": "A", "labels": ["A","B","C"]}` | `{"answer": "A"}` |
| `gen_jmmlu_med` | multiple choice | `{"answer": "D", "labels": ["A".."D"]}` | `{"answer": "D"}` |

Task C fields (all keys required in a schema-compliant answer):
`age` (int|null), `sex` ("男性"|"女性"|null), `chief_complaint` (str|null), and lists of str:
`symptoms`, `negated_symptoms`, `past_history`, `family_history`, `medications`, `allergies`,
`diagnoses`, `suspected_diagnosis`.

## Prediction record (`experiments/01_finetuning/results/predictions/<condition>/<split>/<task>.jsonl`)

```json
{"id": "...", "task": "...", "condition": "E1_qwen3_8b", "model": "qwen3-8b", "output_text": "...",
 "reasoning_text": null, "finish_reason": "stop", "latency_s": 0.84, "prompt_tokens": 412,
 "completion_tokens": 37, "error": null}
```

`output_text` is the final answer text with any `<think>…</think>` block removed; `reasoning_text` holds the
removed reasoning (or the reasoning field returned by the server). A response in which a `<think>` block is
opened but never closed has `output_text = ""`.

## Parsing rules (identical for every model)

1. Strip a leading/trailing markdown code fence if present.
2. Parse the first balanced top-level JSON object found in the text (`json_valid`).
3. `schema_valid` = the parsed object satisfies the task's JSON Schema (required keys and types).
4. Entity/structured tasks: a non-parseable output is scored as an empty prediction.
5. Multiple-choice tasks: the answer letter is taken from the JSON `answer` field; if JSON is invalid, a
   lenient fallback takes the first standalone option letter in the text. Both strict and lenient accuracy
   are reported; lenient is the primary metric.
6. Classification (`B_doctype`): label from JSON `label`; if absent/invalid, the first doc-type name that
   occurs in the text; otherwise counted as invalid (wrong).

## Scoring

- Entity sets: per item, gold and predicted values are de-duplicated after normalisation; TP/FP/FN summed
  over items (micro) and per type; document exact match = every type set equal.
- Structured (C): scalar fields exact match after normalisation (`age` compared as int); list fields
  set-equality for field accuracy plus micro P/R/F1 over values; document EM = all 11 fields correct;
  `missing_fields` = required keys absent; `unsupported` = predicted list/str values whose normalised form
  does not occur in normalised `meta.source_text`.
- Classification/MC: accuracy, macro-F1, per-class precision/recall/F1, confusion matrix, invalid rate.

## Statistics

- 95% CI: percentile bootstrap over items, 10,000 resamples, seed 0.
- Paired differences between conditions on the same items: paired bootstrap (same resampled indices).
- Paired accuracy: exact McNemar test. Multiple comparisons: Holm within each hypothesis family.
