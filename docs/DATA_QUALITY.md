# Synthetic data quality review

All synthetic notes describe fictional patients. Gold labels come from the sampled specification, not from
reading the generated text; a deterministic validator checks that the text actually supports every label.

## Pipeline yield

| Split | Generator | Specs | Accepted at generation | Accepted after re-validation | Notes used |
|---|---|---|---|---|---|
| train | Qwen3.8-27B-FP8 | 1,800 | 1,609 (89.4%) | 1,583 | 1,583 × 3 tasks |
| val | Qwen3.8-27B-FP8 | 180 | 161 (89.4%) | 157 | 157 × 3 tasks |
| test_synth | Qwen3.8-27B-FP8 | 300 | 269 (89.7%) | 267 | 267 × 3 tasks |
| test_xgen | Nemotron-3.5-Lightning | 1,000 | see `data/synthetic/test_xgen.raw.stats.json` | see `…revalidation.json` | |

Up to 3 attempts per spec (6 for Nemotron) with a different sampling seed each time; a spec is dropped
if every attempt fails. No note was edited by hand.

Doc-type distribution (train): 外来診療録 300, 退院時サマリー 273, 入院時記録 264, 検査・画像所見報告書 262,
紹介状 257, 看護記録 227. Median length ≈ 200 characters; ≈ 7 task-A entities per note.

## Validator checks (deterministic, `src/jmedft/datasets/synth_spec.py::validate_note`)

1. Every spec term appears verbatim in the note (after NFKC normalisation).
2. Negated symptoms are followed by a negation cue (なし／認めない／…) within 20 characters.
3. Positive symptoms are not exclusively negated.
4. Suspected diagnoses carry a hedge (疑い／可能性／否定できない／…) in their own clause; confirmed
   diagnoses and past history must have at least one un-hedged mention.
5. No inventory term outside the spec appears (prevents silent extra gold entities).
6. Age and sex consistent with the spec (present when specified, absent when not).
7. No PII patterns (phone numbers, full calendar dates, postcodes).
8. No echo of internal labels (現在の診断, 疑い病名, …) and no document-type title (makes task B non-trivial).
9. Section context: a current diagnosis must be mentioned at least once outside a 既往歴 block / inline
   past-history cue, and a past-history disease must appear inside one.

## Manual review findings (36 notes: 12 train, 5 val, 5 test_synth smoke, 14 test_synth/train spot checks)

| Finding | Action |
|---|---|
| Natural clinical Japanese; SOAP structure for 外来診療録, polite referral language (ご高診のほど…) for 紹介状, findings/impression structure for 検査・画像所見報告書. | — |
| Early prompt version echoed spec field names (「評価：…」「現在の診断：…」) and wrote doc-type titles (「外来診療録」「診療情報提供書」). | Prompt rules 14–16 added; validator checks 8 added. Regenerated all data. |
| Hedge check flagged a confirmed diagnosis when the *next* term was hedged (「大腸癌。緑内障の可能性」). | Hedge window limited to the mention's own clause; regression test added. |
| **Label-fidelity bug found after generation:** ~8% of notes with current diagnoses listed them only under a 既往歴 heading on its own line (「既往歴⏎脳出血」), which the clause-based check missed. Task C gold would have said `diagnoses` while the text says past history. | Section-aware check 9 added (with explicit heading vocabulary, after two over-strict iterations that were caught by spot checks); all generated notes re-validated: train 1,609 → 1,583, val 161 → 157, test_synth 269 → 267. The full training run that had started was stopped (step ~15 of 610) and restarted on the corrected data. |
| Random spec sampling produces clinically unusual combinations (e.g. thyroid ultrasound impression suggesting 乳癌; 4 unrelated drugs in a 30-year-old). | Accepted: extraction tasks must read what is written, and implausible combinations prevent shortcutting via medical priors. Documented as a limitation. |
| 看護記録 notes only partly read as nursing documentation (some are physician-style summaries with a care sentence). | Accepted; this caps task-B accuracy for **all** models equally. Doc-type style is a known weak signal; see report. |
| Accepted Nemotron notes skew toward simpler specs (Nemotron drops required terms more often; 25% acceptance at 6 attempts in a 40-spec pilot). | test_xgen is interpreted only as within-set comparisons between conditions. |

## Known limitations

- Synthetic notes are shorter (median ~200 chars) and cleaner than real clinical documents.
- Label noise is not zero: the validator is lexical; e.g. a term mentioned in a negated *and* an affirmed
  clause can pass check 3. Spot checks after the final validator found no remaining label errors in 12
  sampled rejections/acceptances, but the residual error rate is not measured.
- Generator-style bias: training and test_synth share a generator. test_xgen (different model family) and
  test_ext (human-written, human-annotated) exist to measure how much of any gain survives a change of
  author.
