# Japanese medical LoRA fine-tuning on NVIDIA DGX Spark — Experiment 01

Can a small open-weight model (Qwen3-8B) fine-tuned with LoRA on Japanese medical extraction/classification
tasks match or beat (a) the untuned model, (b) a Japanese-specialised model of the same size
(Qwen3-Swallow-8B) and (c) an OpenAI reference — without losing general Japanese ability?

- Report: [`REPORT.md`](REPORT.md) · customer summary: [`CUSTOMER_SUMMARY.md`](CUSTOMER_SUMMARY.md)
- Pre-registered design: [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md) · data licences & leakage controls:
  [`docs/DATASETS.md`](docs/DATASETS.md) · data quality review: [`docs/DATA_QUALITY.md`](docs/DATA_QUALITY.md)
- Machine-readable results: [`experiments/01_finetuning/results/metrics/`](experiments/01_finetuning/results/metrics/)
  · manifest: [`experiments/01_finetuning/results/manifest.json`](experiments/01_finetuning/results/manifest.json)
  · charts: [`experiments/01_finetuning/results/figures/`](experiments/01_finetuning/results/figures/)

## Results at a glance

Primary metric on held-out data (95% bootstrap CIs and all other metrics in `REPORT.md`):

| | Qwen3-8B | Swallow-8B | **Qwen3-8B + LoRA** | GPT-4.1 |
|---|---|---|---|---|
| A entity F1 (synthetic test) | 0.951 | 0.870 | **0.997** | 0.962 |
| B doc-type macro-F1 (synthetic test) | 0.622 | 0.404 | **0.988** | 0.711 |
| C field accuracy (synthetic test) | 0.906 | 0.876 | **0.993** | 0.902 |
| B doc-type macro-F1 (other-generator test) | 0.335 | 0.213 | **0.789** | 0.556 |
| JMED-LLM MRNER-disease strict F1 (real reports) | 0.265 | 0.171 | **0.473** | 0.110 |
| JMED-LLM MRNER-disease lenient F1 (span-tolerant) | 0.544 | 0.534 | 0.645 | **0.707** |
| JMED-LLM RRTNM accuracy (real reports, untrained task) | 0.770 | 0.680 | 0.690 | **0.910** |
| JCommonsenseQA accuracy (general Japanese) | 0.903 | 0.926 | 0.897 | **0.970** |

H1 (LoRA improves the trained tasks), H2 (beyond Japanese-specialised Swallow) and H3 (no general-Japanese
degradation, 2-point margin) are all **supported** under the pre-registered rules. On untrained real-world medical
tasks the adapter did not help and OpenAI models stay clearly ahead; see `REPORT.md` §4 and §8.

## Tasks

Three tasks are run on fictional Japanese clinical notes, each note being used for all three. Every model receives
the same note, prompt, decoding settings and scoring code. Prompts are defined once in
[`src/jmedft/tasks.py`](src/jmedft/tasks.py). All tasks share this system prompt:

> あなたは日本語の医療文書を扱う情報抽出アシスタントです。指示に従い、文書に書かれている内容だけを根拠に回答してください。診断や治療の推奨は行わないでください。

It says: answer only from what the document says, and give no diagnosis or treatment advice. Every instruction
ends with 「出力はJSONオブジェクトのみとし、説明文やコードブロック記号は付けないでください。」 (JSON object only, no
explanation, no code fences).

Why JSON: answers are scored by code against fixed gold labels, with no LLM judge. A structured answer is also how a
customer would consume the output (EHR fields, a database, a downstream system), so format reliability is itself
measured. Strings are compared after NFKC normalisation and whitespace removal, so full-width and half-width
characters match (「２０㎎」 = 「20mg」).

### Task A: medical entity extraction

List the symptoms, diseases and medications that apply to the patient now, copied verbatim from the note. The
instruction excludes:

- negated symptoms (「〜なし」「〜は認めない」, "none / not observed");
- suspected diagnoses (「〜の疑い」「〜の可能性」, "suspected / possible");
- diseases of family members;
- dosages.

Past history counts as a disease.

| | |
|---|---|
| Output | `{"symptoms": [...], "diseases": [...], "medications": [...]}` |
| Primary metric | entity-level micro F1; an item is correct only if it exactly matches a gold item |
| Also reported | precision, recall, per-type F1, exact match per note, JSON validity, schema validity, lenient (span-overlap) F1 |

Example `test_synth-00139` (outpatient record, SOAP format), with the trap annotations added here:

```
79歳、女性。主訴は腹部膨満感。腹部膨満感、鼻汁、多尿、記憶障害を主訴とする。
意識障害は認めない。不安感は認めない。      ← negated: exclude
既往歴に乳癌あり。                          ← past history: counts as a disease
家族歴は兄が帯状疱疹。                      ← brother's disease: exclude
A: 痛風                                     ← current diagnosis: counts
   脂質異常症の可能性                        ← only "possible": exclude
```

| | `diseases` (symptoms and medications identical for all) |
|---|---|
| Gold | `["乳癌", "痛風"]` |
| Qwen3-8B (base) | `["乳癌"]`: misses the diagnosis in the assessment (A) section |
| Qwen3-8B + LoRA | `["乳癌", "痛風"]`: exact |
| GPT-4.1 | `["乳癌", "痛風"]`: exact |

### Task B: document type classification

Choose which of six clinical document types the note is:

| Label | Meaning |
|---|---|
| 外来診療録 | outpatient record (SOAP) |
| 入院時記録 | admission record |
| 退院時サマリー | discharge summary |
| 紹介状 | referral letter |
| 看護記録 | nursing record |
| 検査・画像所見報告書 | test / imaging report |

The note generator was forbidden to write the type's name or a title such as 診療情報提供書, so the type must be
inferred from structure and wording.

| | |
|---|---|
| Output | `{"label": "<one of the six>"}` |
| Primary metric | macro F1 over the six classes |
| Also reported | accuracy, per-class precision/recall, confusion matrix (with an INVALID column for unusable answers) |

Example `test_synth-00037` (gold 紹介状, referral letter):

```
68歳 女性 / 主訴：動悸 / 現病歴および経過 … 受診された。…
本症の診断としては、大腸ポリープの疑い … を考慮している。
ご高診のほどよろしくお願い申し上げます。     ← closing courtesy line typical of a referral letter
```

The base model answered 外来診療録 (wrong); the LoRA model and GPT-4.1 answered 紹介状.

A caution, example `test_synth-00004` (gold 看護記録): the note reads like a physician's summary apart from
「バイタルサインは安定」. The base model and GPT-4.1 both answered 外来診療録; the LoRA model answered 看護記録,
because it learned how this note generator writes "nursing records". Part of the task B gain is therefore the
generator's house style: on notes written by a different model the LoRA's macro F1 falls from 0.99 to 0.79 (see the
report).

### Task C: structured information extraction

Fill an 11-field record, putting every disease into the correct slot:

| Field | Content |
|---|---|
| `age` | integer, or null |
| `sex` | 男性 / 女性, or null |
| `chief_complaint` | exact text, or null |
| `symptoms` | present symptoms |
| `negated_symptoms` | explicitly denied symptoms |
| `past_history` | past illnesses |
| `family_history` | family members' diseases (without the relation) |
| `medications` | drug names (without dose) |
| `allergies` | allergens |
| `diagnoses` | current confirmed diagnoses |
| `suspected_diagnosis` | diagnoses stated only as suspected or possible |

| | |
|---|---|
| Primary metric | field accuracy: each of the 11 fields scored separately; a list field is right only on an exact set match, a scalar only on an exact value match |
| Also reported | exact match (all 11 fields right), list-value P/R/F1, missing fields, unsupported values (predicted text that does not appear in the note, i.e. hallucination), schema compliance |

Example `test_synth-00006` (test / imaging report):

```
67歳 男性 / 主訴：記憶障害 / 既往歴：慢性腎臓病、腰部脊柱管狭窄症 / 家族歴：父 急性心筋梗塞
内服薬：ファモチジン、アレンドロン酸、クエン酸第一鉄 / アレルギー歴：ペニシリン、NSAIDs
検査所見  脳MRI：海馬萎縮を伴う大脳萎縮所見を認める。
印象  記憶障害 / 肩こりは認めない
```

| Field | Gold | Qwen3-8B (base) | GPT-4.1 | Qwen3-8B + LoRA |
|---|---|---|---|---|
| `negated_symptoms` | `["肩こり"]` | `["肩こりは認めない"]` (whole phrase) | `["肩こりは認めない"]` | `["肩こり"]` |
| `diagnoses` | `[]` | `["記憶障害"]` (symptom promoted to a diagnosis) | `["記憶障害"]` | `[]` |

The other nine fields were correct for all three models.

### How the notes and test sets are built

- **How notes are made:** each synthetic note starts from a sampled specification (age, sex, symptoms,
  negated symptoms, history, diagnoses, …). The gold labels are that specification, not a reading of the
  generated text. A deterministic validator then rejects any note whose text does not support every label
  (see [`docs/DATA_QUALITY.md`](docs/DATA_QUALITY.md)).
- **Held-out vocabulary:** a quarter of the term inventory appears only in test notes, so the tests contain
  medical terms never seen in training.

| Test set | Size | Purpose |
|---|---|---|
| `test_synth` | 223 notes × 3 tasks | held-out notes from the training-data generator (Qwen3.8-27B) |
| `test_xgen` | 215 notes × 3 tasks | notes from a different model (Nemotron), to check that gains are not the generator's style |
| `test_ext` | 4 × 100 | human-annotated real case and radiology reports (JMED-LLM, CC BY 4.0): disease NER, medication NER, adverse-drug-event certainty, TNM staging |
| `test_general` | 1,118 + 2,508 + 100 | general Japanese (JCommonsenseQA, JNLI) and medical knowledge (JMMLU-Med), to check for regressions |

External and general items keep their benchmark's own question wording, with the JSON instruction appended. For
example, CRADE asks 「以下の症例報告において、「膿萢，結痂，ビラン状局面」がADE（有害事象）である可能性は？」 (in
this case report, how likely is this finding to be an adverse drug event?) with options A 無関係 / B 可能性は低い /
C おそらくADE / D 確実にADE (unrelated / unlikely / probable / certain), and the answer format
`{"answer": "<記号>"}`. JMED-LLM labels short spans such as 「胃癌」 while models often return whole clauses, so strict
F1 on its NER tasks is low for every model, and a lenient F1 is reported alongside.

## Layout

```
src/jmedft/
  tasks.py            prompts, label sets, JSON schemas (single source of truth for every model)
  datasets/           synthetic spec sampling + generation + validation, third-party converters, split builder
  training/           LLaMA-Factory LoRA driver, adapter export, Fireworks/vLLM compatibility check
  inference/          async OpenAI-compatible runner (local vLLM or OpenAI), resumable
  evaluation/         parsing, metrics, bootstrap/McNemar/Holm statistics, hypothesis tests
  reporting/          charts, tables, error examples, experiment manifest
  cli.py              `jmedft` command line
experiments/01_finetuning/
  configs/            data.yaml, term_inventory.yaml, conditions.yaml, train_lora.yaml, pricing.yaml
  scripts/            serve_vllm.sh, setup_llamafactory.sh, fetch_external_data.sh, run_all.sh
  results/            adapters/, predictions/, metrics/, figures/, training/, manifest.json
data/
  raw/                third-party data (re-downloadable, git-ignored)
  synthetic/          generated notes with specs and validation logs
  processed/          sft/ and eval/ splits + SHA256SUMS, split_stats.json, leakage_report.json
tests/                unit tests (pytest)
```

## Environment (verified on this machine)

| Item | Value |
|---|---|
| Hardware | NVIDIA DGX Spark, GB10 (sm_121), 128 GB unified memory, aarch64 |
| OS / driver / CUDA | Ubuntu 24.04.4, driver 580.173.02, CUDA 13.0 |
| Main venv (`.venv`, uv) | Python 3.13.15, torch 2.14.1+cu130, transformers 5.20.1, peft 0.21.2 |
| Training venv (`.venv-llamafactory`) | Python 3.12.3, llamafactory 0.9.5, torch 2.11.0+cu130, transformers 5.6.0, peft 0.18.1, trl 0.24.0 |
| Inference | Docker `vllm/vllm-openai:v0.31.0` (arm64, torch 2.13+cu130) |

## Reproduction

Prerequisites: `uv`, Docker with the NVIDIA runtime, ~60 GB free disk for model weights, and a `.env` containing
`OPENAI_API_KEY` (used only by E4/E4b; see `.env.example`). Model revisions are pinned in
`experiments/01_finetuning/configs/conditions.yaml`.

`data/raw/`, `data/synthetic/` and the data files in `data/processed/` are not tracked in git (they contain
third-party licensed text and test gold labels): raw data is re-downloaded by step 1 and synthetic data is regenerated
by step 2. Only `data/processed/SHA256SUMS`, `split_stats.json` and `leakage_report.json` are tracked. Regenerated
notes will not be byte-identical, because vLLM batching is not bit-exact even with fixed seeds; to check whether you
have the exact files used in this run, compare against `data/processed/SHA256SUMS` and the manifest.

```bash
# 0. environments
uv sync --extra dev                                         # main venv
bash experiments/01_finetuning/scripts/setup_llamafactory.sh  # isolated training venv
.venv/bin/pytest                                            # unit tests

# 1. models (pinned revisions) and third-party data
uvx --from "huggingface_hub[hf_xet]" hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218
uvx --from "huggingface_hub[hf_xet]" hf download tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2 --revision 9218f4843b6f93369a0b0999d8f58d61487ea71c
uvx --from "huggingface_hub[hf_xet]" hf download tokyotech-llm/Medical-Qwen3-Swallow-8B --revision f9a3d4dc38270d39196a8c5b5e14f56088fd0876
uvx --from "huggingface_hub[hf_xet]" hf download Qwen/Qwen3.8-27B-FP8 --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a
uvx --from "huggingface_hub[hf_xet]" hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 --revision 6dbbd757ea75a8ece6e0702872e3ae53f9987728
bash experiments/01_finetuning/scripts/fetch_external_data.sh

# 2. synthetic data: generator servers one at a time, then re-validate
VLLM_GPU_UTIL=0.55 VLLM_MAX_MODEL_LEN=16384 bash experiments/01_finetuning/scripts/serve_vllm.sh \
  jmedft-teacher 8200 Qwen/Qwen3.8-27B-FP8 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a qwen3.8-27b --language-model-only
.venv/bin/jmedft data synth --splits test_synth val train
docker stop jmedft-teacher
bash experiments/01_finetuning/scripts/serve_nemotron_generator.sh
.venv/bin/jmedft data synth --splits test_xgen
docker stop jmedft-xgen
# In this run the validator was strengthened after generation (see docs/DATA_QUALITY.md), so the final
# validator was re-applied to the generated notes. With the current code this is a no-op.
.venv/bin/jmedft data revalidate --splits train val test_synth test_xgen

# 3. build + validate splits
.venv/bin/jmedft data build
.venv/bin/jmedft data validate

# 4. fine-tune (smoke test first), export adapter, check Fireworks/vLLM compatibility
.venv/bin/jmedft train --smoke
.venv/bin/jmedft train
.venv/bin/jmedft adapter-check

# 5. inference: local servers one at a time (OpenAI conditions need .env)
.venv/bin/jmedft serve --server qwen3
.venv/bin/jmedft infer --condition E1_qwen3_8b
.venv/bin/jmedft infer --condition E3_qwen3_8b_lora
.venv/bin/jmedft latency --condition E1_qwen3_8b
.venv/bin/jmedft latency --condition E3_qwen3_8b_lora
.venv/bin/jmedft serve --server qwen3 --stop
.venv/bin/jmedft serve --server swallow
.venv/bin/jmedft infer --condition E2_swallow_8b
.venv/bin/jmedft latency --condition E2_swallow_8b
# reasoning-mode sensitivity arm: seeded 60-item sample per task (see docs/PREREGISTRATION.md, amendment 7)
.venv/bin/jmedft infer --condition E2t_swallow_8b_think --splits test_synth test_xgen test_ext --sample 60
.venv/bin/jmedft serve --server swallow --stop
.venv/bin/jmedft infer --condition E4_gpt41 --concurrency 16
.venv/bin/jmedft infer --condition E4b_gpt54mini --concurrency 8 --splits test_synth test_xgen test_ext
.venv/bin/jmedft latency --condition E4_gpt41 --concurrency 16
.venv/bin/jmedft latency --condition E4b_gpt54mini --concurrency 16

# 6. score, test hypotheses, build charts/tables/manifest
.venv/bin/jmedft evaluate
.venv/bin/jmedft report
```

`experiments/01_finetuning/scripts/run_all.sh` chains steps 2–6. Inference resumes safely: an existing prediction is
reused only if its request fingerprint (model, decoding, prompt) matches, so an interrupted run can be restarted
with the same command.

## License

The code in this repository is licensed under the [Apache License 2.0](LICENSE). See [`NOTICE`](NOTICE) for
attributions.

The Apache License covers the code only:

- **Third-party datasets** keep their own licences and are not redistributed here; `fetch_external_data.sh`
  downloads them. Committed results hold model outputs and scores, plus short CC BY 4.0 JMED-LLM excerpts in
  `results/metrics/error_examples.json` for error analysis (attributed in `NOTICE`).
  - JMED-LLM disease/medicine/CRADE/RRTNM, MedTxt and RR-TNM are CC BY 4.0, and any use requires attribution.
  - JGLUE and JMMLU-Med are CC BY-SA 4.0 and are used for evaluation only.
  - Details and excluded datasets are in [`docs/DATASETS.md`](docs/DATASETS.md).
- **The LoRA adapter** is a derivative of Qwen3-8B (Apache-2.0) and is provided under Apache-2.0.
- **Synthetic notes** were generated with Qwen3.8-27B (Apache-2.0) and Nemotron-3.5-Lightning (OpenMDW-1.1); neither
  licence restricts use of outputs. No OpenAI output was used for training.

## Safety and scope

All synthetic notes describe fictional patients; a validator rejects dates, phone numbers and postcodes. No real
patient data is used. The system extracts and classifies information stated in text; it does not diagnose or
recommend treatment. API keys are read from `.env` (git-ignored) and redacted from any logged error.
