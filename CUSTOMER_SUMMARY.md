# Japanese medical AI: can a small fine-tuned open model match OpenAI? — Technical summary

Prepared for Nihin Media · 2026-10-19 · full technical report: `REPORT.md`

## The question

You use OpenAI models today and are happy with their Japanese. We tested a narrower claim: *a small open-weight
model, fine-tuned on Japanese medical tasks, can match or beat OpenAI on those specific tasks while running on
hardware you control.*

## What we did

- Took **Qwen3-8B** (open-weight, Apache-2.0, commercial use allowed) and trained a small add-on (a 61 MB "LoRA
  adapter") on Japanese clinical-style documents, on a single **NVIDIA DGX Spark** desktop system in 2.6 hours.
- Built three tasks representative of clinical document processing: **extracting** symptoms, diseases and
  medications; **classifying** the document type (referral letter, discharge summary, nursing record, …); and
  **filling an 11-field structured record** (negated symptoms, suspected vs confirmed diagnoses, family history, …).
- Compared it, on identical inputs and with identical automatic scoring, against the untuned model, a
  Japanese-specialised model of the same size (Swallow), **GPT-4.1** and **GPT-5.4-mini**.
- Training data was synthetic (fictional patients, no real patient information) plus a small amount of openly
  licensed real text; all data sources were licence-checked for commercial use. We fixed the success criteria in
  writing before seeing any results.

## What we found

**On the trained tasks, the fine-tuned 8B model was the most accurate model tested** — including vs GPT-4.1:

| Task (held-out test) | Untuned 8B | **Fine-tuned 8B** | GPT-4.1 |
|---|---|---|---|
| Entity extraction (F1) | 0.951 | **0.997** | 0.962 |
| Document type (macro-F1) | 0.622 | **0.988** | 0.711 |
| Structured record (field accuracy) | 0.906 | **0.993** | 0.902 |
| Records with all 11 fields correct | 35% | **94%** | 36% |

The improvement held on documents written in a different style (by a different AI model), and on real,
human-annotated Japanese case reports for disease-finding extraction (F1 0.27 → 0.47 under the benchmark's strict
scoring; under a more lenient scoring of span boundaries GPT-4.1 is still ahead, 0.71 vs 0.65). It held
against the Japanese-specialised model too: Japanese language tuning alone did not deliver it — task tuning did.
General Japanese ability was unchanged.

## What it does not show

- **It is a specialist, not a generalist.** On real medical tasks it was not, or barely, trained for (adverse-drug-event
  certainty, cancer staging, medication dosages) the fine-tuned model did no better than the untuned one, and
  OpenAI models were clearly better (e.g. staging 0.91–0.97 vs 0.69). On general Japanese reasoning benchmarks GPT-4.1
  remains well ahead.
- **Part of the gain is learning our labelling conventions** (what counts as a "symptom", how to write a span). That
  is valuable — it is exactly how a model learns *your* house style — but it means results on your own documents
  must be measured, not assumed.
- **Long real documents exposed a fixable reliability issue** (the model occasionally repeats itself until it hits
  its length limit, on 9% of long real reports). A standard decoding setting reduced this to under 2%.

## Economics (measured on this workload)

| | Latency per request (median) | Cost |
|---|---|---|
| GPT-4.1 via API | 1.1 s | **$1.62 per 1,000 documents** (list price) |
| Fine-tuned 8B on one DGX Spark | 4.5 s | ~14,000 documents/hour at full load; cheaper than GPT-4.1 whenever the system's all-in cost is below **~$23/hour** |

The open model is slower per request on a single desktop unit (it is memory-bandwidth bound), so the business
case is **throughput, cost control and data residency** (patient data never leaves your infrastructure), not
latency. The adapter is also compatible with Fireworks AI's hosted LoRA serving for a managed option (not yet
deployed or measured).

## Recommendation

The hypothesis holds **for well-defined, repeatable extraction and classification tasks**: a fine-tuned 8B model
can beat GPT-4.1 on them, and is cheaper per document whenever the self-hosted system costs less than about
$23/hour all-in at full load (an estimate from measured throughput, not a measured bill). Before production we recommend a second, short experiment on
**your own data** (≈ 200 de-identified real documents or physician questions as a test set), on tasks that map to
your products (e.g. MedGen question understanding and cited answers from given sources), with optimised serving
(FP8, hosted LoRA) to measure production latency and cost. Keep OpenAI models for open-ended medical reasoning.
