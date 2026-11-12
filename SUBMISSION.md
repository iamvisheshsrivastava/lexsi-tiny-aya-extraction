# Lexsi hackathon submission: form answers

Track 1 (Fine-tuning and Data Curation), problem: structured output and extraction. Model: Tiny Aya Global (`CohereLabs/tiny-aya-global`).
Repos used together: CuratorKIT, AlignTune, AuditKIT.

## Abstract (under 500 words)

Many applications need a small model to read a short business document and return a fixed JSON object. We study this with Tiny Aya Global
(3.35B) on support tickets and invoices in English, German, French and Hinglish, and ten output fields (document type, language, party,
reference id, product, issue category, priority, amount, currency, ISO date). We chain three Lexsi repos: CuratorKIT curates teacher-written
data (MinHash deduplication, cleaning, splits), AlignTune fine-tunes a 4-bit LoRA adapter, and AuditKIT scores every run with the same
three scorers (parseable JSON, schema validity, per-field F1).

We ran two rounds and report both. Round 1 used 188 training examples and a 24-document test set. The untuned model given only a one-line
prompt returned no parseable JSON; after fine-tuning it returned 92% parseable JSON and field F1 0.78. But the untuned model given the full
schema in its prompt already reached F1 0.80, so in round 1 fine-tuning bought a shorter prompt, not accuracy. We caught that our first
baseline was unfair and added the full-prompt comparison, and we disclose that one prompt wording was fixed after reading test failures.

Round 2 addresses the weak spots. We wrote a harder dataset with a second teacher model (documents without field-name labels, tricky
amounts, local date formats, implied urgency, missing fields), with automatic grounding checks, giving 510 training and 57 validation
documents after CuratorKIT. The test set has 120 documents, 30 per language, checked for near-duplicates against the training data. On
this test set, the untuned model with the full prompt scores field F1 0.70 (94% schema-valid). The fine-tuned model, given only the
one-line prompt, scores **0.83** and returns parseable JSON for 100% of documents (93% schema-valid; exact match 15% against 3%). The
per-document F1 gain is +0.08 with a 95% bootstrap interval of +0.06 to +0.11. Four hosted models with the full prompt (Claude Haiku 5.5,
Gemini 2.5 Flash-Lite, Command A, GPT-5.4 nano) score 0.82 to 0.85, so the 3.35B fine-tuned model matches them on field F1 but is weaker on
strict schema validity; Claude Haiku is slightly ahead in a paired comparison (+0.02).

ROI: hosted cost per 1,000 documents (billed) is $0.07 to $1.75. Self-hosting the fine-tuned model on one T4 took 1.9 s per document with
plain Hugging Face generation, about $0.19 per 1,000 documents at an assumed $0.35 per GPU hour (GPU prices are assumptions; we did not
rent a GPU). That is the same range as the cheap hosted models, so the case for self-hosting here is privacy and control, not price.

Limits: synthetic, model-written, unchecked labels from two teachers; one training run per round; strict-schema failures (mostly an invalid
issue category) remain; no throughput-optimised server was tested. Code, data, adapters, logs and results are public.

## Links

- GitHub repository (code, notebooks, README with reproduction steps): <GITHUB_URL>
- Hugging Face model collection (adapters, logs, results): <MODEL_COLLECTION_URL>
- Hugging Face dataset collection (v1 and v2 data): <DATASET_COLLECTION_URL>
- Write-up: the repository README (`README.md`); the model repo card carries the same text.

## Problem statement

Fine-tune a small open-weights model to turn short multilingual support tickets and invoices into one schema-valid JSON object, and measure
whether it is meaningfully better, and cheaper to run, than prompting the same model or a hosted model.

## Metrics and why

Parseable JSON rate and schema-valid rate (can a program use the output at all), field F1 (are the values right; AuditKIT `FieldF1` with number
and case normalisation), exact whole-document match (strictest), and cost per 1,000 documents (ROI). We report all of them for every run.
