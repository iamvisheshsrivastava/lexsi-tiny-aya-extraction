# Fine-tuning Tiny Aya to read invoices and tickets: what worked, and what only looked like it worked

*Lexsi AI Safety and Alignment Hackathon, Track 1 (fine-tuning and data curation). Stack: CuratorKIT, AlignTune, AuditKIT. Model: Tiny Aya Global (3.35B).*

Code and results: [GitHub](https://github.com/iamvisheshsrivastava/lexsi-tiny-aya-extraction) · Adapters and logs: [model repo](https://huggingface.co/iamvisheshsrivastava/lexsi-tiny-aya-extraction-lora) · Data: [v2](https://huggingface.co/datasets/iamvisheshsrivastava/lexsi-extraction-data-v2), [v1](https://huggingface.co/datasets/iamvisheshsrivastava/lexsi-extraction-data)

## The task

A short support ticket or invoice, in English, German, French or Hinglish, goes in. One JSON object with ten fixed fields comes out: `doc_type, language, party_name, reference_id, product, issue_category, priority, amount, currency, date_iso`. Four fields must always take one of a few allowed values; the rest may be null. A program should be able to use the output without a human looking at it, so we measure three things with AuditKIT: is the output parseable JSON, does it pass the schema, and how many field values are right (field F1).

## The pipeline

1. **CuratorKIT** takes teacher-written examples, removes near-duplicates (MinHash, 0.85), cleans them and splits them.
2. **AlignTune** trains a 4-bit LoRA adapter (rank 16) on a Kaggle T4.
3. **AuditKIT** scores every run, ours and the hosted models', with the same three scorers.

There was no public dataset for this, so the data is synthetic: written by a teacher model, with automatic checks that the amount, date and names really appear in the text.

## Round 1: fine-tuning looked great, until we compared it fairly

With 188 training examples and 24 test documents, the untuned model prompted with only "Extract the structured fields." returned no parseable JSON. After fine-tuning it returned 92% parseable JSON and field F1 0.78. A huge jump.

But that baseline was unfair: the one-line prompt never names the JSON keys. When we gave the untuned model the full schema in its prompt, it scored 0.80 with 100% valid output, no training at all. In round 1, fine-tuning bought a shorter prompt, not accuracy. We also found that our first full-prompt wording contradicted the schema (it said "null for anything not stated" while four fields cannot be null), fixed it after seeing hosted models fail on it, and say so in the repository.

## Round 2: harder data, a bigger test set

The round-1 test set was too small and too easy: many documents contained the field names as labels ("Party Name: ..."). For round 2 a second teacher model wrote new documents under a difficulty list (the amount is a total and not a unit price, no reference id, local date formats, urgency only implied, a second person mentioned) with no field-name labels. After CuratorKIT: 510 training and 57 validation documents. The test set has 120 documents, 30 per language, checked for near-duplicates against everything else.

| Model and prompt | Parseable JSON | Schema valid | Field F1 |
|---|---|---|---|
| Tiny Aya, untuned, one-line prompt | 0% | 0% | 0.00 |
| Tiny Aya, untuned, full schema prompt | 99% | 94% | 0.70 |
| **Tiny Aya, fine-tuned, one-line prompt** | **100%** | **93%** | **0.83** |
| Claude Haiku 5.5, full prompt (hosted) | 100% | 100% | 0.85 |
| Gemini 2.5 Flash-Lite, full prompt (hosted) | 100% | 100% | 0.84 |
| Command A, full prompt (hosted) | 100% | 100% | 0.83 |
| GPT-5.4 nano, full prompt (hosted) | 100% | 98% | 0.82 |

On this harder test the fine-tuned model clearly beats the same model with the full prompt: +0.08 per-document F1, 95% bootstrap interval +0.06 to +0.11. It lands in the same F1 range as four hosted models, but its strict-schema validity is lower (93% against 98 to 100%): seven of its eight failures are an `issue_category` outside the allowed list.

## Cost

Hosted models cost $0.07 to $1.75 per 1,000 documents (billed). The fine-tuned model took 1.9 seconds per document on one T4 with plain Hugging Face generation, about $0.19 per 1,000 documents at an assumed $0.35 per GPU hour. We did not rent a GPU, so the price is an assumption. That is the same range as the cheap hosted models. The case for self-hosting here is privacy and control, not price.

## What to keep in mind

- All text and labels are model-written. We read a sample of 32 test documents against their labels: 8 had at least one clearly wrong or unsupported field (2.5% of fields), and the reader was Claude, not a human. Scores are measured against these labels, so they are a ceiling-limited estimate.
- One training run per round; the interval above covers test-set sampling only.
- The 24-document and 120-document results disagree on whether fine-tuning beats prompting. Data, test set and training length all changed between rounds, so we cannot say which change mattered.
- No throughput-optimised server was tested, so the cost numbers are conservative for self-hosting.

Everything is reproducible from the repository: notebooks for the Kaggle runs, scripts for data, evaluation, audit and cost, and 14 tests.

*Licence: code MIT; adapters and data CC-BY-NC-4.0 (Tiny Aya and Command A are non-commercial).*
