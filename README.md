# Structured extraction with Tiny Aya: CuratorKIT, AlignTune, AuditKIT

Lexsi AI Safety and Alignment Hackathon, Track 1 (fine-tuning and data curation), problem "structured output and extraction".
Code: <https://github.com/iamvisheshsrivastava/lexsi-tiny-aya-extraction>. Models and logs: <https://huggingface.co/iamvisheshsrivastava/lexsi-tiny-aya-extraction-lora>.
Data: <https://huggingface.co/datasets/iamvisheshsrivastava/lexsi-extraction-data-v2> (main) and `lexsi-extraction-data` (round 1).

We fine-tuned a small Cohere open-weights model, **Tiny Aya Global (3.35B)**, to read a short support ticket or invoice written in
English, German, French or Hinglish and return one JSON object with ten fixed fields. The data was curated with **CuratorKIT**,
the model was trained with **AlignTune** (LoRA), and every run was scored with **AuditKIT**.

## The short version

We ran the project twice. Round 1 used a small 24-document test set; round 2 (v2) uses more training data and a harder 120-document test
set. The round-2 result is the main one.

- **On the harder v2 test set (120 documents), fine-tuning clearly beats prompting.** The untuned Tiny Aya with the full schema prompt gets
  AuditKIT field F1 0.70. The fine-tuned model, given only the ten-word prompt "Extract the structured fields.", gets **0.83**, and
  returns parseable JSON for 100% of documents (93% pass the strict schema, against 94% for the untuned model). Exact match is 15%
  against 3%. The difference in per-document F1 is +0.08 with a 95% bootstrap interval of +0.06 to +0.11 (76 documents better, 25 worse).
- **It gets close to hosted models.** With the full schema prompt, four hosted models score 0.82 to 0.85 and 98 to 100% schema-valid
  output. The fine-tuned 3.35B model is within that range on F1 (0.83) but not on strict schema validity (93%). Claude Haiku 5.5 is
  still slightly ahead on a paired comparison (+0.02, interval +0.002 to +0.04).
- **In round 1 the picture was different.** With 188 training examples and 24 test documents, the fine-tuned model (F1 0.78) was no better than the
  untuned model given the full prompt (0.80). We report that too: with little and easy data, prompt design mattered more than fine-tuning.
  With 510 harder training examples it does not. The two rounds differ in data, test set and training length, so we cannot say which
  change mattered most.
- **What fine-tuning buys:** the schema lives in the weights, so the prompt is about 3 times shorter and accuracy is higher on hard
  documents. It does not make the model cheaper than the cheapest hosted ones (see ROI).

## Task and data

Output schema (`src/schema.py`): `doc_type, language, party_name, reference_id, product, issue_category, priority, amount,
currency, date_iso`. `doc_type`, `language`, `issue_category` and `priority` must always take one of the allowed values; the
rest may be null.

There was no suitable public dataset, so we generated one. A teacher model (Cohere `command-a-03-2025`) wrote documents together
with their gold JSON. Checks rejected generations whose amount or date did not appear in the text. CuratorKIT then removed
near-duplicates (MinHash, threshold 0.85), cleaned the text and split the data 80/10/10:
271 generated, 235 kept, **188 train / 23 validation / 24 test**. The test set has 12 English, 7 French, 3 Hinglish and 2 German
documents (13 invoices, 11 tickets). It is small and synthetic; read every number with that in mind.

### Data v2: a larger training set and a harder, balanced test set

The 24-document test set above is too small, and its documents often contain the field names as labels ("Party Name:", "Reference ID:"),
which makes extraction easy. So we built v2 (`src/generate_data_v2.py`, `src/build_v2.py`, files in `data/final_v2/`):

- A second teacher (`deepseek/deepseek-v4-flash` through OpenRouter) wrote 330 new training documents under a difficulty list:
  amount is a total and not a unit price, no reference id, local date formats, urgency only implied, a second person named, a
  product described in a sentence, and others. Documents may not contain the field names as labels. Checks reject an item unless the
  JSON matches the schema, the language and type match, and party name, reference id, amount and the date year appear in the text.
- Training set: 247 v1 documents (v1 test documents removed) plus the 330 new ones, curated with CuratorKIT (MinHash 0.85, cleaning):
  **510 train / 57 validation**.
- Test set: **120 documents, 30 per language** (English, German, French, Hinglish; 61 tickets, 59 invoices), drawn separately and checked
  for near-duplicates (5-gram Jaccard 0.7) against train, validation, the v1 test set and itself. The v1 test set stays as `test_v1`.
- Labels are model-written and not checked by a person; `priority` and `issue_category` are subjective. See the dataset card.

**Hosted models on the v2 test set (full schema prompt, AuditKIT):**

| Model | Parseable JSON | Schema valid | Field F1 | Cost per 1,000 docs |
|---|---|---|---|---|
| Claude Haiku 5.5 | 100% | 100% | 0.85 | $0.235 |
| Gemini 2.5 Flash-Lite | 100% | 100% | 0.84 | $0.072 |
| Command A | 100% | 100% | 0.83 | $1.746 |
| GPT-5.4 nano | 100% | 98% | 0.82 | $0.167 |

All four fall from 0.89 to 0.93 on the 24-document set to 0.82 to 0.85 here, so the v2 test set is harder; they are also closer to each
other. With the minimal prompt every hosted model gets field F1 at or below 0.05 (0% schema-valid), as in round 1. Field F1 by
language (own scorer) runs from 0.75 (English, GPT-5.4 nano) to 0.87 (German, Claude Haiku 5.5). Files: `results/v2_closed/`,
`results/audit_summary_v2.json`, `results/v2_roi.md`.

**Tiny Aya on the v2 test set (120 documents; AuditKIT scores).** One Kaggle T4 run (`notebooks/lexsi_pipeline_v2.ipynb`): LoRA fine-tuning on
the 510 training documents, 2 epochs (128 steps, 106 minutes, training loss 2.86 to 0.39), then evaluation with generation stopped at the
closing brace.

| Model and prompt | Parseable JSON | Schema valid | Field F1 |
|---|---|---|---|
| Tiny Aya Global, untuned, minimal prompt | 0% | 0% | 0.00 |
| Tiny Aya Global, untuned, full schema prompt | 99% | 94% | 0.70 |
| **Tiny Aya Global, fine-tuned, minimal prompt** | **100%** | **93%** | **0.83** |
| Claude Haiku 5.5, full prompt (hosted) | 100% | 100% | 0.85 |
| Gemini 2.5 Flash-Lite, full prompt (hosted) | 100% | 100% | 0.84 |
| Command A, full prompt (hosted) | 100% | 100% | 0.83 |
| GPT-5.4 nano, full prompt (hosted) | 100% | 98% | 0.82 |

Our own scorer gives the same ranking with smaller gaps: untuned full prompt 0.74, fine-tuned 0.82 (exact match 3% and 15%). Field F1 by
language (own scorer, 30 documents each): fine-tuned German 0.83, English 0.78, French 0.85, Hinglish 0.82; untuned 0.75, 0.70, 0.75,
0.77. By type: fine-tuned invoices 0.80 and tickets 0.84; untuned 0.77 and 0.72. AuditKIT's comparison of the untuned full-prompt run with the
fine-tuned run reports 1 document going from wrong to right under its strict whole-document test (it is dominated by the per-field
scores above). On the old 24-document test set, the same fine-tuned model reaches F1 0.89 (own scorer, exact match 33%) against 0.85
(exact match 8%) for the untuned model with the full prompt (`results/v2_tinyaya/v1test_*`).

**How good are the v2 test labels?** The labels were written by a teacher model, so we checked a sample: 32 test documents (8 per language, chosen
with a fixed seed) were read against their gold labels (`results/label_audit.json`). The reader was Claude (Sonnet 5.5), not a human. In 8
of the 32 documents (25%) at least one field was clearly wrong or unsupported by the text, which is 8 of 320 fields (2.5%): three have
`doc_type` set to invoice for what is really a support ticket, two have a date that is not in the text, one names the issuing company instead of the
customer, one has an issue category that the text does not support, and one has a null product that the text does state. Another 4 labels are debatable (mostly
`priority`). So the labels are mostly sound but not clean, and a 25% rate of documents with a defect puts a ceiling on every score here; all models, hosted or not,
are scored against the same labels.

**What the fine-tuned model gets wrong.** Eight of 120 outputs fail the strict schema: seven use an `issue_category` that is not one of the
allowed values, and one adds a key. Nothing hit the token limit, which was a problem in round 1. The untuned model fails seven documents
in other ways (a malformed currency, a language outside the allowed set, missing keys). Files: `results/v2_tinyaya/`, `results/v2_bootstrap.json`,
`results/audit_summary_v2.json`.

## Pipeline

| Stage | Tool | What it did |
|---|---|---|
| Curate | CuratorKIT (`src/curate.py`) | dedup, cleaning, splits, Alpaca export |
| Train | AlignTune (`src/train_sft.py`) | LoRA SFT, rank 16, alpha 32, 4-bit base, learning rate 2e-4. Round 1: 188 examples, 3 epochs (72 steps), about 47 minutes on a Kaggle T4, loss 2.68 to 0.33. Round 2: 510 examples, 2 epochs (128 steps), 106 minutes, loss 2.86 to 0.39. |
| Evaluate | AuditKIT (`src/audit.py`) | the same three scorers on every run: `IsJson`, `JsonSchemaValid`, `FieldF1` |

## Round 1 results (24-document test set, AuditKIT scores)

| Model and prompt | Parseable JSON | Schema valid | Field F1 |
|---|---|---|---|
| Tiny Aya Global, untuned, minimal prompt | 0% | 0% | 0.00 |
| Tiny Aya Global, untuned, **full schema prompt** | 100% | 100% | 0.80 |
| **Tiny Aya Global, fine-tuned, minimal prompt** | 92% | 58% | 0.78 |
| Gemini 2.5 Flash-Lite, full prompt (hosted) | 100% | 100% | 0.89 |
| Command A, full prompt (hosted) | 100% | 100% | 0.90 |
| Claude Haiku 5.5, full prompt (hosted) | 100% | 100% | 0.93 |
| GPT-5.4 nano, full prompt (hosted) | 100% | 100% | 0.93 |
| Hosted models, minimal prompt | 0 to 100% | 0% | 0.00 to 0.12 |

Our own scorer (`src/eval_extraction.py`) is a little more forgiving: it pulls the first JSON object out of any text. With it, the
untuned model with the full prompt gets field F1 0.85 and exact match 8%; the fine-tuned model gets 100% parseable, 67% schema
valid, field F1 0.85 and exact match 21%. Both scorers are in the repo; the table uses AuditKIT because it is applied the same
way to every run. Field F1 by language (own scorer): fine-tuned English 0.85, German 0.84, French 0.85, Hinglish 0.83; untuned
with the full prompt 0.85, 0.74, 0.85 and 0.90 (2 German and 3 Hinglish documents, so these are noisy). AuditKIT's comparison of
the minimal-prompt baseline with the fine-tuned model reports 22 of 24 documents going from wrong to right and none the other way;
that comparison is against the weak baseline, which is why the full-prompt row matters.

**What the fine-tuned model gets wrong.** Eight of its 24 outputs fail the strict schema: five miss a key (mostly `date_iso`), four use
an enum value in the document's language (for example `facturation` or `Haute`), and two add keys that are not in the schema. One output
runs on to 751 characters. These are the typical errors of a model trained on 188 examples; more and more varied data should help,
but we did not test that.

**Things we changed or found along the way, in the open.**
1. The first wording of the full instruction said "use null for anything not stated", while the schema forbids null for
   `priority` and `issue_category`. Hosted models followed the instruction and failed the schema (25 to 42% valid). We fixed the
   wording after reading those failures on the test documents, so this is a small amount of prompt tuning on the test set. Both
   result sets are kept (`results/closed_first_wording/`). The untuned Tiny Aya was run only with the fixed wording.
2. Our first baseline used the minimal prompt, which never names the JSON keys. We caught this while preparing the cost
   comparison and added the full-prompt runs; the table above shows both.

## ROI: hosted tokens against self-hosting

**Round 2 (v2 test set, 120 documents).** Hosted costs are what OpenRouter billed for the 120 documents, scaled to 1,000. Self-hosted costs
use the time measured on one Kaggle T4 (4-bit weights, plain Hugging Face generation, batches of 8, generation stopped at the closing
brace) times a GPU price that we **assume**; we did not rent a GPU.

| Option | Field F1 | Cost per 1,000 documents |
|---|---|---|
| Gemini 2.5 Flash-Lite, hosted | 0.84 | $0.07 |
| GPT-5.4 nano, hosted | 0.82 | $0.17 |
| Claude Haiku 5.5, hosted | 0.85 | $0.24 |
| Command A, hosted | 0.83 | $1.75 |
| Fine-tuned Tiny Aya, minimal prompt, self-hosted, assumed $0.35/h T4 (1.9 s per document) | 0.83 | $0.19 |
| Fine-tuned Tiny Aya, assumed $0.80/h T4 | 0.83 | $0.43 |
| Untuned Tiny Aya, full prompt, self-hosted, assumed $0.35/h T4 (1.7 s per document) | 0.70 | $0.17 |
| Either Tiny Aya on Kaggle's free tier | | $0.00 (limited hours) |

Reading: at an assumed $0.35 per GPU hour the fine-tuned model costs about as much as GPT-5.4 nano and Claude Haiku, less than Command A and
more than Gemini Flash-Lite, with similar F1 to all of them but weaker strict-schema validity (93% against 98 to 100%). So on this task the
case for self-hosting is privacy, offline use and control, and a cost in the same range, not a cost advantage. A throughput-oriented server
(vLLM, larger batches) could lower the self-hosted cost; we did not test one. Full numbers: `results/v2_roi.md`.

**Round 1 (24 documents).**

Hosted costs are what OpenRouter billed for the 24 test documents, scaled to 1,000. Self-hosted costs use the time we measured on
one Kaggle T4 (4-bit weights, plain Hugging Face generation, batches of 8) times a GPU price that we **assume**, since we did not
rent one.

| Option | Cost per 1,000 documents |
|---|---|
| Gemini 2.5 Flash-Lite, hosted | $0.07 |
| GPT-5.4 nano, hosted | $0.16 |
| Claude Haiku 5.5, hosted | $0.19 |
| Command A, hosted | $1.66 |
| Untuned Tiny Aya, full prompt, self-hosted, assumed $0.35/h T4 (1.3 s per document) | $0.13 |
| Untuned Tiny Aya, full prompt, self-hosted, assumed $0.80/h T4 | $0.30 |
| Fine-tuned Tiny Aya, short prompt, self-hosted, assumed $0.35/h T4 (4.1 s per document) | $0.40 |
| Either Tiny Aya on Kaggle's free tier | $0.00 (limited hours) |

The fine-tuned model looks slower only because a few of its outputs ran to the 256-token limit and held up their batch, so its
4.1 s is not a fair estimate of what a tuned server could do. The honest reading: a self-hosted 3B model is in the same price range
as the cheap hosted models (cheaper than Command A, about the same as GPT-5.4 nano and Claude Haiku, dearer than Gemini Flash-Lite),
and it is less accurate than all of them. The case for self-hosting is privacy and offline use, not price or accuracy. A
throughput-oriented server could change the cost; we did not test one. Full numbers: `results/roi.md`.

## Limitations

The round-1 numbers rest on 24 test documents; the v2 test set has 120 but its labels are model-written and not checked by a person. All data is synthetic, written by two teacher models (Command A, DeepSeek V4 Flash); a single training run and seed per round (the bootstrap interval covers test-set sampling only, not training variance; a second seed and a constrained-choice repair of invalid `issue_category` values, `src/repair_enums.py`, are prepared but not run); AuditKIT and our scorer disagree by a few points because of how strictly they parse;
GPU prices are assumed; the hosted-model comparison reflects prices and model versions on the day we ran it; the full-prompt wording
was corrected once after seeing test failures.

## Reproduce

```bash
pip install -r requirements.txt                                # Python 3.11+
# round 1 (v1 data)
python src/generate_data.py                                    # teacher data (needs a Cohere key in .env)
python src/curate.py && python src/prepare_splits.py           # CuratorKIT, then train/val/test
# GPU (Kaggle T4): notebooks/lexsi_pipeline.ipynb              # minimal-prompt baseline, AlignTune LoRA, tuned eval
# GPU:             notebooks/lexsi_baseline_full.ipynb         # untuned model with the full schema prompt
python src/eval_closed.py --models google/gemini-2.5-flash-lite anthropic/claude-haiku-5.5 --modes full   # needs an OpenRouter key
python src/audit.py                                            # AuditKIT over every predictions file
python src/roi.py                                              # cost table
# v2 data (needs an OpenRouter key): generate, curate, build the test set
python src/generate_data_v2.py --calls 130 --out data/v2_train_raw.jsonl --seed 2000
python src/generate_data_v2.py --calls 50 --out data/v2_test_raw.jsonl --seed 1000
python src/build_v2.py prepare
python src/curate.py --raw data/v2_train_candidates.jsonl --out data/curated_v2 --name lexsi_extraction_v2 --splits 0.9 0.1 0
python src/build_v2.py finalize --test-size 120
python src/eval_closed.py --models google/gemini-2.5-flash-lite --data data/final_v2 --out results/v2_closed
python src/audit.py --data data/final_v2 --closed v2_closed --runs --out audit_summary_v2.json
# GPU: notebooks/lexsi_pipeline_v2.ipynb                       # Tiny Aya on v2 (reads the v2 dataset from Hugging Face)
# GPU: notebooks/lexsi_seed2_repair.ipynb                      # prepared, not yet run: a second training seed (7) and enum repair
python src/bootstrap.py --data data/final_v2 --a <predictions A> --b <predictions B>   # paired bootstrap interval
python -m pytest -q src                                        # 14 tests, including "every gold output matches the schema"
```

Keys live in a git-ignored `.env`; nothing secret is in this repository.

## Adapters (Hugging Face model repo)

The files at the top of `iamvisheshsrivastava/lexsi-tiny-aya-extraction-lora` are the **round-2 adapter** (510 training documents, the one in
the main results above). The round-1 adapter (188 documents) is in `adapter_v1/`. Load either on top of `CohereLabs/tiny-aya-global`
(accept its licence first) with PEFT. Logs: `logs_v2/` (round 2) and `logs/` (round 1).

## Files

`src/` code and tests, `results/` metrics, predictions and cost tables (`audit_summary*.json`, `roi*.md`, `baseline*.json`, `tuned.json`,
`closed/`, `v2_closed/`), `notebooks/` the four Kaggle notebooks (the last, `lexsi_seed2_repair.ipynb`, has not been run), `data/final/` (v1) and `data/final_v2/` (v2) the splits, plus the raw
generated files and logs. Hugging Face: the adapters, logs and metrics
([`iamvisheshsrivastava/lexsi-tiny-aya-extraction-lora`](https://huggingface.co/iamvisheshsrivastava/lexsi-tiny-aya-extraction-lora)) and the datasets
[`lexsi-extraction-data`](https://huggingface.co/datasets/iamvisheshsrivastava/lexsi-extraction-data) (v1) and
[`lexsi-extraction-data-v2`](https://huggingface.co/datasets/iamvisheshsrivastava/lexsi-extraction-data-v2).

## Licence

The code is MIT-licensed (`LICENSE`). The adapters are derived from Tiny Aya Global (CC-BY-NC-4.0) and the data was generated with Cohere
Command A (CC-BY-NC-4.0) and DeepSeek V4 Flash, so the adapters and the datasets are released under CC-BY-NC-4.0 and follow those
services' terms: non-commercial use, with attribution to Cohere Labs for the base model.
