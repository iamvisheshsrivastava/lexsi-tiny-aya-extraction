---
license: cc-by-nc-4.0
language: [en, de, fr, hi]
task_categories: [text-generation]
tags: [structured-output, information-extraction, synthetic, multilingual]
pretty_name: Lexsi extraction data v2
---

# Structured extraction data, v2

Synthetic support tickets and invoices in English, German, French and Hinglish (romanised Hindi), each paired with a gold JSON
object (ten fields: `doc_type, language, party_name, reference_id, product, issue_category, priority, amount, currency, date_iso`).
Built for the Lexsi AI Safety and Alignment Hackathon, Track 1. Code: see the project README.

| File | Rows | Use |
|---|---|---|
| `train.jsonl` | 510 | fine-tuning (after CuratorKIT dedup and cleaning) |
| `val.jsonl` | 57 | validation |
| `test.jsonl` | 120 | held-out test, 30 documents per language, harder than v1 |
| `test_v1.jsonl` | 24 | the original v1 test set, kept so older results stay comparable |

Each row has `prompt` (minimal instruction plus the document), `output` (gold JSON as text), `language`, `doc_type`.

## How it was made

- 247 documents from v1 (Cohere `command-a-03-2025` as teacher; the 24 v1 test documents are excluded) and 330 new documents from
  `deepseek/deepseek-v4-flash` through OpenRouter, written under a difficulty list (amount is a total not a unit price, no reference
  id, local date formats, urgency only implied, a second person mentioned, and others). Documents may not contain the field names as
  labels, which v1 documents often did.
- Automatic checks reject an item unless the JSON matches the schema, the language and type match, `party_name`, `reference_id` and
  the amount appear in the text, and the date year appears in the text.
- CuratorKIT (MinHash 0.85, cleaning) produced the train and validation splits (577 in, 567 kept).
- The v2 test set was drawn separately, checked for near-duplicates (5-gram Jaccard 0.7) against train, validation, the v1 test set and
  itself (0 removed), and balanced by language.

## Limits

All text and labels are model-generated and were not checked by a person. Teacher labels can be wrong or debatable, in particular
`priority` and `issue_category` (subjective) and `party_name` when a document names both a company and a customer. The 120-document
test set is larger than v1 but still small. The data is synthetic, so scores on it do not transfer directly to real documents.

## Terms

The text was generated with Cohere `command-a-03-2025` (CC-BY-NC-4.0) and `deepseek/deepseek-v4-flash`. To stay within the stricter of those
terms the data is released under CC-BY-NC-4.0: non-commercial use only.
