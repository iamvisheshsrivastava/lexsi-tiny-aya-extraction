---
license: cc-by-nc-4.0
language: [en, de, fr, hi]
task_categories: [text-generation]
tags: [structured-output, information-extraction, synthetic, multilingual]
pretty_name: Lexsi extraction data v1
---

# Structured extraction data, v1 (round 1)

271 synthetic support tickets and invoices in English, German, French and Hinglish, each with a gold JSON object (ten fields:
`doc_type, language, party_name, reference_id, product, issue_category, priority, amount, currency, date_iso`). Written by Cohere
`command-a-03-2025`; amounts and dates were checked against the text. CuratorKIT (MinHash 0.85, cleaning) kept 235 and split them
**188 train / 23 validation / 24 test**. Rows have `prompt`, `instruction`, `output`, `language`, `doc_type`.

Limits: documents often contain the field names as labels ("Party Name:"), which makes extraction easy, and the 24-document test set
is small. Labels are model-written and unchecked. The harder, larger [v2 dataset](https://huggingface.co/datasets/iamvisheshsrivastava/lexsi-extraction-data-v2)
supersedes it for evaluation. Released under CC-BY-NC-4.0 (non-commercial) because of the generator's terms.
