---
tags:
- curatorkit
configs:
- config_name: sft_alpaca
  default: true
  description: curatorkit sft_alpaca export, sha256 2ddbbba442ab567669a724c69e2acbd06140401718455bc7c2fa1dd2495fc768
  data_files:
  - split: train
    path: train/sft_alpaca.jsonl
  - split: val
    path: val/sft_alpaca.jsonl
---

# Dataset Card: lexsi_extraction_v2

## Summary

- **Run timestamp**: 2026-11-12T18:31:52.919036Z
- **Wall clock time**: 12.8s
- **Pipeline config hash**: `c6fbeff5fb46f844`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 577 | 577 | — | 0 | 577 |
| SchemaGate | 577 | 577 | 0 | 0 | 577 |
| ExactDeduplicator | 577 | 577 | — | — | 577 |
| MinHashDeduplicator | 577 | 567 | — | — | 567 |
| TextCleaner | 567 | 567 | — | — | 567 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 577/0 passed (N/A), 0 rejected
- **SchemaGate**: 577/577 passed (100.0%), 0 rejected


**Rejection reasons:**
  - None

## Deduplication

  - near_duplicates_removed: 10
  - surviving_count: 567

## Known Limitations

- MinHash threshold used: **0.85** — this is a starting point, not a universal answer. Adjust per domain (legal, medical, and other domain-specific corpora may need a lower threshold to avoid losing structurally similar but semantically valid examples).
- Table extraction requires `extract_tables: true` on the PDF reader and MinerU with GPU support. Disabled by default.
- Token counting uses a whitespace tokenizer by default. For exact LLM context window
  enforcement, enable `use_tiktoken: true` in the SchemaGate config.
- Both within-run (MinHash/Diversity) and cross-run (EmbeddingDeduplicator) deduplication
  are available. Cross-run dedup persists an embedding index across runs; use
  `embedding_reset_index: true` to start fresh.

## Reproduction

```bash
pip install curatorkit
curatorkit run pipeline.yaml --output-dir ./out/
```

## Tool Versions

- **curatorkit**: 0.1.23
- **python**: 3.11.9

## Configs

Load one with `datasets.load_dataset("<this folder or Hub repo>", "<config>")`.

| Config | Splits (rows) | Columns |
|--------|---------------|---------|
| `sft_alpaca` | train (510), val (57) | `instruction`: string<br>`input`: string (empty when there is no input)<br>`output`: string |
