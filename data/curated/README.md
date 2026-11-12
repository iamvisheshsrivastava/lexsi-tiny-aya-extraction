---
tags:
- curatorkit
configs:
- config_name: sft_alpaca
  default: true
  description: curatorkit sft_alpaca export, sha256 bf874b1e46e0ca10af6b96f2ff1f0e57eb56c08f4485487d8b2a1b039ed40626
  data_files:
  - split: test
    path: test/sft_alpaca.jsonl
  - split: train
    path: train/sft_alpaca.jsonl
  - split: val
    path: val/sft_alpaca.jsonl
---

# Dataset Card: lexsi_extraction

## Summary

- **Run timestamp**: 2026-11-07T22:18:23.492848Z
- **Wall clock time**: 2.1s
- **Pipeline config hash**: `3beb99a5452066d0`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 241 | 241 | — | 0 | 241 |
| SchemaGate | 241 | 241 | 0 | 0 | 241 |
| ExactDeduplicator | 241 | 241 | — | — | 241 |
| MinHashDeduplicator | 241 | 235 | — | — | 235 |
| TextCleaner | 235 | 235 | — | — | 235 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 241/0 passed (N/A), 0 rejected
- **SchemaGate**: 241/241 passed (100.0%), 0 rejected


**Rejection reasons:**
  - None

## Deduplication

  - near_duplicates_removed: 6
  - surviving_count: 235

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
| `sft_alpaca` | test (24), train (188), val (23) | `instruction`: string<br>`input`: string (empty when there is no input)<br>`output`: string |
