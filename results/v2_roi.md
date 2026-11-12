| Option | Cost per 1,000 documents | Notes |
|---|---|---|
| anthropic/claude-haiku-5.5 (hosted) | $0.235 | billed; 376 input + 395 output tokens per document |
| cohere/command-a (hosted) | $1.746 | billed; 269 input + 107 output tokens per document |
| google/gemini-2.5-flash-lite (hosted) | $0.072 | billed; 266 input + 114 output tokens per document |
| openai/gpt-5.4-nano (hosted) | $0.167 | billed; 239 input + 96 output tokens per document |
| Tiny Aya, fine-tuned, short prompt, self-hosted (Kaggle free tier) | $0.000 | measured 1.9 s per document on one T4; GPU price is an assumption |
| Tiny Aya, fine-tuned, short prompt, self-hosted (assumed cheap T4 rental) | $0.188 | measured 1.9 s per document on one T4; GPU price is an assumption |
| Tiny Aya, fine-tuned, short prompt, self-hosted (assumed on-demand T4) | $0.431 | measured 1.9 s per document on one T4; GPU price is an assumption |
| Tiny Aya, untuned, full schema prompt, self-hosted (Kaggle free tier) | $0.000 | measured 1.7 s per document on one T4; GPU price is an assumption |
| Tiny Aya, untuned, full schema prompt, self-hosted (assumed cheap T4 rental) | $0.165 | measured 1.7 s per document on one T4; GPU price is an assumption |
| Tiny Aya, untuned, full schema prompt, self-hosted (assumed on-demand T4) | $0.378 | measured 1.7 s per document on one T4; GPU price is an assumption |

Throughput was measured with 4-bit weights and plain Hugging Face generation in batches of 8 on one Kaggle T4. A server built for throughput would be faster; that is not measured here.

Documents per second a self-hosted server would need to match each hosted model's cost:

| Hosted model | at $0.35/h | at $0.80/h |
|---|---|---|
| anthropic/claude-haiku-5.5 | 0.41 | 0.94 |
| cohere/command-a | 0.06 | 0.13 |
| google/gemini-2.5-flash-lite | 1.34 | 3.07 |
| openai/gpt-5.4-nano | 0.58 | 1.33 |
