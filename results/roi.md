| Option | Cost per 1,000 documents | Notes |
|---|---|---|
| anthropic/claude-haiku-5.5 (hosted) | $0.194 | billed; 344 input + 320 output tokens per document |
| cohere/command-a (hosted) | $1.657 | billed; 250 input + 103 output tokens per document |
| google/gemini-2.5-flash-lite (hosted) | $0.069 | billed; 247 input + 111 output tokens per document |
| openai/gpt-5.4-nano (hosted) | $0.157 | billed; 222 input + 90 output tokens per document |
| Tiny Aya, fine-tuned, short prompt, self-hosted (Kaggle free tier) | $0.000 | measured 4.1 s per document on one T4; GPU price is an assumption |
| Tiny Aya, fine-tuned, short prompt, self-hosted (assumed cheap T4 rental) | $0.397 | measured 4.1 s per document on one T4; GPU price is an assumption |
| Tiny Aya, fine-tuned, short prompt, self-hosted (assumed on-demand T4) | $0.907 | measured 4.1 s per document on one T4; GPU price is an assumption |
| Tiny Aya, untuned, full schema prompt, self-hosted (Kaggle free tier) | $0.000 | measured 1.3 s per document on one T4; GPU price is an assumption |
| Tiny Aya, untuned, full schema prompt, self-hosted (assumed cheap T4 rental) | $0.131 | measured 1.3 s per document on one T4; GPU price is an assumption |
| Tiny Aya, untuned, full schema prompt, self-hosted (assumed on-demand T4) | $0.300 | measured 1.3 s per document on one T4; GPU price is an assumption |

Throughput was measured with 4-bit weights and plain Hugging Face generation in batches of 8 on one Kaggle T4. A server built for throughput would be faster; that is not measured here.

Documents per second a self-hosted server would need to match each hosted model's cost:

| Hosted model | at $0.35/h | at $0.80/h |
|---|---|---|
| anthropic/claude-haiku-5.5 | 0.50 | 1.14 |
| cohere/command-a | 0.06 | 0.13 |
| google/gemini-2.5-flash-lite | 1.41 | 3.21 |
| openai/gpt-5.4-nano | 0.62 | 1.41 |
