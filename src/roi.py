"""Cost per 1,000 documents: hosted models (billed token usage, measured) versus self-hosted Tiny Aya (measured speed).

Hosted costs are what OpenRouter billed for the 24 test documents, scaled to 1,000 (results/closed/*_full.json).
Self-hosted cost uses the measured generation time per document on one Kaggle T4 (results/tuned.json and
results/baseline_full.json) multiplied by an ASSUMED hourly GPU price; the prices below are assumptions, not quotes.
The tuned model's time includes a few outputs that ran to the 256-token limit and held up their batch of 8, so its
speed is not a fair measure of what a tuned server could do; the untuned run did not have that problem.

Run: python src/roi.py   ->  results/roi.json and results/roi.md
     python src/roi.py --closed v2_closed --local "Tiny Aya, fine-tuned, short prompt=v2_tuned.json" --prefix v2_roi   (v2 test set)
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
GPU_PRICES = {"Kaggle free tier": 0.0, "assumed cheap T4 rental": 0.35, "assumed on-demand T4": 0.80}
SELF_HOSTED = {"Tiny Aya, fine-tuned, short prompt": "tuned.json", "Tiny Aya, untuned, full schema prompt": "baseline_full.json"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--closed", default="closed", help="sub-folder of results/ with the hosted-model runs")
    parser.add_argument("--local", nargs="*", metavar="LABEL=FILE", help="self-hosted result files (default: the v1 tuned and untuned runs)")
    parser.add_argument("--prefix", default="roi", help="output file stem inside results/")
    args = parser.parse_args()
    self_hosted = dict(item.split("=", 1) for item in args.local) if args.local is not None else SELF_HOSTED
    hosted = {}
    for path in sorted((RESULTS / args.closed).glob("*_full.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        hosted[d["model"]] = {"cost_per_1000": d["cost_per_1000_docs_usd"], "prompt_tokens": d["tokens_per_doc"]["prompt"],
                              "completion_tokens": d["tokens_per_doc"]["completion"]}
    measured, self_cost = {}, {}
    for label, name in self_hosted.items():
        d = json.loads((RESULTS / name).read_text(encoding="utf-8"))
        sec = d["seconds"] / d["overall"]["n"]
        measured[label] = {"seconds_per_document": sec, "documents_per_second": 1 / sec}
        self_cost[label] = {price_label: sec * 1000 / 3600 * price for price_label, price in GPU_PRICES.items()}
    breakeven = {m: {label: (price / 3600) / (v["cost_per_1000"] / 1000) for label, price in GPU_PRICES.items() if price > 0}
                 for m, v in hosted.items()}
    out = {"measured_T4_4bit_HF_generate_batch8": measured, "assumed_gpu_price_per_hour_usd": GPU_PRICES,
           "self_hosted_cost_per_1000_docs_usd": self_cost, "hosted_full_prompt": hosted,
           "breakeven_documents_per_second": breakeven}
    (RESULTS / f"{args.prefix}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = ["| Option | Cost per 1,000 documents | Notes |", "|---|---|---|"]
    for m, v in hosted.items():
        lines.append(f"| {m} (hosted) | ${v['cost_per_1000']:.3f} | billed; {v['prompt_tokens']:.0f} input + {v['completion_tokens']:.0f} output tokens per document |")
    for label, costs in self_cost.items():
        for price_label, cost in costs.items():
            lines.append(f"| {label}, self-hosted ({price_label}) | ${cost:.3f} | measured {measured[label]['seconds_per_document']:.1f} s per document on one T4; GPU price is an assumption |")
    lines += ["", "Throughput was measured with 4-bit weights and plain Hugging Face generation in batches of 8 on one Kaggle T4. A server built for "
              "throughput would be faster; that is not measured here.", "",
              "Documents per second a self-hosted server would need to match each hosted model's cost:", "",
              "| Hosted model | " + " | ".join(f"at ${p:.2f}/h" for p in GPU_PRICES.values() if p > 0) + " |",
              "|---|" + "---|" * sum(1 for p in GPU_PRICES.values() if p > 0)]
    for m, row in breakeven.items():
        lines.append(f"| {m} | " + " | ".join(f"{v:.2f}" for v in row.values()) + " |")
    (RESULTS / f"{args.prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
