"""Paired bootstrap confidence interval for the difference in per-document field F1 between two prediction files.

Per-document F1 uses the same matching rule as src/eval_extraction.py (micro over fields, normalised values), so the numbers are
close to, but not identical to, the AuditKIT table. Documents are resampled with replacement (10,000 times, seed 0).

Run: python src/bootstrap.py --data data/final_v2 --a results/v2_tinyaya/v2_baseline_full.predictions.json \
         --b results/v2_tinyaya/v2_tuned.predictions.json --out results/v2_bootstrap.json
"""
import argparse
import json
import random
from pathlib import Path

from eval_extraction import field_scores, parse_json


def per_document_f1(rows, predictions):
    scores = []
    for i, row in enumerate(rows):
        tp, n_pred, n_gold = field_scores(parse_json(predictions.get(str(i), "")), json.loads(row["output"]))
        scores.append(2 * tp / (n_pred + n_gold) if n_pred + n_gold else 1.0)
    return scores


def paired_bootstrap(a, b, rounds=10000, seed=0):
    rng = random.Random(seed)
    n = len(a)
    diffs = sorted(sum(b[j] - a[j] for j in (rng.randrange(n) for _ in range(n))) / n for _ in range(rounds))
    return diffs[int(0.025 * rounds)], diffs[int(0.975 * rounds)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--a", required=True, help="predictions of the first run")
    parser.add_argument("--b", required=True, help="predictions of the second run")
    parser.add_argument("--out")
    args = parser.parse_args()
    rows = [json.loads(line) for line in open(Path(args.data) / "test.jsonl", encoding="utf-8")]
    a = per_document_f1(rows, json.loads(Path(args.a).read_text(encoding="utf-8")))
    b = per_document_f1(rows, json.loads(Path(args.b).read_text(encoding="utf-8")))
    low, high = paired_bootstrap(a, b)
    result = {"n_documents": len(rows), "mean_f1_a": sum(a) / len(a), "mean_f1_b": sum(b) / len(b),
              "difference_b_minus_a": sum(b) / len(b) - sum(a) / len(a), "ci95_low": low, "ci95_high": high,
              "documents_b_better": sum(y > x for x, y in zip(a, b)), "documents_a_better": sum(x > y for x, y in zip(a, b))}
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
