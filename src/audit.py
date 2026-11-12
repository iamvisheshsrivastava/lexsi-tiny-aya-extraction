"""Audit every set of predictions on the test split with AuditKIT, using the same three scorers for all of them.

Runs audited (predictions are read from results/):
  baseline        untuned Tiny Aya Global, minimal prompt        results/baseline.predictions.json
  tuned           LoRA-tuned Tiny Aya Global, minimal prompt     results/tuned.predictions.json
  <model>_full    hosted models with the full schema prompt      results/closed/<model>_full.predictions.json
  <model>_short   hosted models with the minimal prompt          results/closed/<model>_short.predictions.json
Markdown code fences are stripped from every output before scoring (the same rule for all runs).

Scorers: IsJson (the whole output is JSON), JsonSchemaValid (validates against src/schema.py) and FieldF1 (per-field precision,
recall and F1 with number and case normalisation). Also saves a baseline -> tuned AuditKIT comparison.

Run: python src/audit.py            ->  results/audit_summary.json
"""
import argparse
import json
import re
from pathlib import Path

import auditkit
from auditkit import FieldF1, IsJson, JsonSchemaValid, Sample

from schema import SCHEMA

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"


def strip_fences(text):
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.M).strip()


def build_samples(rows, predictions):
    return [Sample(input=row["prompt"], target=row["output"], actual_output=strip_fences(predictions.get(str(i), "")),
                   id=str(i), metadata={"language": row["language"], "doc_type": row["doc_type"]})
            for i, row in enumerate(rows)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(RESULTS))
    parser.add_argument("--data", default=str(HERE.parent / "data" / "final"))
    parser.add_argument("--closed", default="closed", help="sub-folder of --results with the hosted-model predictions")
    parser.add_argument("--runs", nargs="*", metavar="NAME=FILE", help="local-model prediction files (default: the v1 baseline/tuned files)")
    parser.add_argument("--compare", nargs=2, default=["baseline", "tuned"], metavar=("BEFORE", "AFTER"))
    parser.add_argument("--out", default="audit_summary.json", help="summary file name inside --results")
    args = parser.parse_args()
    root = Path(args.results)
    rows = [json.loads(line) for line in open(Path(args.data) / "test.jsonl", encoding="utf-8")]
    scorers = [IsJson(), JsonSchemaValid(SCHEMA, strip_fences=True),
               FieldF1(normalize_numbers=True, case_sensitive=False, strip_fences=True)]
    if args.runs is not None:
        files = {item.split("=", 1)[0]: root / item.split("=", 1)[1] for item in args.runs}
    else:
        files = {"baseline": root / "baseline.predictions.json", "baseline_full_prompt": root / "baseline_full.predictions.json",
                 "tuned": root / "tuned.predictions.json"}
    for path in sorted((root / args.closed).glob("*.predictions.json")):
        files[path.name.replace(".predictions.json", "")] = path
    runs, summary = {}, {}
    for name, path in files.items():
        if not path.exists():
            continue
        predictions = json.loads(path.read_text(encoding="utf-8"))
        runs[name] = auditkit.evaluate(build_samples(rows, predictions), "precomputed", scorers,
                                       experiment_name=f"lexsi-extraction-{name}")
        summary[name] = {k: round(float(v), 4) for k, v in runs[name].headline.items()}
        print(f"{name:<44}", summary[name])
    out = {"n_test_documents": len(rows), "runs": summary}
    before, after = args.compare
    if before in runs and after in runs:
        comparison = auditkit.compare(runs[before], runs[after])
        out["comparison"] = {"before": before, "after": after, "summary": str(comparison.summary())}
        print(comparison.summary())
    (root / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
