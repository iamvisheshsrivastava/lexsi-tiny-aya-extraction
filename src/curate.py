"""Curate teacher-generated examples with CuratorKIT: dedup, clean, split, export.

Input : a JSONL of {"text", "label"} (default data/raw_generated.jsonl, from generate_data.py)
Output: <out>/ (CuratorKIT export with the requested splits, manifest and rejected samples) and <data>/<name>_instructions.jsonl
        (the Alpaca-style file CuratorKIT ingests)

Run: python src/curate.py                                         # v1: 80/10/10 into data/curated
     python src/curate.py --raw data/v2_train_candidates.jsonl --out data/curated_v2 --name lexsi_extraction_v2 --splits 0.9 0.1 0
"""
import argparse
import json
from pathlib import Path

from curatorkit import Curator, CuratorConfig

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=str(DATA / "raw_generated.jsonl"))
    parser.add_argument("--out", default=str(DATA / "curated"))
    parser.add_argument("--name", default="lexsi_extraction")
    parser.add_argument("--splits", nargs=3, type=float, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--threshold", type=float, default=0.85, help="MinHash near-duplicate threshold")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.raw).read_text(encoding="utf-8").splitlines() if line.strip()]
    instructions = DATA / f"{args.name}_instructions.jsonl"
    with open(instructions, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({
                "instruction": "Extract the structured fields.",  # short on purpose: the full prompt is added in prepare_splits.py so dedup compares documents, not boilerplate
                "input": row["text"],
                "output": json.dumps(row["label"], ensure_ascii=False),
                "language": row["label"]["language"],
                "doc_type": row["label"]["doc_type"],
            }, ensure_ascii=False) + "\n")
    split = {name: ratio for name, ratio in zip(("train", "val", "test"), args.splits) if ratio > 0}
    result = Curator(CuratorConfig(
        dataset=str(instructions),
        dedup="minhash",
        minhash_threshold=args.threshold,
        clean=True,
        min_tokens=5,
        max_tokens=2048,
        export_formats=["alpaca"],
        output_split=split,
        output_dir=args.out,
        name=args.name,
    )).run()
    result.print_summary()


if __name__ == "__main__":
    main()
