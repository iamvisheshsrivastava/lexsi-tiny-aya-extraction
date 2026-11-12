"""Turn the CuratorKIT export into flat train/val/test JSONL files for training and evaluation.

Each line: {"prompt": instruction + document, "output": gold JSON string, "language", "doc_type"}.
CuratorKIT dedups on the raw document text; the instruction is merged in only here.

Run: python src/prepare_splits.py
"""
import json
from pathlib import Path

from schema import INSTRUCTION

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"


def read_split(name, curated=None):
    rows = []
    for path in sorted(((curated or DATA / "curated") / name).rglob("*")):
        if path.suffix in (".jsonl", ".json") and path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if path.suffix == ".jsonl":
                rows += [json.loads(line) for line in text.splitlines() if line.strip()]
            else:
                loaded = json.loads(text)
                rows += loaded if isinstance(loaded, list) else [loaded]
    return rows


def main():
    out = DATA / "final"
    out.mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        rows = read_split(split)
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as handle:
            for row in rows:
                gold = json.loads(row["output"])
                handle.write(json.dumps({
                    "prompt": f"{row['instruction']}\n\nDocument:\n{row['input']}",
                    "instruction": f"{row['instruction']}\n\nDocument:\n{row['input']}",
                    "output": row["output"],
                    "language": gold["language"],
                    "doc_type": gold["doc_type"],
                }, ensure_ascii=False) + "\n")
        print(split, len(rows))


if __name__ == "__main__":
    main()
