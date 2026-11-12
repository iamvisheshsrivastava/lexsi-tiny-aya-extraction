"""Assemble the v2 dataset: a larger training set and a bigger, harder, leakage-checked test set.

Stage 1 (prepare): v1 documents (minus the 24 v1 test documents) + new v2 training documents -> data/v2_train_candidates.jsonl
Stage 2: run CuratorKIT on it (see the command printed by stage 1).
Stage 3 (finalize): read the CuratorKIT train/val export, build the v2 test set from data/v2_test_raw.jsonl after removing any
document that is a near-duplicate of a training/validation/v1-test document, and write data/final_v2/{train,val,test,test_v1}.jsonl.

The v1 test set is kept as test_v1.jsonl so older results stay comparable; test.jsonl is the new, larger v2 test set.

Run: python src/build_v2.py prepare
     <venv> python src/curate.py --raw data/v2_train_candidates.jsonl --out data/curated_v2 --name lexsi_extraction_v2 --splits 0.9 0.1 0
     python src/build_v2.py finalize [--test-size 120]
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from prepare_splits import read_split

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
SHINGLE = 5
NEAR_DUPLICATE = 0.7


def doc_text(prompt):
    return prompt.split("Document:\n", 1)[1]


def shingles(text):
    text = re.sub(r"\s+", " ", text.casefold())
    return {text[i:i + SHINGLE] for i in range(max(1, len(text) - SHINGLE + 1))}


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def round_robin(items, size):
    """Pick `size` items taking one per language in turn, so the test set is balanced across languages."""
    groups = {}
    for item in items:
        groups.setdefault(item["label"]["language"], []).append(item)
    picked = []
    while len(picked) < size and any(groups.values()):
        for code in sorted(groups):
            if groups[code] and len(picked) < size:
                picked.append(groups[code].pop(0))
    return picked


def prepare():
    v1_test = {doc_text(r["prompt"]) for r in read_jsonl(DATA / "final" / "test.jsonl")}
    rows = []
    for row in read_jsonl(DATA / "raw_generated.jsonl"):
        if row["text"] not in v1_test:
            rows.append({"text": row["text"], "label": row["label"], "source": "v1"})
    new = read_jsonl(DATA / "v2_train_raw.jsonl")
    rows += [{"text": r["text"], "label": r["label"], "source": "v2"} for r in new]
    with open(DATA / "v2_train_candidates.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} candidates ({sum(r['source'] == 'v1' for r in rows)} from v1, {len(new)} new); v1 test excluded: {len(v1_test)}")
    print("next: <venv python> src/curate.py --raw data/v2_train_candidates.jsonl --out data/curated_v2 "
          "--name lexsi_extraction_v2 --splits 0.9 0.1 0")


def finalize(test_size, seed):
    curated = DATA / "curated_v2"
    out = DATA / "final_v2"
    out.mkdir(exist_ok=True)
    sets, seen = {}, []
    for split in ("train", "val"):
        rows = read_split(split, curated)
        sets[split] = [{"prompt": f"{r['instruction']}\n\nDocument:\n{r['input']}", "output": r["output"]} for r in rows]
        seen += [doc_text(r["prompt"]) for r in sets[split]]
    test_v1 = read_jsonl(DATA / "final" / "test.jsonl")
    seen += [doc_text(r["prompt"]) for r in test_v1]
    seen_shingles = [shingles(t) for t in seen]
    candidates, dropped = [], 0
    for item in read_jsonl(DATA / "v2_test_raw.jsonl"):
        s = shingles(item["text"])
        if any(jaccard(s, other) >= NEAR_DUPLICATE for other in seen_shingles):
            dropped += 1
            continue
        # also drop near-duplicates inside the candidate test set itself
        if any(jaccard(s, shingles(c["text"])) >= NEAR_DUPLICATE for c in candidates):
            dropped += 1
            continue
        candidates.append(item)
    random.Random(seed).shuffle(candidates)
    test = round_robin(candidates, test_size)
    sets["test"] = [{"prompt": f"Extract the structured fields.\n\nDocument:\n{c['text']}",
                     "output": json.dumps(c["label"], ensure_ascii=False)} for c in test]
    for split, rows in sets.items():
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as handle:
            for r in rows:
                gold = json.loads(r["output"])
                handle.write(json.dumps({"prompt": r["prompt"], "instruction": r["prompt"], "output": r["output"],
                                         "language": gold["language"], "doc_type": gold["doc_type"]}, ensure_ascii=False) + "\n")
    with open(out / "test_v1.jsonl", "w", encoding="utf-8") as handle:
        for r in test_v1:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"train": len(sets["train"]), "val": len(sets["val"]), "test_v2": len(sets["test"]), "test_v1": len(test_v1),
               "test_candidates_dropped_as_near_duplicates": dropped,
               "test_v2_by_language": dict(Counter(json.loads(r["output"])["language"] for r in sets["test"])),
               "test_v2_by_doc_type": dict(Counter(json.loads(r["output"])["doc_type"] for r in sets["test"])),
               "train_by_language": dict(Counter(json.loads(r["output"])["language"] for r in sets["train"]))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "finalize"])
    parser.add_argument("--test-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    prepare() if args.stage == "prepare" else finalize(args.test_size, args.seed)


if __name__ == "__main__":
    main()
