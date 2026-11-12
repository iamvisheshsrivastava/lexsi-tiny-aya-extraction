"""Zero-shot extraction with hosted (closed or large) models on the held-out test set, for the ROI comparison.

Two prompts are used for every model:
  short  the minimal prompt the fine-tuned Tiny Aya was trained and tested with ("Extract the structured fields.")
  full   the full schema instruction (keys, allowed values, formats, "Output only the JSON")
The key is read from the repo-root .env (OPENROUTER_API_KEY), never printed. Spend is capped by --budget.

Run: python src/eval_closed.py --models anthropic/claude-haiku-5.5 google/gemini-2.5-flash-lite --budget 0.5
"""
import argparse
import json
import time
from pathlib import Path

import requests
from dotenv import dotenv_values

from eval_extraction import evaluate
from schema import INSTRUCTION, SCHEMA

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE.parent / "results" / "closed"
URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_TOKENS = {"openai/gpt-5.4-nano": 4000, "default": 1000}


def full_prompt(prompt):
    return INSTRUCTION + "\n\nDocument:\n" + prompt.split("Document:\n", 1)[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--data", default=str(HERE.parent / "data" / "final"))
    parser.add_argument("--budget", type=float, default=0.5)
    parser.add_argument("--modes", nargs="+", default=["short", "full"])
    parser.add_argument("--out", default=str(HERE.parent / "results" / "closed"), help="output folder")
    args = parser.parse_args()
    key = dotenv_values(ROOT / ".env")["OPENROUTER_API_KEY"].strip().strip('"').strip("'")
    rows = [json.loads(line) for line in open(Path(args.data) / "test.jsonl", encoding="utf-8")]
    references = {i: json.loads(r["output"]) for i, r in enumerate(rows)}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    spent = 0.0
    for model in args.models:
        for mode in args.modes:
            predictions, usage = {}, {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "seconds": 0.0}
            for i, row in enumerate(rows):
                if spent >= args.budget:
                    raise SystemExit("budget reached")
                text = row["prompt"] if mode == "short" else full_prompt(row["prompt"])
                payload = {"model": model, "temperature": 0, "max_tokens": MAX_TOKENS.get(model, MAX_TOKENS["default"]),
                           "messages": [{"role": "user", "content": text}], "usage": {"include": True}}
                started = time.time()
                for attempt in range(4):
                    try:
                        resp = requests.post(URL, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=180)
                        if resp.status_code == 200:
                            break
                    except requests.RequestException:
                        pass
                    time.sleep(3 * (attempt + 1))
                else:
                    predictions[i] = ""
                    continue
                data = resp.json()
                usage["seconds"] += time.time() - started
                u = data.get("usage", {})
                usage["prompt_tokens"] += u.get("prompt_tokens", 0)
                usage["completion_tokens"] += u.get("completion_tokens", 0)
                cost = float(u.get("cost", 0) or 0)
                usage["cost"] += cost
                spent += cost
                predictions[i] = data["choices"][0]["message"].get("content") or ""
            result = {"model": model, "prompt_mode": mode, "overall": evaluate(predictions, references, SCHEMA), "usage": usage,
                      "tokens_per_doc": {"prompt": usage["prompt_tokens"] / len(rows), "completion": usage["completion_tokens"] / len(rows)},
                      "cost_per_1000_docs_usd": usage["cost"] / len(rows) * 1000}
            slug = model.replace("/", "_")
            (out_dir / f"{slug}_{mode}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            (out_dir / f"{slug}_{mode}.predictions.json").write_text(json.dumps(predictions, indent=1, ensure_ascii=False), encoding="utf-8")
            o = result["overall"]
            print(f"{model:<34} {mode:<5} valid {o['valid_json_rate']:.2f} schema {o['schema_pass_rate']:.2f} F1 {o['field_f1']:.3f} "
                  f"exact {o['exact_match_rate']:.2f} | {result['tokens_per_doc']['prompt']:.0f}+{result['tokens_per_doc']['completion']:.0f} tok/doc "
                  f"${result['cost_per_1000_docs_usd']:.3f}/1k docs (spent ${spent:.3f})", flush=True)


if __name__ == "__main__":
    main()
