"""Evaluate a model (optionally with a LoRA adapter) on the extraction test set.

Reports valid-JSON rate, schema pass rate, field F1, exact match (overall and per language / doc type)
plus generation throughput (for the ROI comparison). Meant for a GPU session (Kaggle / Colab).

Run: python eval_model.py --model CohereLabs/tiny-aya-global --data data/final --out results/baseline.json
     python eval_model.py --model CohereLabs/tiny-aya-global --adapter out/adapter --out results/tuned.json
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from eval_extraction import evaluate
from schema import INSTRUCTION, SCHEMA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--data", default="data/final")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--no-stop", action="store_true", help="do not stop at the closing brace (the first evaluation run did not)")
    parser.add_argument("--prompt", choices=["short", "full"], default="short",
                        help="short = the minimal prompt used for training; full = the full schema instruction")
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(Path(args.data) / f"{args.split}.jsonl", encoding="utf-8")]
    rows = rows[: args.limit] if args.limit else rows
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                               bnb_4bit_quant_type="nf4"))
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    # Stop when the model ends its turn, ends its response, or closes the (flat) JSON object. Without this a few
    # outputs ran to the token limit and held up their whole batch in the first evaluation run.
    end_response = tokenizer.convert_tokens_to_ids("<|END_RESPONSE|>")
    eos_ids = sorted({i for i in (tokenizer.eos_token_id, end_response) if isinstance(i, int) and i != tokenizer.unk_token_id})
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=eos_ids)
    if not args.no_stop:
        gen_kwargs.update(stop_strings=["}"], tokenizer=tokenizer)
    predictions, generated_tokens, at_limit, start = {}, 0, 0, time.time()
    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i + args.batch]
        texts = [tokenizer.apply_chat_template([{"role": "user", "content": r["prompt"] if args.prompt == "short" else
                                                 INSTRUCTION + "\n\nDocument:\n" + r["prompt"].split("Document:\n", 1)[1]}], tokenize=False,
                                               add_generation_prompt=True) for r in chunk]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            try:
                output = model.generate(**inputs, **gen_kwargs)
            except (TypeError, ValueError) as error:  # stop_strings unsupported by this transformers version: fall back
                if "stop_strings" not in gen_kwargs:
                    raise
                print("warning: stop_strings unavailable, generating without it:", str(error)[:120], flush=True)
                gen_kwargs.pop("stop_strings")
                gen_kwargs.pop("tokenizer")
                output = model.generate(**inputs, **gen_kwargs)
        for j, row in enumerate(chunk):
            new = output[j][inputs["input_ids"].shape[1]:]
            at_limit += int((new != tokenizer.pad_token_id).sum() >= args.max_new_tokens)
            generated_tokens += int((new != tokenizer.pad_token_id).sum())
            predictions[i + j] = tokenizer.decode(new, skip_special_tokens=True)
        print(f"[eval] {min(i + args.batch, len(rows))}/{len(rows)} done, {time.time() - start:.0f}s elapsed", flush=True)
    seconds = time.time() - start

    references = {i: json.loads(r["output"]) for i, r in enumerate(rows)}
    result = {"model": args.model, "adapter": args.adapter, "split": args.split, "prompt": args.prompt,
              "stop_at_brace": not args.no_stop, "outputs_at_token_limit": at_limit, "n_documents": len(rows),
              "overall": evaluate(predictions, references, SCHEMA),
              "tokens_per_second": generated_tokens / seconds, "seconds": seconds}
    for field in ("language", "doc_type"):
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            groups[r[field]].append(i)
        result["by_" + field] = {g: evaluate({i: predictions[i] for i in idx}, {i: references[i] for i in idx}, SCHEMA)
                                 for g, idx in groups.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    Path(args.out).with_suffix(".predictions.json").write_text(json.dumps(predictions, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
