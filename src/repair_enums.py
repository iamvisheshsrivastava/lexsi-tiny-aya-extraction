"""Repair invalid enum values in model outputs by letting the model choose among the allowed values.

The fine-tuned model sometimes writes a value that is not allowed for a closed field (for example issue_category "error" or
"delivery"). This script finds such values in the generated text and replaces each with the allowed value that the SAME model rates
most likely at that position (the log-likelihood of the candidate, given the document prompt and the text generated before it). It
is constrained choice at the failing field only: no rules fitted to test documents, no changes to values that are already allowed, and
no help for missing keys or malformed JSON (those are counted, not fixed).

Run (GPU): python repair_enums.py --model CohereLabs/tiny-aya-global --adapter out/adapter --data data/final_v2 \
               --predictions results/v2_tuned.predictions.json --out results/v2_tuned_repaired.json
"""
import argparse
import json
import re
from pathlib import Path

from schema import CATEGORIES, DOC_TYPES, LANGUAGES, PRIORITIES

ENUMS = {"doc_type": DOC_TYPES, "language": list(LANGUAGES), "issue_category": CATEGORIES, "priority": PRIORITIES}


def find_invalid_enums(text):
    """Return [(key, start, end, current, quoted)] for closed fields whose value is not allowed.

    A quoted value that is not allowed (start/end delimit the text between the quotes) and a bare null (start/end delimit `null`)
    are both reported. Missing keys and numbers are not; use `count_unfixable` for those.
    """
    found = []
    for key, allowed in ENUMS.items():
        match = re.search(r'"%s"\s*:\s*(?:"([^"]*)"|(null))' % re.escape(key), text)
        if not match:
            continue
        if match.group(1) is not None and match.group(1) not in allowed:
            found.append((key, match.start(1), match.end(1), match.group(1), True))
        elif match.group(2) is not None:
            found.append((key, match.start(2), match.end(2), "null", False))
    return sorted(found, key=lambda item: item[1])


def count_unfixable(text):
    """Closed fields that are missing altogether; these are not repaired."""
    return sum(1 for key in ENUMS if not re.search(r'"%s"\s*:' % re.escape(key), text))


def repair_text(text, choose):
    """Replace each invalid enum value with choose(prefix, key, allowed); prefix is the text before the value, ending with an open quote.

    If choose returns None the value is left as it was.
    """
    changes, repaired, shift = [], text, 0
    for key, start, end, current, quoted in find_invalid_enums(text):
        prefix = repaired[: start + shift] + ("" if quoted else '"')
        best = choose(prefix, key, ENUMS[key])
        if best is None:
            continue
        replacement = best if quoted else '"' + best + '"'
        repaired = repaired[: start + shift] + replacement + repaired[end + shift:]
        shift += len(replacement) - (end - start)
        changes.append({"key": key, "from": current, "to": best})
    return repaired, changes


def main():
    import time

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from eval_extraction import evaluate
    from schema import SCHEMA

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--prompt", choices=["short", "full"], default="short")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cpu-test", action="store_true", help="float32 on CPU without 4-bit weights, for tiny test models only")
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(Path(args.data) / f"{args.split}.jsonl", encoding="utf-8")]
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.cpu_test:
        model = AutoModelForCausalLM.from_pretrained(args.model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"))
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    def prompt_text(row):
        from schema import INSTRUCTION
        content = row["prompt"] if args.prompt == "short" else INSTRUCTION + "\n\nDocument:\n" + row["prompt"].split("Document:\n", 1)[1]
        if tokenizer.chat_template is None:  # only for tiny test models
            return content + "\n"
        return tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)

    start_token = "<|START_RESPONSE|>"
    probe = "" if tokenizer.chat_template is None else tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}], tokenize=False)
    uses_start = start_token in probe  # the model's replies begin with this token, which decoding with skip_special_tokens drops

    def make_chooser(row):
        context = prompt_text(row)
        if uses_start and not context.endswith(start_token):
            context += start_token
        n_context = tokenizer(context, add_special_tokens=False, return_tensors="pt").input_ids.shape[1]

        def choose(prefix, key, allowed):
            scores = []
            for candidate in allowed:
                ids = tokenizer(context + prefix + candidate + '"', return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
                with torch.no_grad():
                    logits = model(ids).logits[0, n_context - 1:-1].float().log_softmax(-1)
                # log-likelihood of everything after the prompt: the text before the field is the same for every candidate,
                # so the differences between candidates are the likelihood of the candidate value and its closing quote
                scores.append(logits.gather(1, ids[0, n_context:].unsqueeze(1)).sum().item())
            return allowed[max(range(len(allowed)), key=lambda i: scores[i])]
        return choose

    repaired, log, unfixable, start = {}, [], 0, time.time()
    for i, row in enumerate(rows):
        text = predictions.get(str(i), "")
        chooser = make_chooser(row)
        new_text, changes = repair_text(text, chooser)
        repaired[str(i)] = new_text
        unfixable += count_unfixable(text) > 0
        log += [dict(c, doc=i) for c in changes]
    references = {i: json.loads(r["output"]) for i, r in enumerate(rows)}
    before = evaluate({i: predictions.get(str(i), "") for i in range(len(rows))}, references, SCHEMA)
    after = evaluate({i: repaired[str(i)] for i in range(len(rows))}, references, SCHEMA)
    result = {"model": args.model, "adapter": args.adapter, "split": args.split, "documents": len(rows),
              "values_repaired": len(log), "documents_with_missing_closed_field": unfixable, "seconds": time.time() - start,
              "before": before, "after": after, "changes": log}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    Path(args.out).with_suffix(".predictions.json").write_text(json.dumps(repaired, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"values_repaired": len(log), "before": before, "after": after}, indent=2))


if __name__ == "__main__":
    main()
