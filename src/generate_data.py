"""Generate synthetic multilingual tickets/invoices with their gold JSON using a Cohere teacher model.

Each call asks for BATCH documents for one (doc_type, language, industry, style) combination. Output goes
to data/raw_generated.jsonl (git-ignored). Labels are checked against the schema and for grounding:
party name, reference id and amount must literally appear in the text.

Run: python src/generate_data.py --calls 5      (small trial)
     python src/generate_data.py --calls 300    (full run; resumable)
Needs COHERE key in .env as TEACHER_API_KEY.
"""
import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import cohere
from dotenv import load_dotenv
from jsonschema import Draft202012Validator

from grounding import numbers_in
from schema import CATEGORIES, DOC_TYPES, LANGUAGES, PRIORITIES, SCHEMA

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data" / "raw_generated.jsonl"
MODEL = "command-a-03-2025"
BATCH = 5
INDUSTRIES = ["e-commerce", "telecom", "banking app", "airline", "food delivery", "SaaS software", "electronics retail",
              "utility company", "online education", "travel booking", "insurance", "furniture store"]
STYLES = ["short and angry", "polite and detailed", "messy with typos", "formal business tone", "casual chat style",
          "copy-pasted email with signature and footer", "bullet-point list"]

PROMPT = """Create {n} realistic, different {kind} documents written in {language}. Industry: {industry}. Style: {style}.
For each document also give the correct extraction label as a JSON object with exactly these keys:
doc_type ("{doc_type}"), language ("{code}"), party_name, reference_id, product, issue_category (one of {cats}),
priority (one of {prios}; infer from urgency), amount (number or null), currency (3-letter ISO code or null),
date_iso (YYYY-MM-DD or null).
Rules: party_name, reference_id and amount must appear literally in the document text (amount as digits). Use null
for anything the document does not state. Vary names, ids, dates, products, amounts. Some documents may omit fields.
Return ONLY a JSON object: {{"items": [{{"text": "...", "label": {{...}}}}, ...]}}"""


def valid(item, validator, code, doc_type):
    label = item.get("label")
    text = item.get("text")
    if not isinstance(label, dict) or not isinstance(text, str):
        return "malformed"
    errors = list(validator.iter_errors(label))
    if errors:
        return "schema: " + errors[0].message[:80]
    if label["language"] != code or label["doc_type"] != doc_type:
        return "wrong language/doc_type"
    low = text.casefold()
    for key in ("party_name", "reference_id"):
        if label[key] is not None and str(label[key]).casefold() not in low:
            return f"ungrounded {key}"
    if label["amount"] is not None and not any(abs(v - label["amount"]) < 0.005 for v in numbers_in(text)):
        return "ungrounded amount"
    return None


def main():
    load_dotenv(HERE.parent / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    client = cohere.ClientV2(os.environ["TEACHER_API_KEY"])
    validator = Draft202012Validator(SCHEMA)
    OUT.parent.mkdir(exist_ok=True)
    done = sum(1 for _ in open(OUT, encoding="utf-8")) if OUT.exists() else 0
    rng = random.Random(args.seed + done)
    kept = rejected = 0
    for call in range(args.calls):
        doc_type, code = rng.choice(DOC_TYPES), rng.choice(list(LANGUAGES))
        params = dict(n=BATCH, kind="customer support ticket" if doc_type == "support_ticket" else "invoice",
                      language=LANGUAGES[code], industry=rng.choice(INDUSTRIES), style=rng.choice(STYLES),
                      doc_type=doc_type, code=code, cats="/".join(CATEGORIES), prios="/".join(PRIORITIES))
        for attempt in range(6):
            try:
                reply = client.chat(model=MODEL, temperature=0.9, response_format={"type": "json_object"},
                                    messages=[{"role": "user", "content": PROMPT.format(**params)}])
                items = json.loads(reply.message.content[0].text)["items"]
                break
            except Exception as error:  # rate limit or malformed JSON: back off and retry
                print(f"  attempt {attempt + 1} failed: {type(error).__name__}: {str(error)[:200]}", flush=True)
                time.sleep(min(60, 5 * 2 ** attempt))
                items = []
        with open(OUT, "a", encoding="utf-8") as handle:
            for item in items:
                reason = valid(item, validator, code, doc_type)
                if reason is None:
                    handle.write(json.dumps({"text": item["text"], "label": item["label"], "industry": params["industry"],
                                             "style": params["style"]}, ensure_ascii=False) + "\n")
                    kept += 1
                else:
                    rejected += 1
                    print("   rejected:", reason, flush=True)
        print(f"call {call + 1}/{args.calls}: kept {kept}, rejected {rejected}", flush=True)
        time.sleep(3.2)  # stay under the trial-key rate limit


if __name__ == "__main__":
    main()
