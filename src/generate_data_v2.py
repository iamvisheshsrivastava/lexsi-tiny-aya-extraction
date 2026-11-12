"""Generate a harder, more natural v2 extraction dataset with a hosted teacher (OpenRouter).

Why v2: in the v1 data many documents state the field names outright ("Party Name: ...", "Reference ID: ..."), which leaks the
answer and makes every model look good. v2 asks for natural text with the fields implicit, plus deliberate difficulty features
(several amounts, local date formats, bystander names, missing ids, priority only implied by tone).

Checks applied to every item: schema, language and doc_type match, party name / reference id / amount grounded in the text, the
year of date_iso appears in the text, and no literal field-name labels. Items are also near-duplicate checked later (src/build_v2.py).
The teacher is NOT one of the models evaluated in the comparison, so the test set does not favour any of them.

Run: python src/generate_data_v2.py --calls 110 --out data/v2_test_raw.jsonl --seed 1000
     python src/generate_data_v2.py --calls 90  --out data/v2_train_raw.jsonl --seed 2000
Needs OPENROUTER_API_KEY in the repo-root .env (never printed). Spend is capped by --budget.
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import requests
from dotenv import dotenv_values
from jsonschema import Draft202012Validator

from grounding import numbers_in
from schema import CATEGORIES, DOC_TYPES, LANGUAGES, PRIORITIES, SCHEMA

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TEACHER = "deepseek/deepseek-v4-flash"
BATCH = 5
INDUSTRIES = ["e-commerce", "telecom", "banking app", "airline", "food delivery", "SaaS software", "electronics retail",
              "utility company", "online education", "travel booking", "insurance", "furniture store", "car rental",
              "pharmacy", "streaming service", "gym membership", "hotel", "courier company"]
STYLES = ["short and angry", "polite and detailed", "messy with typos", "formal business tone", "casual chat style",
          "copy-pasted email with signature and footer", "bullet-point list", "voice-message transcript", "one long paragraph"]
HARD = ["the text mentions both a unit price and a total, and the label amount is the total the customer must pay or was charged",
        "the date is written in a local format (for example 3 March 2024, 03.03.24 or March 3rd) and date_iso must be YYYY-MM-DD",
        "there is no reference id in the text, so reference_id is null",
        "a second person is mentioned (an agent, a colleague or a manager) and party_name is the customer or invoiced party only",
        "there is no amount or currency in the text, so both are null",
        "the urgency is only implied by the tone or by deadlines, never stated as a word like 'urgent' or 'high'",
        "the product is described in a sentence rather than named like a label",
        "the amount is written with a thousands separator and decimals in the local style"]
LABEL_WORDS = re.compile(r"\b(party name|reference id|reference_id|issue category|issue_category|priority\s*:|doc_type|date_iso)\b", re.I)

PROMPT = """Write {n} realistic, clearly different {kind} documents in {language}. Industry: {industry}. Style: {style}.
Write them the way real people or systems write: do NOT use form labels such as "Party Name:", "Reference ID:", "Priority:" or
"Category:"; the information must sit naturally inside the text. Difficulty for all documents: {hard}.
For each document give the correct extraction label as a JSON object with exactly these keys:
doc_type ("{doc_type}"), language ("{code}"), party_name, reference_id, product, issue_category (one of {cats}),
priority (one of {prios}; infer it from the urgency in the text), amount (number or null), currency (3-letter ISO code or null),
date_iso (YYYY-MM-DD or null).
Rules: party_name, reference_id and the amount digits must appear literally in the text; use null for anything the text does not
state; vary names, ids, dates, products and amounts; do not mention the labels or this task.
Return ONLY a JSON object: {{"items": [{{"text": "...", "label": {{...}}}}, ...]}}"""


def valid(item, validator, code, doc_type):
    label, text = item.get("label"), item.get("text")
    if not isinstance(label, dict) or not isinstance(text, str) or len(text) < 25:
        return "malformed"
    errors = list(validator.iter_errors(label))
    if errors:
        return "schema: " + errors[0].message[:80]
    if label["language"] != code or label["doc_type"] != doc_type:
        return "wrong language/doc_type"
    if LABEL_WORDS.search(text):
        return "field-name label in text"
    low = text.casefold()
    for key in ("party_name", "reference_id"):
        if label[key] is not None and str(label[key]).casefold() not in low:
            return f"ungrounded {key}"
    if label["amount"] is not None and not any(abs(v - label["amount"]) < 0.005 for v in numbers_in(text)):
        return "ungrounded amount"
    if label["date_iso"] is not None:
        year = label["date_iso"][:4]
        if year not in text and year[2:] not in text:
            return "date year not in text"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=float, default=0.5)
    args = parser.parse_args()
    key = dotenv_values(ROOT / ".env")["OPENROUTER_API_KEY"].strip().strip('"').strip("'")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    validator = Draft202012Validator(SCHEMA)
    done = sum(1 for _ in open(out, encoding="utf-8")) if out.exists() else 0
    rng = random.Random(args.seed + done)
    kept = rejected = 0
    spent = 0.0
    for call in range(args.calls):
        if spent >= args.budget:
            print("budget reached")
            break
        doc_type, code = rng.choice(DOC_TYPES), rng.choice(list(LANGUAGES))
        params = dict(n=BATCH, kind="customer support ticket" if doc_type == "support_ticket" else "invoice",
                      language=LANGUAGES[code], industry=rng.choice(INDUSTRIES), style=rng.choice(STYLES), hard=rng.choice(HARD),
                      doc_type=doc_type, code=code, cats="/".join(CATEGORIES), prios="/".join(PRIORITIES))
        items = []
        for attempt in range(5):
            try:
                resp = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=240,
                                     headers={"Authorization": f"Bearer {key}"},
                                     json={"model": TEACHER, "temperature": 0.9, "max_tokens": 4000, "usage": {"include": True},
                                           "response_format": {"type": "json_object"},
                                           "messages": [{"role": "user", "content": PROMPT.format(**params)}]})
                if resp.status_code != 200:
                    raise RuntimeError(f"http {resp.status_code}")
                data = resp.json()
                spent += float(data.get("usage", {}).get("cost", 0) or 0)
                text = data["choices"][0]["message"]["content"]
                text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
                items = json.loads(text)["items"]
                break
            except Exception as error:  # rate limit or malformed JSON: back off and retry
                print(f"  attempt {attempt + 1} failed: {type(error).__name__}: {str(error)[:120]}", flush=True)
                time.sleep(min(30, 3 * 2 ** attempt))
        with open(out, "a", encoding="utf-8") as handle:
            for item in items:
                reason = valid(item, validator, code, doc_type)
                if reason is None:
                    handle.write(json.dumps({"text": item["text"], "label": item["label"], "industry": params["industry"],
                                             "style": params["style"], "hard": params["hard"], "teacher": TEACHER},
                                            ensure_ascii=False) + "\n")
                    kept += 1
                else:
                    rejected += 1
        print(f"call {call + 1}/{args.calls}: kept {kept}, rejected {rejected}, spent ${spent:.4f}", flush=True)


if __name__ == "__main__":
    main()
