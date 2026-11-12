"""Metrics for structured-extraction outputs: JSON validity, schema pass, field F1 and exact match.

Predictions and references are dicts keyed by example id. A prediction is the raw model text; the
reference is the gold JSON object. Usage as a module: evaluate(predictions, references, schema).
"""
import json
import re

from jsonschema import Draft202012Validator


def parse_json(text):
    """Return the first JSON object in text, or None."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start = text.find("{")
        while start != -1:
            depth = 0
            for end in range(start, len(text)):
                depth += text[end] == "{"
                depth -= text[end] == "}"
                if depth == 0:
                    try:
                        value = json.loads(text[start:end + 1])
                        return value if isinstance(value, dict) else None
                    except json.JSONDecodeError:
                        break
            start = text.find("{", start + 1)
    return None


def normalise(value):
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip().casefold()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def field_scores(prediction, reference):
    """Return (true positives, predicted fields, reference fields) over top-level key/value pairs."""
    gold = {k: normalise(v) for k, v in reference.items() if v not in (None, "", [])}
    pred = {k: normalise(v) for k, v in (prediction or {}).items() if v not in (None, "", [])}
    tp = sum(1 for k, v in pred.items() if gold.get(k) == v)
    return tp, len(pred), len(gold)


def evaluate(predictions, references, schema):
    validator = Draft202012Validator(schema)
    n = len(references)
    valid_json = schema_pass = exact = 0
    tp = n_pred = n_gold = 0
    for key, reference in references.items():
        parsed = parse_json(predictions.get(key, ""))
        valid_json += parsed is not None
        if parsed is not None and not list(validator.iter_errors(parsed)):
            schema_pass += 1
        a, b, c = field_scores(parsed, reference)
        tp, n_pred, n_gold = tp + a, n_pred + b, n_gold + c
        exact += parsed is not None and a == b == c
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    return {
        "n": n,
        "valid_json_rate": valid_json / n,
        "schema_pass_rate": schema_pass / n,
        "field_precision": precision,
        "field_recall": recall,
        "field_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "exact_match_rate": exact / n,
    }
