"""Unit tests for the schema, the extraction metrics and the prompt builders. Run: python -m pytest -q src/test_eval_extraction.py"""
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from eval_closed import full_prompt
from eval_extraction import evaluate, field_scores, parse_json
from schema import FIELDS, INSTRUCTION, SCHEMA

DATA = Path(__file__).resolve().parent.parent / "data" / "final"
GOLD = {"doc_type": "invoice", "language": "en", "party_name": "Ana", "reference_id": "INV-1", "product": "Router",
        "issue_category": "billing", "priority": "low", "amount": 12.5, "currency": "EUR", "date_iso": "2024-05-01"}


def test_parse_json_variants():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! Here it is: {"a": {"b": 2}} Hope that helps.') == {"a": {"b": 2}}
    assert parse_json("no json here") is None
    assert parse_json("[1, 2, 3]") is None
    assert parse_json("") is None


def test_field_scores_ignore_case_whitespace_and_null():
    tp, n_pred, n_gold = field_scores({"party_name": " ANA ", "product": None, "amount": 12.0}, {"party_name": "ana", "product": None, "amount": 12})
    assert (tp, n_pred, n_gold) == (2, 2, 2)


def test_evaluate_perfect_and_empty():
    perfect = evaluate({0: json.dumps(GOLD)}, {0: GOLD}, SCHEMA)
    assert perfect["valid_json_rate"] == perfect["schema_pass_rate"] == perfect["exact_match_rate"] == 1.0
    assert perfect["field_f1"] == 1.0
    empty = evaluate({0: "nothing"}, {0: GOLD}, SCHEMA)
    assert empty["valid_json_rate"] == 0.0 and empty["field_f1"] == 0.0


def test_schema_rejects_null_enum_and_extra_keys():
    validator = Draft202012Validator(SCHEMA)
    assert not list(validator.iter_errors(GOLD))
    assert list(validator.iter_errors({**GOLD, "priority": None}))
    assert list(validator.iter_errors({**GOLD, "extra": 1}))
    assert list(validator.iter_errors({**GOLD, "currency": "eur"}))


def test_instruction_names_every_field_and_forbids_null_enums():
    assert all(field in INSTRUCTION for field in FIELDS)
    assert "never null" in INSTRUCTION  # the wording must not contradict the schema's non-nullable enums


def test_full_prompt_keeps_the_document():
    short = "Extract the structured fields.\n\nDocument:\nHallo Welt 12,50 EUR"
    built = full_prompt(short)
    assert built.startswith(INSTRUCTION) and built.endswith("Hallo Welt 12,50 EUR")


@pytest.mark.skipif(not (DATA / "test.jsonl").exists(), reason="curated splits not present")
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_gold_outputs_match_schema(split):
    validator = Draft202012Validator(SCHEMA)
    for line in open(DATA / f"{split}.jsonl", encoding="utf-8"):
        row = json.loads(line)
        assert not list(validator.iter_errors(json.loads(row["output"]))), row["output"]
        assert "Document:\n" in row["prompt"]
