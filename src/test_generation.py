"""Tests for the v2 data checks, number grounding and the near-duplicate filter. Run: python -m pytest -q src/test_generation.py"""
from jsonschema import Draft202012Validator

from build_v2 import jaccard, shingles
from generate_data_v2 import valid
from grounding import numbers_in
from schema import SCHEMA

VALIDATOR = Draft202012Validator(SCHEMA)
LABEL = {"doc_type": "invoice", "language": "fr", "party_name": "Mme Fauconnier", "reference_id": "9901",
         "product": "nuitee", "issue_category": "billing", "priority": "high", "amount": 234.5, "currency": "EUR",
         "date_iso": "2025-01-20"}
TEXT = "FACTURE 9901 - Mme Fauconnier / Vous nous devez 234,50 EUR pour la nuitee. Emise le 20 janvier 2025."


def check(item, code="fr", doc_type="invoice"):
    return valid(item, VALIDATOR, code, doc_type)


def test_numbers_in_reads_both_decimal_styles():
    assert 1234.5 in numbers_in("Total 1.234,50 EUR")
    assert 1234.5 in numbers_in("Total 1,234.50 USD")
    assert 12.5 in numbers_in("Betrag 12,5")


def test_good_item_passes():
    assert check({"text": TEXT, "label": LABEL}) is None


def test_rejects_field_name_labels_and_ungrounded_values():
    assert check({"text": TEXT + " Party Name: Mme Fauconnier", "label": LABEL}) == "field-name label in text"
    assert check({"text": TEXT, "label": {**LABEL, "amount": 999.0}}) == "ungrounded amount"
    assert check({"text": TEXT, "label": {**LABEL, "party_name": "Unknown Person"}}) == "ungrounded party_name"
    assert check({"text": TEXT, "label": {**LABEL, "date_iso": "2031-01-20"}}) == "date year not in text"
    assert check({"text": TEXT, "label": {**LABEL, "priority": None}}).startswith("schema")
    assert check({"text": TEXT, "label": LABEL}, code="de") == "wrong language/doc_type"


def test_near_duplicate_filter():
    a = shingles("Ihre Rechnung 4711 über 99,00 EUR ist überfällig, bitte zahlen Sie sofort.")
    b = shingles("Ihre Rechnung 4711 über 99,00 EUR ist überfällig, bitte zahlen Sie heute.")
    c = shingles("My router stopped working yesterday and nobody answers the phone at your support line.")
    assert jaccard(a, b) >= 0.7
    assert jaccard(a, c) < 0.3
