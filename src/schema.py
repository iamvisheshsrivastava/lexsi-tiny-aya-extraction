"""The extraction target: one flat JSON object per support ticket or invoice."""

DOC_TYPES = ["support_ticket", "invoice"]
LANGUAGES = {"en": "English", "de": "German", "fr": "French", "hi": "Hinglish (Hindi written in Latin script mixed with English)"}
CATEGORIES = ["billing", "shipping", "technical", "account", "refund", "other"]
PRIORITIES = ["low", "medium", "high", "urgent"]

FIELDS = ["doc_type", "language", "party_name", "reference_id", "product", "issue_category", "priority",
          "amount", "currency", "date_iso"]

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": FIELDS,
    "properties": {
        "doc_type": {"enum": DOC_TYPES},
        "language": {"enum": list(LANGUAGES)},
        "party_name": {"type": ["string", "null"]},
        "reference_id": {"type": ["string", "null"]},
        "product": {"type": ["string", "null"]},
        "issue_category": {"enum": CATEGORIES},
        "priority": {"enum": PRIORITIES},
        "amount": {"type": ["number", "null"]},
        "currency": {"type": ["string", "null"], "pattern": "^[A-Z]{3}$"},
        "date_iso": {"type": ["string", "null"], "pattern": "^\\d{4}-\\d{2}-\\d{2}$"},
    },
}

INSTRUCTION = (
    "Extract the information from the document below into a single JSON object with exactly these keys: "
    + ", ".join(FIELDS) + ". Use null for anything not stated, except doc_type, language, issue_category and priority, "
    "which must always be one of the allowed values and never null. "
    "doc_type is support_ticket or invoice; language is en, de, fr or hi (hi = Hinglish); "
    "issue_category is one of " + "/".join(CATEGORIES) + "; priority is one of " + "/".join(PRIORITIES) + "; "
    "amount is a number; currency is a 3-letter ISO code; date_iso is YYYY-MM-DD. Output only the JSON."
)
