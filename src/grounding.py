"""Helpers shared by the data generators."""
import re


def numbers_in(text):
    """Every number in the text, reading both 1,234.50 and 1.234,50 styles."""
    values = []
    for token in re.findall(r"\d[\d.,' ]*\d|\d", text):
        token = token.replace("'", "").replace(" ", "")
        if "," in token and "." in token:
            decimal = "," if token.rfind(",") > token.rfind(".") else "."
        elif "," in token:
            decimal = "," if len(token) - token.rfind(",") - 1 in (1, 2) else None
        elif "." in token:
            decimal = "." if len(token) - token.rfind(".") - 1 in (1, 2) else None
        else:
            decimal = None
        for sep in ",.":
            if sep != decimal:
                token = token.replace(sep, "")
        try:
            values.append(float(token.replace(",", ".")))
        except ValueError:
            pass
    return values
