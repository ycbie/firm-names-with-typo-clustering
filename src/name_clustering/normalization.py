import math
import re


def normalize_name(value: object) -> str:
    """Minimal firm-name normalization used throughout the project."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[^0-9a-z]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
