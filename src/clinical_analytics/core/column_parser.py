"""
Column Parser - Extract metadata from column names.

Parses column names to extract:
- Display names (human-readable labels)
- Value mappings (e.g., "Yes: 1 No: 2" or "1: Normal 2: Osteopenia")
- Units (if embedded in name)

Handles real-world cases:
- Spaces in labels
- Punctuation
- Ranges
- Multi-digit codes
- Various formats (Yes:1 No:2, 1:Normal 2:Osteopenia)
"""

import re
from dataclasses import dataclass


@dataclass
class ColumnMetadata:
    """
    Metadata extracted from a column name.

    Attributes:
        canonical_name: Original column name (unchanged)
        display_name: Human-readable display name
        value_mapping: Dict mapping codes to labels {code: label}
        reverse_mapping: Dict mapping labels to codes {label: code}
        unit: Unit of measurement (if extracted from name)
    """

    canonical_name: str
    display_name: str
    value_mapping: dict[str, str] | None = None
    reverse_mapping: dict[str, str] | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        """Initialize reverse mapping if value_mapping exists."""
        if self.value_mapping is None:
            self.value_mapping = {}
        if self.reverse_mapping is None:
            # Build reverse mapping from value_mapping
            self.reverse_mapping = {v: k for k, v in self.value_mapping.items()}


def parse_column_name(column_name: str) -> ColumnMetadata:
    """
    Parse column name to extract metadata.

    Args:
        column_name: Original column name (e.g., "DEXA_SCAN_RESULT_Yes:1_No:2")

    Returns:
        ColumnMetadata with extracted information

    Example:
        >>> meta = parse_column_name("DEXA_SCAN_RESULT_Yes:1_No:2")
        >>> meta.display_name
        'DEXA Scan Result'
        >>> meta.value_mapping
        {'1': 'Yes', '2': 'No'}
    """
    canonical_name = column_name
    display_name = _extract_display_name(column_name)
    value_mapping, reverse_mapping = _extract_value_mapping(column_name)
    unit = _extract_unit(column_name)

    return ColumnMetadata(
        canonical_name=canonical_name,
        display_name=display_name,
        value_mapping=value_mapping,
        reverse_mapping=reverse_mapping,
        unit=unit,
    )


def _extract_display_name(column_name: str) -> str:
    """
    Extract human-readable display name from column name.

    Removes:
    - Value mapping suffixes (e.g., "_Yes:1_No:2")
    - Unit suffixes (e.g., "_mg_dl")
    - Common separators (underscores, dashes)

    Example:
        >>> _extract_display_name("DEXA_SCAN_RESULT_Yes:1_No:2")
        'DEXA Scan Result'
        >>> _extract_display_name("patient_age_years")
        'Patient Age'
    """
    # Remove value mapping patterns (e.g., "_Yes:1_No:2", "_1:Normal_2:Osteopenia")
    # Pattern: _ followed by label:code pairs
    name = re.sub(r"_[A-Za-z\s]+:\d+(?:_[A-Za-z\s]+:\d+)*", "", column_name)
    name = re.sub(r"_\d+:[A-Za-z\s]+(?:_\d+:[A-Za-z\s]+)*", "", name)

    # Remove unit suffixes (common patterns)
    unit_patterns = [
        r"_mg_dl$",
        r"_g_dl$",
        r"_mmol_l$",
        r"_years?$",
        r"_days?$",
        r"_months?$",
        r"_hours?$",
        r"_minutes?$",
        r"_seconds?$",
        r"_percent$",
        r"_pct$",
        r"_%$",
    ]
    for pattern in unit_patterns:
        name = re.sub(pattern, "", name, flags=re.IGNORECASE)

    # Convert underscores/dashes to spaces
    name = re.sub(r"[_-]+", " ", name)

    # Title case (preserve acronyms like DEXA)
    words = name.split()
    title_words = []
    for word in words:
        # If word is all uppercase (acronym), keep it
        if word.isupper() and len(word) > 1:
            title_words.append(word)
        else:
            title_words.append(word.title())

    return " ".join(title_words).strip()


def _extract_value_mapping(column_name: str) -> tuple[dict[str, str], dict[str, str]]:
    """
    Extract value mapping from column name.

    Handles multiple formats:
    - "Yes:1 No:2" → {'1': 'Yes', '2': 'No'}
    - "1:Normal 2:Osteopenia" → {'1': 'Normal', '2': 'Osteopenia'}
    - Spaces, punctuation, ranges

    Args:
        column_name: Column name potentially containing mapping

    Returns:
        Tuple of (code_to_label, label_to_code) dicts
    """
    code_to_label: dict[str, str] = {}
    label_to_code: dict[str, str] = {}

    # Pattern 1: Label:Code format (e.g., "Yes:1 No:2")
    # Match: word(s) : number
    pattern1 = r"([A-Za-z\s]+):(\d+)"
    matches1 = re.findall(pattern1, column_name)

    if matches1:
        for label, code in matches1:
            label = label.strip()
            code = code.strip()
            code_to_label[code] = label
            label_to_code[label] = code

    # Pattern 2: Code:Label format (e.g., "1:Normal 2:Osteopenia", "0: n/a 1: Atorvastatin")
    # Match: number : word(s) including special chars like "/" in "n/a"
    # Stop at next code:label pair or end of string
    pattern2 = r"(\d+):([A-Za-z\s/]+?)(?=\s+\d+:|$)"
    matches2 = re.findall(pattern2, column_name)

    if matches2:
        for code, label in matches2:
            label = label.strip()
            code = code.strip()
            code_to_label[code] = label
            label_to_code[label] = code

    return code_to_label, label_to_code


def _extract_unit(column_name: str) -> str | None:
    """
    Extract unit of measurement from column name.

    Common units: mg/dl, g/dl, mmol/l, years, days, months, hours, percent, etc.

    Args:
        column_name: Column name potentially containing unit

    Returns:
        Unit string if found, None otherwise
    """
    # Common unit patterns
    unit_patterns = [
        (r"_mg_dl$", "mg/dl"),
        (r"_g_dl$", "g/dl"),
        (r"_mmol_l$", "mmol/l"),
        (r"_years?$", "years"),
        (r"_days?$", "days"),
        (r"_months?$", "months"),
        (r"_hours?$", "hours"),
        (r"_minutes?$", "minutes"),
        (r"_seconds?$", "seconds"),
        (r"_percent$", "%"),
        (r"_pct$", "%"),
        (r"_%$", "%"),
    ]

    for pattern, unit in unit_patterns:
        if re.search(pattern, column_name, re.IGNORECASE):
            return unit

    return None
