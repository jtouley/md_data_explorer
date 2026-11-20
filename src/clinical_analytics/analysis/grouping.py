# SPDX-License-Identifier: MIT
"""Which columns may split patients into groups, and how category spellings are compared.

A column with more than ``MAX_GROUP_LEVELS`` levels is never a grouping or crosstab dimension
(``grouping_problem`` gives the sentence): a table of hundreds of exact values (one row per LDL
of 80.9) is a measurement read as categories, too long to read and no comparison of groups.
Ask about such a column as a measurement instead. (This limit is about what an answer can show
usefully, not about small cells: no level is hidden for its size, ADR015.)
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from clinical_analytics.analysis.variable_types import short_name
from clinical_analytics.contracts.config import LEVEL_SYNONYMS, LEVEL_VARIANT_MAX_LEVELS

# About 20 rows is what a reader can scan as a table of groups.
MAX_GROUP_LEVELS: Final = 20

# A model whose fitted terms are not exactly one per indicator: two predictors give the same term
# name (a column "site_West" beside "site" with level West), so one term's odds ratio would be
# read as the other's. Refused (the fit would be wrong), naming neither column.
TERM_NAME_CLASH_MODEL_REFUSAL: Final = (
    "Two predictors give this model the same term name, so its odds ratios can't be told apart. "
    "Rename one of the columns or leave it out."
)


def too_many_levels_message(column: str, levels: int) -> str:
    """Plain refusal for grouping by a column with too many levels."""
    name = short_name(column)
    return (
        f"{name} has {levels} different values, too many to split patients into groups that can be "
        f"read as a table. Group by a column with at most "
        f"{MAX_GROUP_LEVELS} categories, or ask about {name} as a measurement."
    )


def grouping_problem(column: str, levels: int) -> str | None:
    """Why ``column`` cannot be a grouping dimension, or None when it can."""
    if levels > MAX_GROUP_LEVELS:
        return too_many_levels_message(column, levels)
    return None


def normalised_level(text: str) -> str:
    """A category spelling as compared for variants: case, spacing and known synonyms removed."""
    key = re.sub(r"\s+", " ", text.strip()).casefold()
    return LEVEL_SYNONYMS.get(key, key)


def level_variants(levels: Sequence[object]) -> list[tuple[str, ...]]:
    """Groups of two or more distinct text levels that are the same after ``normalised_level``.

    "Male", "male" and "M" are one group. A column with more than ``LEVEL_VARIANT_MAX_LEVELS``
    levels is free text, not a category, and has none. Nulls and non-text values are ignored.
    """
    texts = sorted({level for level in levels if isinstance(level, str)})
    if len(texts) > LEVEL_VARIANT_MAX_LEVELS:
        return []
    groups: dict[str, list[str]] = {}
    for text in texts:
        groups.setdefault(normalised_level(text), []).append(text)
    return [tuple(group) for _, group in sorted(groups.items()) if len(group) > 1]
