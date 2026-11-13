# SPDX-License-Identifier: MIT
"""Which columns may split patients into groups, and how a small group's label is shown.

A group label is shown next to per-group values. A column with hundreds of levels (an exact
LDL of 80.9) turns every label into one patient's value, and a small group's label beside
another group's 0 pins those patients. So:

* a column with more than ``MAX_GROUP_LEVELS`` levels is never a grouping or crosstab
  dimension (``grouping_problem`` gives the refusal sentence);
* a level with 1 to ``SMALL_CELL_THRESHOLD - 1`` patients is never shown under its own label:
  the presenter merges such levels into one ``OTHER_LABEL`` row or column. A level hidden only
  as their complement keeps its own label (its values hidden), unless that would pin a small
  count; then every hidden level folds under ``HIDDEN_GROUPS_LABEL``, which claims no size.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

import numpy as np

from clinical_analytics.analysis.variable_types import short_name
from clinical_analytics.contracts.config import LEVEL_SYNONYMS, LEVEL_VARIANT_MAX_LEVELS
from clinical_analytics.core.nl_query_config import SMALL_CELL_THRESHOLD

# About 20 rows is what a reader can scan, and far fewer groups than patients in any cohort the
# small-cell rule lets through (21 groups of 11 is already 231 patients).
MAX_GROUP_LEVELS: Final = 20

OTHER_LABEL: Final = f"Other (groups under {SMALL_CELL_THRESHOLD})"
# The fold's label when it must also take a group hidden only as a complement (keeping that
# group's own row would tell a reader it is not small and pin the small one): it then claims no
# size, so a 150-patient group is never labelled "under 11".
HIDDEN_GROUPS_LABEL: Final = "Other groups (hidden)"
FOLD_LABELS: Final = frozenset({OTHER_LABEL, HIDDEN_GROUPS_LABEL})


def too_many_levels_message(column: str, levels: int) -> str:
    """Plain refusal for grouping by a column with too many levels."""
    name = short_name(column)
    return (
        f"{name} has {levels} different values, so it cannot be used to split patients into groups: "
        f"each group would show a few patients' exact values. Group by a column with at most "
        f"{MAX_GROUP_LEVELS} categories, or ask about {name} as a measurement."
    )


def grouping_problem(column: str, levels: int) -> str | None:
    """Why ``column`` cannot be a grouping dimension, or None when it can."""
    if levels > MAX_GROUP_LEVELS:
        return too_many_levels_message(column, levels)
    return None


def is_small_level(count: object) -> bool:
    """A level whose own patient count must not be shown under its label (1..threshold-1).

    A count that is missing or not an integer is treated as small (fail closed). Zero is not
    a level anyone has, so it is not small.
    """
    if isinstance(count, bool) or not isinstance(count, int | np.integer):
        return True
    return 0 < int(count) < SMALL_CELL_THRESHOLD


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
