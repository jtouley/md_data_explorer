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
from collections.abc import Iterable, Mapping, Sequence
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

# A model whose predictor level, 0/1 predictor or outcome value is held by 1 to
# SMALL_CELL_THRESHOLD - 1 patients. It names no column or level: for a 0/1 predictor the column
# is the label of its level, so naming it would tell the reader a few patients have it.
SMALL_VALUE_MODEL_REFUSAL: Final = (
    f"A predictor or outcome value is held by fewer than {SMALL_CELL_THRESHOLD} patients, so this model can't be "
    f"shown without identifying them. Try fewer or broader variables."
)
# A model whose every predictor value and outcome value has SMALL_CELL_THRESHOLD or more patients,
# but where one predictor value has 1 to SMALL_CELL_THRESHOLD - 1 patients with one outcome value:
# the odds ratio and its CI, with totals shown elsewhere, give that number back. Names no column.
SMALL_CELL_MODEL_REFUSAL: Final = (
    f"Fewer than {SMALL_CELL_THRESHOLD} patients have one combination of a predictor value and the outcome, "
    f"so this model can't be shown without letting their number be worked out. Try a broader predictor or "
    f"a more common outcome."
)
# A model whose fitted terms are not exactly the terms its sizes were counted for: two predictors
# give the same term name (a column "site_West" beside "site" with level West), so one term's size
# would stand for the other's. Withheld (fail closed); names neither column, since one of them
# may be the label of a small level.
TERM_NAME_CLASH_MODEL_REFUSAL: Final = (
    "Two predictors give this model the same term name, so it can't be checked for small groups and "
    "is not shown. Rename one of the columns or leave it out."
)
# A stored model whose group sizes are missing (a payload from before they were recorded): it is
# withheld (fail closed), but no value is small, so none is blamed.
UNRECORDED_SIZES_MODEL_REFUSAL: Final = (
    "This model can't be shown because its group sizes weren't recorded; ask again to recompute."
)


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

    A count that is missing, not an integer or negative is treated as small (fail closed). Zero
    is not a level anyone has, so it is not small.
    """
    if isinstance(count, bool) or not isinstance(count, int | np.integer):
        return True
    return int(count) != 0 and int(count) < SMALL_CELL_THRESHOLD


def predictor_model_refusal(terms: Iterable[object], term_sizes: object, cell_sizes: object) -> str | None:
    """Why a logistic model with ``terms`` may not be shown, or None when it may.

    The one decision behind both ``compute_predictor_analysis`` (which refuses the model) and the
    presenter (which withholds a payload that skipped that refusal), so the two cannot disagree.
    ``term_sizes`` maps each term to the patients behind the level it names and its reference;
    ``cell_sizes`` to the fewest patients in one of those levels with one outcome value. Both
    are None for a slope, which names no level. A model is withheld when:

    * the sizes are not recorded for every term, or a term is a slope in one and not the other
      (a payload from before they were recorded, or a malformed one): ``UNRECORDED_SIZES_MODEL_REFUSAL``;
    * a term's level holds 1-10 patients: ``SMALL_VALUE_MODEL_REFUSAL``;
    * a term's level holds 1-10 patients with one outcome value: ``SMALL_CELL_MODEL_REFUSAL``.

    A size that is not a count is small (``is_small_level``).
    """
    listed = list(terms)
    if not listed:
        return None
    if (
        not isinstance(term_sizes, Mapping)
        or not isinstance(cell_sizes, Mapping)
        or any(not isinstance(term, str) or term not in term_sizes or term not in cell_sizes for term in listed)
        or any((term_sizes[term] is None) != (cell_sizes[term] is None) for term in listed)
    ):
        return UNRECORDED_SIZES_MODEL_REFUSAL
    if any(term_sizes[term] is not None and is_small_level(term_sizes[term]) for term in listed):
        return SMALL_VALUE_MODEL_REFUSAL
    if any(cell_sizes[term] is not None and is_small_level(cell_sizes[term]) for term in listed):
        return SMALL_CELL_MODEL_REFUSAL
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
