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

from clinical_analytics.analysis.cell_solvability import LEVEL_STATISTICS
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


# The share of patients a column may hold outside its MAX_GROUP_LEVELS most common values and
# still behave like categories. A 0/1 exposure with 21 patients at 21 odd values (0.37, 0.52,
# ...) beside 979 at 0 or 1 has 23 values, but a correlation, mean or odds ratio of it is that of
# the 0/1 table up to a few patients, so with the margins it gives a small cell back nearly
# exactly. A measurement spreads well past 5% of patients over values beyond its 20 most common.
MAX_STRAY_SHARE: Final = 0.05


def behaves_like_categories(sizes: Mapping[object, object]) -> bool:
    """Whether a column with these patients per value (``{value: patients}``, each value present)
    behaves like categories, so a statistic of it is a statistic of its table of counts.

    One rule for a model's predictor, a compared outcome and a correlated column. A column
    behaves like categories when it has at most ``MAX_GROUP_LEVELS`` values, or when all but
    ``MAX_STRAY_SHARE`` of its patients hold one of its ``MAX_GROUP_LEVELS`` most common values
    (a few stray values do not make a coded column a measurement). Its table is then counted
    over every value, stray ones included, and checked by ``is_small_level``.

    Why up to ``MAX_GROUP_LEVELS`` and not two: a column with that few values can be cross-
    tabulated (``grouping_problem``), and the crosstab shows every cell but the small ones and
    their complements. A group's mean, a slope's odds ratio or r is one more linear equation in
    those cells (a mean is the sum of value x cell over n), which with the line totals solves
    the hidden ones: a dose of 0, 0.5 or 1 is three groups of patients however it is stored.

    A column with more values, spread over more patients, is a measurement and rests on no
    table (``NO_COUNT_TABLE``): it cannot be cross-tabulated, and one equation does not pin its
    many cells. So is a column of three or more values where most patients hold a value nobody
    else has (12 patients, 12 blood pressures): its crosstab hides every cell and folds every
    label, so there is nothing for one equation to complete, and the small-group rules for
    sizes, means and medians already cover each patient's own value. Two values are always
    categories, as before. A size that is not a positive count cannot be weighed, so the column
    is then treated as categories (fail closed: its table is counted and checked).
    """
    if len(sizes) <= 2 or not all(_is_count(n) for n in sizes.values()):
        return True
    counts = sorted((int(n) for n in sizes.values() if isinstance(n, int | np.integer)), reverse=True)
    total = sum(counts)
    if 2 * sum(1 for n in counts if n == 1) > total:
        return False
    return len(counts) <= MAX_GROUP_LEVELS or sum(counts[MAX_GROUP_LEVELS:]) <= MAX_STRAY_SHARE * total


def _is_count(value: object) -> bool:
    """A positive patient count (a present value has at least one patient)."""
    return isinstance(value, int | np.integer) and not isinstance(value, bool) and int(value) > 0


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


# A statistic computed from a table of patient counts: a chi-square, Fisher or log-rank test of
# groups by outcome, and a correlation of two coded variables (for two 0/1 columns it is phi).
# With the table's totals, which other answers show, and the direction a claim states, the
# statistic is monotone in the cells (for a 2x2, in |ad - bc|), so it gives a cell back. It is
# withheld when any non-empty cell holds 1 to SMALL_CELL_THRESHOLD - 1 patients. The refusal
# names no column or level: for a 0/1 column the column is the label of its level.
SMALL_CELL_TEST_REFUSAL: Final = (
    f"A significance test or correlation isn't shown: fewer than {SMALL_CELL_THRESHOLD} patients have one "
    f"combination of the values compared, and its result, with totals shown elsewhere, would let their number "
    f"be worked out. Try broader groups or a more common outcome."
)
# A stored result whose table cell sizes are missing (a payload from before they were recorded):
# withheld (fail closed), but no cell is blamed.
UNRECORDED_CELLS_TEST_REFUSAL: Final = (
    "A significance test or correlation isn't shown because the sizes of the groups it rests on weren't "
    "recorded; ask again to recompute."
)
# ``table_min_cell`` of a statistic that rests on no table of counts (a correlation with a
# measurement, or group means of a measured outcome: neither column behaves like categories).
# Only compute records it; any other value that is not a count is withheld.
NO_COUNT_TABLE: Final = "no_count_table"


def table_statistic_refusal(min_cell: object, *, may_rest_on_no_table: bool = False) -> str | None:
    """Why a statistic computed from a table of patient counts may not be shown, or None.

    The one decision behind compute (which does not record a withheld statistic) and the
    presenter (which withholds one from a payload that skipped that step), so they cannot
    disagree: the counterpart, for a table, of ``predictor_model_refusal`` for a model's terms,
    with the same ``is_small_level`` rule for a count. ``min_cell``: the fewest patients in any
    non-empty cell of the tested table, as compute records it (``table_min_cell``).
    ``NO_COUNT_TABLE`` says the statistic rests on no table; it is accepted only with
    ``may_rest_on_no_table`` (a correlation, or a comparison of group means, either of which may
    be of a measurement), never for a test of a table, so a stray marker cannot let a test
    through.
    Fail closed:

    * None (not recorded), or ``NO_COUNT_TABLE`` where a table is required:
      ``UNRECORDED_CELLS_TEST_REFUSAL``;
    * anything else that is not a count of at least ``SMALL_CELL_THRESHOLD`` (1-10, 0 or a
      negative count, which a minimum over non-empty cells never is, a float, a bool, other
      text): ``SMALL_CELL_TEST_REFUSAL``.
    """
    if isinstance(min_cell, str) and min_cell == NO_COUNT_TABLE:
        return None if may_rest_on_no_table else UNRECORDED_CELLS_TEST_REFUSAL
    if min_cell is None:
        return UNRECORDED_CELLS_TEST_REFUSAL
    # ``is_small_level`` is the level rule (1-10, negative or not a count is small); a minimum
    # over non-empty cells is never 0, so a recorded 0 is malformed and withheld as well.
    if is_small_level(min_cell) or min_cell == 0:
        return SMALL_CELL_TEST_REFUSAL
    return None


# The payload field that records which statistics of a column that behaves like categories may be
# shown although one of its cells holds 1 to SMALL_CELL_THRESHOLD - 1 patients: a level of
# ``cell_solvability.LEVELS``, as compute records it from ``cell_solvability.statistics_level``.
STATISTICS_LEVEL: Final = "statistics_level"


def value_statistic_refusal(min_cell: object, level: object, statistic: str) -> str | None:
    """Why ``statistic`` ("mean", "std", "median", "min" or "max") of a column, or a test that
    shows it, may not be shown, or None when it may.

    The one decision behind compute and the presenters for every statistic of a column that may
    behave like categories (a comparison of means, a described column, a statistic per group),
    so they cannot disagree. ``min_cell``: the fewest patients holding one value in a group
    (``table_min_cell``); ``level``: the recorded ``STATISTICS_LEVEL``.

    * ``NO_COUNT_TABLE``: a measurement, which rests on no table of counts: shown;
    * a count of ``SMALL_CELL_THRESHOLD`` or more: no cell is hidden, shown;
    * a count of 1 to threshold-1: shown only when ``level`` is a level whose statistics
      (``cell_solvability.LEVEL_STATISTICS``) include ``statistic``: the exact test found that
      they, with everything below them, pin no hidden count. Not recorded, or anything else:
      ``SMALL_CELL_TEST_REFUSAL``;
    * anything else (None, malformed): as ``table_statistic_refusal`` (fail closed).
    """
    refusal = table_statistic_refusal(min_cell, may_rest_on_no_table=True)
    if refusal != SMALL_CELL_TEST_REFUSAL or not _is_count(min_cell):
        return refusal
    if isinstance(level, str) and level in LEVEL_STATISTICS and statistic in LEVEL_STATISTICS[level]:  # type: ignore[index]
        return None
    return refusal
