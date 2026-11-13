# SPDX-License-Identifier: MIT
"""Per-layer index of the values stored in each column, built once and reused by every question.

The cohort peel (cohort_constraints) and the word-accounting gate (word_accounting) match the
words of a question against the values each column stores. Done naively that is a Python loop
over every distinct value of every column, patient identifiers included, on every question:
seconds per question at 100k rows. This module keeps, per column, the distinct values together
with everything derived from them (value words, yes/no labels, numeric range, a word index), so
the per-question work is a few dictionary lookups.

Two rules keep this from ever changing an answer:

- Every lookup is exact. An index only narrows which stored values are tried; each candidate
  is then checked by the same function as before, and a value the index leaves out is one that
  function would have rejected.
- Stored values that a question can never name are not matched as words: identifier columns
  (the patient id, or a text column with a different value in every row) and numeric columns
  with more values than a grouping has. Only their stored values are left out; a header
  codebook ("Smoker 1: Yes 2: No") is still read. A question word that would have named such a
  value stays unbound, so the plan is held: never a wrong answer.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar

from clinical_analytics.core.nl_query_config import STEM_MATCH_MIN_PREFIX

T = TypeVar("T")


class ColumnValues(list[Any]):
    """The distinct values of one column, with values derived from them cached alongside.

    A plain list to every caller. ``match_stored`` is False for a column whose stored values
    are never matched as words (identifiers, many-valued measures).
    """

    def __init__(self, column: str, values: Iterable[Any], *, match_stored: bool = True) -> None:
        super().__init__(values)
        self.column = column
        self.match_stored = match_stored
        self._derived: dict[str, Any] = {}

    def derived(self, key: str, compute: Callable[[], T]) -> T:
        if key not in self._derived:
            self._derived[key] = compute()
        return self._derived[key]  # type: ignore[no-any-return]


def cached(values: Sequence[Any], column: str, key: str, compute: Callable[[], T]) -> T:
    """``compute()``, cached on ``values`` when they are a column's indexed distinct values.

    The cache is used only when ``values`` is that column's own ColumnValues; any other list
    (a filtered copy, a test double, one plan value) is computed afresh, exactly as before.
    """
    if isinstance(values, ColumnValues) and values.column == column:
        return values.derived(key, compute)
    return compute()


def matches_stored(values: Sequence[Any]) -> bool:
    """Whether the stored values of this list may be matched as question words."""
    return not isinstance(values, ColumnValues) or values.match_stored


class StemIndex:
    """The words of a set that a question word stem-matches, without trying every word.

    ``stem_match(word, target)`` is true only when the two are equal, equal without trailing
    "s", or share a prefix of at least five letters. Candidates are looked up by those keys and
    each one is confirmed with ``stem_match`` itself, so the result is exactly
    ``{target for target in words if stem_match(word, target)}``.
    """

    def __init__(self, words: Iterable[str], stem_match: Callable[[str, str], bool]) -> None:
        self._stem_match = stem_match
        self._by_bare: dict[str, list[str]] = {}
        self._by_prefix: dict[str, list[str]] = {}
        for target in set(words):
            self._by_bare.setdefault(target.rstrip("s"), []).append(target)
            if len(target) >= STEM_MATCH_MIN_PREFIX:
                self._by_prefix.setdefault(target[:STEM_MATCH_MIN_PREFIX], []).append(target)
        self._sorted = sorted({target for group in self._by_bare.values() for target in group})
        self._memo: dict[str, frozenset[str]] = {}

    def matches(self, word: str) -> frozenset[str]:
        found = self._memo.get(word)
        if found is None:
            candidates = set(self._by_bare.get(word.rstrip("s"), ()))
            if len(word) >= STEM_MATCH_MIN_PREFIX:
                candidates.update(self._by_prefix.get(word[:STEM_MATCH_MIN_PREFIX], ()))
            found = frozenset(target for target in candidates if self._stem_match(word, target))
            self._memo[word] = found
        return found

    def any_match(self, word: str) -> bool:
        return bool(self.matches(word))

    def starting_with(self, prefix: str) -> list[str]:
        """Every word that starts with ``prefix``."""
        start = bisect.bisect_left(self._sorted, prefix)
        found: list[str] = []
        for target in self._sorted[start:]:
            if not target.startswith(prefix):
                break
            found.append(target)
        return found


class IndexedWords(frozenset[str]):
    """A frozen set of value words that answers "does this word name one of them?" by index."""

    _index: StemIndex | None
    _stem: Callable[[str, str], bool]

    def __new__(cls, words: Iterable[str], stem_match: Callable[[str, str], bool]) -> IndexedWords:
        made = super().__new__(cls, words)
        made._stem = stem_match
        made._index = None
        return made

    def __init__(self, words: Iterable[str], stem_match: Callable[[str, str], bool]) -> None:
        super().__init__()

    def named_by(self, word: str) -> bool:
        if self._index is None:
            self._index = StemIndex(self, self._stem)
        return self._index.any_match(word)
