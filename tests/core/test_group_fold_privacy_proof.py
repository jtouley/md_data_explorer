"""Brute-force proof that folding small groups never pins a group of 1 to 10 patients.

Attacker model. The reader sees everything a grouped count answer shows, and knows the
algorithm that produced it:

* the claim, with the total n when it is shown ("213 total across 3 site groups.");
* every group label shown, with its count, or with its count hidden ("suppressed");
* which labels are missing (folded), and the label of the folded row: "Other (groups under 11)",
  which says each folded group is small, or "Other groups (hidden)", which claims no size.

The reader knows every group's label (the levels of the column), and that each group holds 0 to
``TOP`` patients. Nothing else.

A group with a true size of 1 to 10 is pinned when every vector of group sizes that produces
exactly the same answer gives it that same size. The test enumerates, for each true vector,
every vector consistent with the answer, and asserts that no small group is pinned.

Exhaustive part: every vector of 2 to 5 groups with sizes 0 to ``TOP`` (29.6 million vectors).
The answer is equivariant under relabelling groups (renaming the groups renames the answer the
same way: suppression hides tied cells together, and the fold rule reads only the sizes as a
multiset), so the vectors that give one answer are the relabellings of the multisets that give
an answer with the same permutation-free signature. For a hidden group, the values the reader
cannot rule out are then the values of the groups with the same status (folded, or labelled with
a hidden count) across those multisets. One rendering per multiset (377,000) covers every
vector; the equivariance itself is checked on a fixed sample of permutations.

Sampled part: vectors of 6 to 8 groups. For each, every assignment of the hidden mass (the
shown total minus the shown counts) to the hidden groups is rendered and compared.

A weakened rule (always split the fold under "under 11") must be caught, so the audit is shown
to find pins when they exist.
"""

from __future__ import annotations

import itertools
import random
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import pytest

from clinical_analytics.core import result_presentation
from clinical_analytics.core.result_presentation import SECONDARY_SUPPRESSED, present_analysis

TOP = 30  # largest group size the attacker's prior allows
SMALL = range(1, 11)  # true sizes that must never be pinned
LABELS = tuple(f"g{i}" for i in range(8))
FOLDED, NO_TABLE = "folded", "no table"
Status = Any  # a shown count (int), SECONDARY_SUPPRESSED, FOLDED or NO_TABLE
Answer = tuple[str | None, str | None, tuple[Status, ...]]


def _answer(sizes: tuple[int, ...]) -> Answer:
    """(claim, fold label, status per group) as a reader sees the grouped count answer."""
    analysis = {
        "type": "count",
        "total_count": sum(sizes),
        "grouped_by": "g",
        "group_counts": [{"g": LABELS[i], "count": n} for i, n in enumerate(sizes)],
    }
    shown = present_analysis(analysis)
    rows = (shown.get("table") or {}).get("rows") or []
    counts = {row["g"]: row["count"] for row in rows if row["g"] in LABELS}
    fold = next((row["g"] for row in rows if row["g"] not in LABELS), None)
    if not rows:
        return shown.get("claim"), None, tuple(NO_TABLE for _ in sizes)
    return shown.get("claim"), fold, tuple(counts.get(LABELS[i], FOLDED) for i in range(len(sizes)))


def _signature(answer: Answer) -> tuple[Any, ...]:
    """The answer with the labels forgotten: what every relabelling of it shares."""
    claim, fold, status = answer
    shown = tuple(sorted(s for s in status if isinstance(s, int)))
    return claim, fold, shown, status.count(SECONDARY_SUPPRESSED), status.count(FOLDED), status.count(NO_TABLE)


def _exhaustive_pins(groups: int, check_every: int = 7) -> list[tuple[tuple[int, ...], int]]:
    """(true sizes, pinned group) over every vector of ``groups`` groups of 0..TOP."""
    possible: dict[tuple[Any, ...], dict[Status, set[int]]] = defaultdict(lambda: defaultdict(set))
    rendered: list[tuple[tuple[int, ...], tuple[Any, ...], tuple[Status, ...]]] = []
    rng = random.Random(groups)
    for index, sizes in enumerate(itertools.combinations_with_replacement(range(TOP + 1), groups)):
        answer = _answer(sizes)
        if index % check_every == 0:
            order = list(range(groups))
            rng.shuffle(order)
            relabelled = _answer(tuple(sizes[i] for i in order))
            assert relabelled == (answer[0], answer[1], tuple(answer[2][i] for i in order)), sizes
        signature = _signature(answer)
        for size, status in zip(sizes, answer[2], strict=True):
            if not isinstance(status, int):
                possible[signature][status].add(size)
        rendered.append((sizes, signature, answer[2]))
    return [
        (sizes, i)
        for sizes, signature, statuses in rendered
        for i, (size, status) in enumerate(zip(sizes, statuses, strict=True))
        if not isinstance(status, int) and size in SMALL and possible[signature][status] == {size}
    ]


def _partitions(total: int, parts: int, low: int = 0) -> Iterator[tuple[int, ...]]:
    """Non-decreasing tuples of ``parts`` sizes in [low, TOP] that sum to ``total``."""
    if parts == 0:
        if total == 0:
            yield ()
        return
    for first in range(low, min(TOP, total // parts) + 1):
        for rest in _partitions(total - first, parts - 1, first):
            yield (first, *rest)


def _splits(values: tuple[int, ...], size: int) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Each distinct sub-multiset of ``values`` with ``size`` items, and what is left."""
    seen: set[tuple[int, ...]] = set()
    for chosen_at in itertools.combinations(range(len(values)), size):
        chosen = tuple(values[i] for i in chosen_at)
        if chosen not in seen:
            seen.add(chosen)
            yield chosen, tuple(v for i, v in enumerate(values) if i not in chosen_at)


def _sampled_pins(sizes: tuple[int, ...]) -> list[int]:
    """Groups of ``sizes`` pinned, by rendering every assignment of the hidden mass."""
    answer = _answer(sizes)
    labelled = [i for i, s in enumerate(answer[2]) if s == SECONDARY_SUPPRESSED]
    folded = [i for i, s in enumerate(answer[2]) if s == FOLDED]
    hidden = labelled + folded
    if not hidden or answer[2][0] == NO_TABLE:
        return []  # nothing hidden, or no group shown at all (the exhaustive part covers it)
    possible: dict[str, set[int]] = {"labelled": set(), "folded": set()}
    for values in _partitions(sum(sizes[i] for i in hidden), len(hidden)):
        for in_labelled, in_folded in _splits(values, len(labelled)):
            candidate = list(sizes)
            for i, value in zip([*labelled, *folded], [*in_labelled, *in_folded], strict=True):
                candidate[i] = value
            if _answer(tuple(candidate)) == answer:
                possible["labelled"] |= set(in_labelled)
                possible["folded"] |= set(in_folded)
    return [
        i for i in hidden if sizes[i] in SMALL and possible["labelled" if i in labelled else "folded"] == {sizes[i]}
    ]


def _sample(groups: int, count: int) -> list[tuple[int, ...]]:
    rng = random.Random(9600 + groups)
    return [tuple(rng.choice((rng.randint(0, 10), rng.randint(0, TOP))) for _ in range(groups)) for _ in range(count)]


class TestGroupFoldPins:
    @pytest.mark.slow
    @pytest.mark.parametrize("groups", [2, 3, 4, 5])
    def test_groupFold_everyVectorUpToFiveGroups_noSmallGroupPinned(self, groups: int) -> None:
        # Arrange / Act
        pins = _exhaustive_pins(groups)

        # Assert
        assert pins == [], f"{len(pins)} pinned, e.g. {pins[:5]}"

    @pytest.mark.slow
    @pytest.mark.parametrize("groups", [6, 7, 8])
    def test_groupFold_sampledVectorsUpToEightGroups_noSmallGroupPinned(self, groups: int) -> None:
        # Arrange
        vectors = _sample(groups, 40)

        # Act
        pins = {sizes: found for sizes in vectors if (found := _sampled_pins(sizes))}

        # Assert
        assert pins == {}

    @pytest.mark.parametrize("groups", [2, 3])
    def test_groupFold_weakenedRule_auditFindsPins(self, groups: int, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange: always fold small groups alone under "under 11" (the rule before the check)
        monkeypatch.setattr(result_presentation, "_split_fold_allowed", lambda *args, **kwargs: True)

        # Act
        pins = _exhaustive_pins(groups)

        # Assert: the audit sees the leak, e.g. 20 total under "under 11": each group must be 10
        assert pins
        assert all(sizes[i] in SMALL for sizes, i in pins)
        if groups == 2:
            assert ((10, 10), 0) in pins
