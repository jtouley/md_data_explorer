"""Cohort gap checks used by headless ask and query execution."""

from clinical_analytics.core.cohort_constraints import unapplied_constraints


class _Plan:
    def __init__(self, group_by: str | None = None) -> None:
        self.cohort_steps: list = []
        self.group_by = group_by
        self.metric = None
        self.proportion_column = None
        self.rate_column = None


def test_grouped_count_is_not_an_unbound_subject() -> None:
    """how many patients by sex applies the group column and must not fail closed."""
    gaps = unapplied_constraints("how many patients by sex?", _Plan(group_by="sex"), ["sex", "age"])
    assert gaps == []


def test_unbound_death_word_is_a_gap() -> None:
    """died is not a stored column token, so the count must not pass as applied.

    The gap names the word itself, because the refusal shown to the user lists it.
    """
    gaps = unapplied_constraints(
        "How many MIMIC patients died?",
        _Plan(),
        ["dod", "gender", "anchor_age"],
    )
    assert "died" in gaps


def test_unappliedConstraints_functionWordInCount_isNotAGap() -> None:
    """'are there' adds no limit: word accounting treats it as a function word, so must this check."""
    # Arrange
    columns = ["patient_id", "sex", "age"]

    # Act
    gaps = unapplied_constraints("how many patients are there", _Plan(), columns)

    # Assert
    assert gaps == []
