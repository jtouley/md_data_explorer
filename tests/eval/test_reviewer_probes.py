"""Reviewer probes: phrasings outside the golden set must be answered correctly or held.

The 117 cohort probes (tests/eval/reviewer_probes.py) are the independent review's question
set, run on the base branch and on PR #60. Truth is computed with Polars from each question's
plain meaning. A HOLD probe has no single correct answer, so it must not succeed at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from clinical_analytics.eval.headless import actual_value, ask, values_match
from tests.eval.reviewer_probes import PROBES, Probe


def _assert_correct_or_held(probe: Probe, result: dict[str, Any], frame: pl.DataFrame) -> None:
    if not result["success"]:
        assert result["held"] or result["failure_reason"] or result["issues"], "refusal must say why"
        return
    assert probe.kind != "hold", f"must not succeed: {probe.query!r}"
    expected = probe.truth(frame) if probe.truth else None
    actual = actual_value(probe.kind, probe.stat, probe.group_by, result["payload"])
    assert values_match(expected, actual), f"{probe.query!r}: expected {expected!r}, got {actual!r}"


def test_reviewerProbes_truthTable_computesForEveryAnswerableProbe(load_synthetic_dataset) -> None:
    # Arrange
    frame = load_synthetic_dataset("cohort")

    # Act
    missing = [probe.query for probe in PROBES if probe.kind != "hold" and probe.truth(frame) is None]  # type: ignore[misc]

    # Assert
    assert missing == []
    assert len({probe.query for probe in PROBES}) == len(PROBES)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("probe", [pytest.param(probe, id=probe.query) for probe in PROBES])
def test_reviewerProbe_cohortQuestion_answeredCorrectlyOrHeld(
    probe: Probe, project_root: Path, load_synthetic_dataset
) -> None:
    # Arrange
    frame = load_synthetic_dataset("cohort")

    # Act
    result = ask("cohort", probe.query, project_root)

    # Assert
    _assert_correct_or_held(probe, result, frame)


_REGIONS = ["North", "North East", "South"]
_DIABETES = ["Type 1", "Type 2", "None"]


@pytest.fixture
def region_service(make_semantic_layer):
    """QueryService over a frame with overlapping stored values (North / North East, Type 1 / Type 2)."""
    from clinical_analytics.core.query_service import QueryService

    size = 60
    frame = pl.DataFrame(
        {
            "patient_id": [f"Q{index:03d}" for index in range(size)],
            "region": [_REGIONS[index % 3] for index in range(size)],
            "diabetes status": [_DIABETES[(index // 3) % 3] for index in range(size)],
            "age": [30 + (index * 7) % 60 for index in range(size)],
            "ldl": [80.0 + (index * 13) % 100 for index in range(size)],
        }
    )
    layer = make_semantic_layer(dataset_name="regions", data=frame)
    return QueryService(layer), frame


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        pytest.param(
            "how many patients in North East?",
            lambda f: f.filter(pl.col("region") == "North East").height,
            id="north_east",
        ),
        pytest.param(
            "how many patients in the North East region?",
            lambda f: f.filter(pl.col("region") == "North East").height,
            id="north_east_region",
        ),
        pytest.param(
            "how many patients with type 2 diabetes?",
            lambda f: f.filter(pl.col("diabetes status") == "Type 2").height,
            id="type_2",
        ),
        pytest.param("how many patients with type diabetes?", None, id="type_tie_must_hold"),
    ],
)
def test_overlappingValues_question_answeredCorrectlyOrHeld(region_service, query: str, expected: Any) -> None:
    # Arrange
    service, frame = region_service

    # Act
    result = service.ask(query, dataset_id="regions", dataset_version="probe")

    # Assert
    payload = result.result if isinstance(result.result, dict) else None
    succeeded = (
        payload is not None
        and payload.get("success") is True
        and not result.requires_confirmation
        and not any(item.get("severity") == "error" for item in result.issues)
    )
    if not succeeded:
        return
    assert expected is not None, f"must not succeed: {query!r}"
    assert values_match(expected(frame), actual_value("count", None, None, payload))
