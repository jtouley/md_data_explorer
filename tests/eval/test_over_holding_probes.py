"""Over-holding regressions: answerable questions are answered with the true value, and the
cases that a laxer gate could get wrong stay held. Graded on what the app shows."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import polars as pl
import pytest

from clinical_analytics.eval.catalog import repo_root
from clinical_analytics.eval.headless import ask, classify_presented, register_frame
from tests.eval.adversarial_probes import own_frame
from tests.eval.over_holding_probes import ANSWERED, HELD, Probe, creatinine_frame


@lru_cache(maxsize=1)
def _frames() -> dict[str, pl.DataFrame]:
    frames = {
        "coh": pl.read_csv(repo_root() / "tests" / "fixtures" / "synthetic" / "cohort.csv"),
        "own": own_frame(),
        "creat": creatinine_frame(),
    }
    for name, frame in frames.items():
        register_frame(f"overhold_{name}", frame)
    return frames


def _kind(probe: Probe) -> str:
    return "stat" if probe.stat else "by" if probe.group_by else "value"


def outcome(probe: Probe) -> tuple[str, dict[str, Any], Any]:
    """(graded outcome, result, expected) for one probe, through the display path."""
    frame = _frames()[probe.dataset]
    result = ask(f"overhold_{probe.dataset}", probe.query)
    if probe.truth is None:
        graded = "wrong" if result["success"] else ("held" if result["held"] else "refused")
        return graded, result, None
    expected = probe.truth(frame)
    return classify_presented(_kind(probe), probe.stat, probe.group_by, expected, result), result, expected


def test_overHoldingProbes_set_hasEveryQuestionOnce() -> None:
    # Arrange
    keys = [(probe.dataset, probe.query) for probe in (*ANSWERED, *HELD)]

    # Act
    duplicates = {key for key in keys if keys.count(key) > 1}

    # Assert
    assert duplicates == set()


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("probe", [pytest.param(probe, id=f"{probe.dataset}:{probe.query}") for probe in ANSWERED])
def test_overHoldingProbe_answerableQuestion_answeredWithTrueValue(probe: Probe) -> None:
    # Arrange: probe above

    # Act
    graded, result, expected = outcome(probe)

    # Assert
    assert graded == "answered", (
        f"{probe.query!r}: {graded}, expected {expected!r}, "
        f"shown {(result.get('presentation') or {}).get('claim')!r}, reason {result.get('failure_reason')!r}"
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("probe", [pytest.param(probe, id=f"{probe.dataset}:{probe.query}") for probe in HELD])
def test_overHoldingProbe_dangerousQuestion_heldWithReason(probe: Probe) -> None:
    # Arrange: probe above

    # Act
    graded, result, _expected = outcome(probe)

    # Assert
    assert graded in {"held", "refused"}, f"{probe.query!r} ran: {(result.get('presentation') or {}).get('claim')!r}"
    assert result["failure_reason"] or result["issues"], "a hold must say why"
