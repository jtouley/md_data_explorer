"""Second review's 103 adversarial probes: each is answered correctly, held, or refused.

Never a success with a different value, and never a success for a question with no single
correct answer. Result extraction and comparison follow the review script, independently of
production's own result formatting.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import polars as pl
import pytest

from clinical_analytics.eval.headless import ask, register_frame
from tests.eval.adversarial_probes import own_frame, probes

_COHORT = "tests/fixtures/synthetic/cohort.csv"
_STATS = ("mean", "median", "min", "max")


@lru_cache(maxsize=1)
def _probe_set() -> list[tuple[str, str, Any]]:
    from clinical_analytics.eval.catalog import repo_root

    coh = pl.read_csv(repo_root() / _COHORT)
    own = own_frame()
    register_frame("probe_coh", coh)
    register_frame("probe_own", own)
    return probes(coh, own)


def _answer(result: dict[str, Any]) -> Any:
    payload = result.get("payload") or {}
    raw = payload.get("result")
    if raw is None:
        return None
    if isinstance(raw, int | float):
        return raw
    frame = raw if isinstance(raw, pl.DataFrame) else pl.from_pandas(raw)
    if frame.height == 0:
        return None
    if frame.width == 1 and frame.height == 1:
        return frame.item()
    for key in ("proportion", "corr"):
        if key in frame.columns:
            return frame[key][0]
    group = result.get("group_by")
    if group and group in frame.columns:
        value = next((key for key in ("rate", "mean", "count", "n") if key in frame.columns), None)
        if value:
            return {str(k): v for k, v in zip(frame[group].to_list(), frame[value].to_list(), strict=True)}
    if frame.height == 1 and any(key in frame.columns for key in _STATS):
        return {key: frame[key][0] for key in _STATS if key in frame.columns}
    return f"rows:{frame.height}"


def _matches(truth: Any, answer: Any) -> bool:
    if truth is None or answer is None:
        return False
    if isinstance(truth, dict):
        if not isinstance(answer, dict) or set(truth) != set(answer):
            return False
        return all(_matches(truth[key], answer[key]) for key in truth)
    if isinstance(answer, dict):
        return any(isinstance(value, int | float) and abs(value - truth) < 1e-6 for value in answer.values())
    try:
        return abs(float(truth) - float(answer)) < 1e-6
    except (TypeError, ValueError):
        return False


def test_adversarialProbes_set_hasEveryReviewQuestionOnce() -> None:
    # Arrange / Act
    questions = [(dataset, question) for dataset, question, _truth in _probe_set()]

    # Assert
    assert len(questions) == 103
    assert len(set(questions)) == len(questions)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("index", range(103), ids=lambda index: f"probe{index:03d}")
def test_adversarialProbe_question_answeredCorrectlyOrHeld(index: int) -> None:
    # Arrange
    dataset, question, truth = _probe_set()[index]

    # Act
    result = ask(f"probe_{dataset}", question)

    # Assert
    if not result["success"]:
        assert result["held"] or result["failure_reason"] or result["issues"], "refusal must say why"
        return
    assert truth is not None, f"must not succeed: {question!r}"
    answer = _answer(result)
    assert _matches(truth, answer), f"{question!r}: truth {truth!r}, answer {answer!r}"
