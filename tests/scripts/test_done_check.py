"""Tests for scripts/done_check.py (make done-check summary)."""

import re
from collections.abc import Sequence
from pathlib import Path

import pytest

from scripts import done_check

PYTEST_TAIL = "\x1b[32m===== 120 passed, 3 skipped in 41.20s =====\x1b[0m\n"
MUTATION_FAIL = (
    "mutation-changed: FAIL: 8/9 killed, 1 unkilled\n"
    "  clinical_analytics.core.x: 9 mutants: 8 killed, 1 survived\n"
    "    survived: clinical_analytics.core.x.x_f__mutmut_3  (mutmut show clinical_analytics.core.x.x_f__mutmut_3)\n"
)


@pytest.mark.parametrize(
    ("step", "rc", "output", "result", "detail"),
    [
        ("check-fast", 0, "lint ok\n✅ All fast checks passed!\n", "PASS", "✅ All fast checks passed!"),
        (
            "check-fast",
            2,
            "Found 2 errors.\nmake[2]: *** [Makefile:9: lint] Error 1\n"
            "make[1]: *** [Makefile:20: check-fast] Error 1\n",
            "FAIL",
            "lint: Found 2 errors.",
        ),
        (
            "check-fast",
            2,
            "lint ok\nsrc/a.py:3: matched a protected evaluation token\n1 added line(s) name an evaluation token.\n"
            "warning: The `UV_X` environment variable is deprecated\n"
            "make[2]: *** [Makefile:471: check-eval-tokens] Error 1\nmake[1]: *** [Makefile:487: check-fast] Error 1\n",
            "FAIL",
            "check-eval-tokens: 1 added line(s) name an evaluation token.",
        ),
        ("check-fast", 2, "make[1]: *** [Makefile:9: lint] Error 1\n", "FAIL", "lint failed"),
        (
            "check-fast",
            2,
            "Found 2 errors.\ngmake[2]: *** [Makefile:9: lint] Error 1\n"
            "gmake[1]: *** [Makefile:20: check-fast] Error 1\n",
            "FAIL",
            "lint: Found 2 errors.",
        ),
        (
            "check-fast",
            2,
            "Found 2 errors.\nmake-4.4: *** [Makefile:9: lint] Error 1\n",
            "FAIL",
            "lint: Found 2 errors.",
        ),
        (
            "check-fast",
            2,
            "Found 2 errors.\n/usr/local/bin/gmake[2]: *** [Makefile:9: lint] Error 1\n",
            "FAIL",
            "lint: Found 2 errors.",
        ),
        ("test-fast", 0, "collected\n" + PYTEST_TAIL, "PASS", "120 passed, 3 skipped in 41.20s"),
        ("test-fast", 2, "", "FAIL", "exit 2"),
        ("mutation-changed", 0, "mutation-changed: SKIP: no mutable src module changed\n", "SKIP", None),
        (
            "mutation-changed",
            0,
            "mutation-changed: SKIP: no mutants in the changed modules\n  m: SKIP: nothing to mutate\n",
            "SKIP",
            "mutation-changed: SKIP: no mutants in the changed modules",
        ),
        ("mutation-changed", 1, MUTATION_FAIL, "FAIL", "mutation-changed: FAIL: 8/9 killed, 1 unkilled"),
        ("pr-size", 0, "pr-size: 10 added non-test lines (warn above 1500, fail above 3000)\n", "PASS", None),
    ],
    ids=[
        "check_pass",
        "check_fail_lint",
        "check_fail_eval_tokens",
        "check_fail_silent",
        "check_fail_gmake",
        "check_fail_versioned_make",
        "check_fail_make_path",
        "tests_pass",
        "tests_no_output",
        "mutation_skip",
        "mutation_nothing_to_mutate",
        "mutation_fail",
        "size_ok",
    ],
)
def test_classify_stepOutput_resultAndKeyLine(step: str, rc: int, output: str, result: str, detail: str | None) -> None:
    # Act
    outcome = done_check.classify(step, rc, output)

    # Assert
    assert outcome.result == result
    assert outcome.detail == (detail or output.strip())


def test_classify_mutationFailure_keepsUnkilledMutantsAsNotes() -> None:
    # Act
    outcome = done_check.classify("mutation-changed", 1, "noise\n" + MUTATION_FAIL)

    # Assert
    assert outcome.notes == tuple(MUTATION_FAIL.splitlines()[1:])


def test_main_oneStepFails_runsEveryStepAndReportsOverallFail(capsys: pytest.CaptureFixture[str]) -> None:
    # Arrange
    outputs = {
        "check-fast": (0, "✅ All fast checks passed!\n"),
        "test-fast": (0, PYTEST_TAIL),
        "mutation-changed": (1, MUTATION_FAIL),
        "pr-size": (0, "pr-size: 10 added non-test lines\n"),
    }
    calls: list[str] = []

    def runner(make: str, step: str, extra: Sequence[str]) -> tuple[int, str]:
        calls.append(step)
        return outputs[step]

    # Act
    rc = done_check.main(["--base", "origin/main"], runner=runner)

    # Assert
    out = capsys.readouterr().out
    assert rc == 1
    assert calls == list(done_check.STEPS)
    assert "Overall: **FAIL**" in out
    assert "| `make mutation-changed` | FAIL | mutation-changed: FAIL: 8/9 killed, 1 unkilled |" in out
    assert "survived: clinical_analytics.core.x.x_f__mutmut_3" in out


def test_main_allPassOrSkip_exitsZero(capsys: pytest.CaptureFixture[str]) -> None:
    # Arrange
    def runner(make: str, step: str, extra: Sequence[str]) -> tuple[int, str]:
        return 0, "mutation-changed: SKIP: nothing\n" if step == "mutation-changed" else "ok\n"

    # Act
    rc = done_check.main([], runner=runner)

    # Assert
    assert rc == 0
    assert "Overall: **PASS**" in capsys.readouterr().out


def test_main_base_isPassedToEveryStep(capsys: pytest.CaptureFixture[str]) -> None:
    # Arrange
    seen: dict[str, list[str]] = {}

    def runner(make: str, step: str, extra: Sequence[str]) -> tuple[int, str]:
        seen[step] = list(extra)
        return 0, "ok\n"

    # Act
    rc = done_check.main(["--base", "origin/develop", "PYTEST_ARGS=-x"], runner=runner)

    # Assert
    assert rc == 0
    expected = [f"{var}=origin/develop" for var in done_check.BASE_VARS] + ["PYTEST_ARGS=-x"]
    assert seen == {step: expected for step in done_check.STEPS}
    assert "against `origin/develop`" in capsys.readouterr().out


@pytest.mark.parametrize("var", done_check.BASE_VARS)
def test_base_vars_eachIsAMakefileBaseVariable(var: str) -> None:
    # Arrange
    makefile = (Path(__file__).resolve().parents[2] / "Makefile").read_text(encoding="utf-8")

    # Act
    defined = re.search(rf"^{var} \?= origin/main$", makefile, re.M)
    used = re.search(rf"--base \$\({var}\)", makefile)

    # Assert
    assert defined and used
