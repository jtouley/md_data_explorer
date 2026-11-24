"""Tests for scripts/check_pr_size.py (added non-test lines against the base branch)."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scripts import check_pr_size as size
from tests.scripts.conftest import BranchRepo


@pytest.mark.parametrize(
    ("lines", "approved", "expected"),
    [
        (0, False, "ok"),
        (size.WARN_LINES, False, "ok"),
        (size.WARN_LINES + 1, False, "warn"),
        (size.FAIL_LINES, False, "warn"),
        (size.FAIL_LINES + 1, False, "fail"),
        (size.FAIL_LINES + 1, True, "approved"),
        (size.WARN_LINES + 1, True, "warn"),
    ],
    ids=["empty", "at_warn", "above_warn", "at_fail", "above_fail", "above_fail_labelled", "warn_labelled"],
)
def test_verdict_boundaries_warnAndFailStrictlyAbove(lines: int, approved: bool, expected: str) -> None:
    # Act / Assert
    assert size.verdict(lines, approved=approved) == expected


@given(lines=st.integers(0, 10_000), approved=st.booleans())
def test_verdict_anyCount_failsOnlyAboveFailLimitWithoutLabel(lines: int, approved: bool) -> None:
    # Act
    result = size.verdict(lines, approved=approved)

    # Assert
    assert (result == "fail") == (lines > size.FAIL_LINES and not approved)
    assert (result == "ok") == (lines <= size.WARN_LINES)


@pytest.mark.parametrize(
    ("path", "exempt"),
    [
        ("tests/core/test_x.py", True),
        ("tests/fixtures/synthetic/cohort.csv", True),
        ("electron/tests/e2e/upload.spec.js", True),
        ("electron/src/renderer/app.test.js", True),
        ("scripts/test_helper.py", True),
        ("uv.lock", True),
        ("electron/package-lock.json", True),
        ("src/clinical_analytics/core/semantic.py", False),
        ("electron/src/renderer/app.js", False),
        ("docs/guide.md", False),
        ("src/clinical_analytics/contest.py", False),
    ],
    ids=[
        "py_test",
        "test_fixture",
        "e2e_spec",
        "vitest",
        "test_named_script",
        "uv_lock",
        "npm_lock",
        "src",
        "electron_src",
        "docs",
        "test_in_word",
    ],
)
def test_is_exempt_path_onlyTestsAndLockFiles(path: str, exempt: bool) -> None:
    # Act / Assert
    assert size.is_exempt(path) is exempt


def test_parse_numstat_binaryAndText_countsBinaryAsZero() -> None:
    # Arrange
    numstat = "10\t2\tsrc/a.py\0-\t-\tdocs/img.png\0"

    # Act
    result = size.parse_numstat(numstat)

    # Assert
    assert result == [size.FileDelta("src/a.py", 10), size.FileDelta("docs/img.png", 0)]


@given(
    deltas=st.lists(st.tuples(st.sampled_from(["src/a.py", "tests/t.py", "uv.lock", "docs/x.md"]), st.integers(0, 500)))
)
def test_counted_lines_anyDeltas_sumOfNonExemptOnly(deltas: list[tuple[str, int]]) -> None:
    # Arrange
    files = [size.FileDelta(path, added) for path, added in deltas]

    # Act
    result = size.counted_lines(files)

    # Assert
    assert result == sum(added for path, added in deltas if path in {"src/a.py", "docs/x.md"})


@pytest.mark.parametrize(
    ("approved", "github", "rc", "annotation"),
    [
        (False, False, 1, None),
        (True, False, 0, None),
        (False, True, 1, "::error title=PR size::"),
        (True, True, 0, "::warning title=PR size::"),
    ],
    ids=["local_fail", "local_approved", "ci_fail", "ci_approved"],
)
def test_main_overFailLimit_failsUnlessApproved(
    approved: bool,
    github: bool,
    rc: int,
    annotation: str | None,
    make_branch_repo: BranchRepo,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    big = "".join(f"x{i} = {i}\n" for i in range(size.FAIL_LINES + 1))
    repo = make_branch_repo({"src/a.py": ""}, {"src/a.py": big, "tests/test_a.py": big})
    monkeypatch.chdir(repo)
    monkeypatch.setenv("GITHUB_ACTIONS", "true" if github else "false")

    # Act
    result = size.main(["--base", "base", *(["--approved"] if approved else [])])

    # Assert
    out = capsys.readouterr().out
    assert result == rc
    assert f"pr-size: {size.FAIL_LINES + 1} added non-test lines" in out
    assert (annotation in out) if annotation else ("::" not in out)


def test_main_missingBaseRef_exitsTwo(
    make_branch_repo: BranchRepo, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    repo = make_branch_repo({"src/a.py": ""}, {})
    monkeypatch.chdir(repo)

    # Act
    result = size.main(["--base", "origin/nope"])

    # Assert
    assert result == 2
    assert "origin/nope" in capsys.readouterr().err
