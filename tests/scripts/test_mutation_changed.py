"""Tests for scripts/mutation_changed.py (make mutation-changed scope and survivor summary)."""

import io

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scripts import mutation_changed as mc

EXCLUDED = ("src/clinical_analytics/ui/pages/*", "*/__init__.py")
MOD = "clinical_analytics.core.schema"


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        (["src/clinical_analytics/core/schema.py"], [f"{MOD}.*"]),
        (
            ["src/clinical_analytics/b.py", "src/clinical_analytics/a.py", "src/clinical_analytics/a.py"],
            ["clinical_analytics.a.*", "clinical_analytics.b.*"],
        ),
        (["src/clinical_analytics/ui/pages/01_ask.py"], []),
        (["src/clinical_analytics/core/__init__.py"], []),
        (["tests/core/test_schema.py", "scripts/x.py", "src/clinical_analytics/data.yaml"], []),
        ([""], []),
        ([], []),
    ],
    ids=["one_module", "sorted_unique", "streamlit_page", "init", "not_src_python", "empty_name", "nothing"],
)
def test_module_patterns_changedPaths_onlyMutableSrcModules(paths, expected):
    # Act / Assert
    assert mc.module_patterns(paths, EXCLUDED) == expected


def test_do_not_mutate_pyproject_readsMutmutSection():
    # Arrange
    text = '[tool.mutmut]\ndo_not_mutate = ["a/*", "*/__init__.py"]\n'

    # Act / Assert
    assert mc.do_not_mutate(text) == ("a/*", "*/__init__.py")
    assert mc.do_not_mutate("[project]\nname='x'\n") == ()


def test_parse_results_mutmutOutput_mapsNameToStatus():
    # Arrange
    text = f"    {MOD}.x_f__mutmut_1: survived\n    {MOD}.xǁCǁm__mutmut_2: no tests\nnoise\n"

    # Act / Assert
    assert mc.parse_results(text) == {f"{MOD}.x_f__mutmut_1": "survived", f"{MOD}.xǁCǁm__mutmut_2": "no tests"}


@pytest.mark.parametrize(
    ("statuses", "ok"),
    [
        (["killed", "killed"], True),
        (["killed", "timeout", "suspicious", "caught by type check"], True),
        (["killed", "survived"], False),
        (["killed", "no tests"], False),
        (["not checked"], False),
    ],
    ids=["all_killed", "timeouts_allowed", "survivor", "no_tests", "not_checked"],
)
def test_summarise_statuses_failsOnUnkilledMutants(statuses, ok):
    # Arrange
    results = {f"{MOD}.x_f__mutmut_{i}": s for i, s in enumerate(statuses)}

    # Act
    text, passed = mc.summarise(results, [f"{MOD}.*"])

    # Assert
    assert passed is ok
    assert text.startswith(f"mutation-changed: {'PASS' if ok else 'FAIL'}")
    for name, status in results.items():
        assert (f"{status}: {name}" in text) is (status != "killed")


def test_summarise_patternWithoutResults_fails():
    # Arrange
    results = {"clinical_analytics.other.x_f__mutmut_1": "survived"}

    # Act
    text, passed = mc.summarise(results, [f"{MOD}.*"])

    # Assert
    assert not passed
    assert f"{MOD}: no mutants in the results" in text
    assert "clinical_analytics.other" not in text


statuses = st.sampled_from(["killed", "survived", "no tests", "timeout", "suspicious", "not checked"])


@given(results=st.lists(statuses, min_size=1, max_size=30))
def test_summarise_anyResults_passesIffNoFailingStatus(results):
    # Arrange
    named = {f"{MOD}.x_f__mutmut_{i}": s for i, s in enumerate(results)}

    # Act
    text, passed = mc.summarise(named, [f"{MOD}.*"])

    # Assert
    assert passed == (not set(results) & mc.FAILING_STATUSES)
    assert f"{results.count('killed')}/{len(results)} killed" in text


def test_main_summary_readsStdinAndExitsOneOnSurvivor(monkeypatch, capsys):
    # Arrange
    monkeypatch.setattr("sys.stdin", io.StringIO(f"    {MOD}.x_f__mutmut_1: survived\n"))

    # Act
    rc = mc.main(["summary", f"{MOD}.*"])

    # Assert
    assert rc == 1
    assert f"survived: {MOD}.x_f__mutmut_1" in capsys.readouterr().out


def test_main_patterns_changedSrcModulesOnly(make_branch_repo, monkeypatch, capsys):
    # Arrange
    pyproject = '[tool.mutmut]\ndo_not_mutate = ["src/clinical_analytics/ui/pages/*", "*/__init__.py"]\n'
    repo = make_branch_repo(
        {
            "pyproject.toml": pyproject,
            "src/clinical_analytics/core/old.py": "x = 1\n",
            "src/clinical_analytics/gone.py": "",
        },
        {
            "src/clinical_analytics/core/schema.py": "def f():\n    return 1\n",
            "src/clinical_analytics/ui/pages/01_ask.py": "x = 1\n",
            "src/clinical_analytics/gone.py": None,
            "tests/test_schema.py": "def test_f(): pass\n",
        },
    )
    monkeypatch.chdir(repo)

    # Act
    rc = mc.main(["patterns", "--base", "base"])

    # Assert
    assert rc == 0
    assert capsys.readouterr().out.split() == [f"{MOD}.*"]
