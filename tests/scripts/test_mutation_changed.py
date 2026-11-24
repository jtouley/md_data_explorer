"""Tests for scripts/mutation_changed.py (make mutation-changed scope and survivor summary)."""

import io
import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scripts import mutation_changed as mc
from tests.scripts.conftest import BranchRepo

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
def test_module_patterns_changedPaths_onlyMutableSrcModules(paths: list[str], expected: list[str]) -> None:
    # Act / Assert
    assert mc.module_patterns(paths, EXCLUDED) == expected


def test_do_not_mutate_pyproject_readsMutmutSection() -> None:
    # Arrange
    text = '[tool.mutmut]\ndo_not_mutate = ["a/*", "*/__init__.py"]\n'

    # Act / Assert
    assert mc.do_not_mutate(text) == ("a/*", "*/__init__.py")
    assert mc.do_not_mutate("[project]\nname='x'\n") == ()


def test_parse_results_mutmutOutput_mapsNameToStatus() -> None:
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
        (["killed", "check was interrupted by user"], False),
        (["killed", "skipped"], False),
        (["killed", "segfault"], False),
        (["killed", "a status a later mutmut adds"], False),
    ],
    ids=[
        "all_killed",
        "timeouts_allowed",
        "survivor",
        "no_tests",
        "not_checked",
        "interrupted",
        "skipped",
        "segfault",
        "unknown_status",
    ],
)
def test_summarise_statuses_failsOnUnkilledMutants(statuses: list[str], ok: bool) -> None:
    # Arrange
    results = {f"{MOD}.x_f__mutmut_{i}": s for i, s in enumerate(statuses)}

    # Act
    text, passed = mc.summarise(results, [f"{MOD}.*"])

    # Assert
    assert passed is ok
    assert text.startswith(f"mutation-changed: {'PASS' if ok else 'FAIL'}")
    for name, status in results.items():
        assert (f"{status}: {name}" in text) is (status != "killed")


def test_summarise_patternWithoutResults_fails() -> None:
    # Arrange
    results = {"clinical_analytics.other.x_f__mutmut_1": "survived"}

    # Act
    text, passed = mc.summarise(results, [f"{MOD}.*"])

    # Assert
    assert not passed
    assert f"{MOD}: no mutants in the results" in text
    assert "clinical_analytics.other" not in text


statuses = st.sampled_from(
    [
        "killed",
        "survived",
        "no tests",
        "timeout",
        "suspicious",
        "not checked",
        "caught by type check",
        "skipped",
        "segfault",
        "check was interrupted by user",
    ]
)


@given(results=st.lists(statuses, min_size=1, max_size=30))
def test_summarise_anyResults_passesIffNoFailingStatus(results: list[str]) -> None:
    # Arrange
    named = {f"{MOD}.x_f__mutmut_{i}": s for i, s in enumerate(results)}

    # Act
    text, passed = mc.summarise(named, [f"{MOD}.*"])

    # Assert
    assert passed == (set(results) <= mc.PASSING_STATUSES)
    assert f"{results.count('killed')}/{len(results)} killed" in text


@pytest.mark.parametrize(
    ("generated", "ok", "line"),
    [
        (0, True, f"{MOD}: SKIP: nothing to mutate"),
        (None, False, f"{MOD}: no mutants in the results"),
        (3, False, f"{MOD}: no mutants in the results"),
    ],
    ids=["nothing_to_mutate", "never_generated", "generated_but_no_results"],
)
def test_summarise_moduleWithoutResults_skipsOnlyWhenNothingWasGenerated(
    generated: int | None, ok: bool, line: str
) -> None:
    # Arrange
    results = {"clinical_analytics.other.x_f__mutmut_1": "killed"}

    # Act
    text, passed = mc.summarise(results, [f"{MOD}.*", "clinical_analytics.other.*"], lambda _module: generated)

    # Assert
    assert passed is ok
    assert line in text
    assert text.startswith(f"mutation-changed: {'PASS' if ok else 'FAIL'}: 1/1 killed")


def test_summarise_everyModuleHasNothingToMutate_headIsSkip() -> None:
    # Act
    text, passed = mc.summarise({}, [f"{MOD}.*", "clinical_analytics.config.*"], lambda _module: 0)

    # Assert
    assert passed
    assert text.splitlines()[0] == "mutation-changed: SKIP: no mutants in the changed modules"


def _write_meta(mutants_dir: Path, module: str, text: str) -> None:
    meta = mutants_dir / "src" / (module.replace(".", "/") + ".py.meta")
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(text, encoding="utf-8")


def _meta(keys: dict[str, int | None]) -> str:
    return json.dumps({"exit_code_by_key": keys, "durations_by_key": {}, "estimated_durations_by_key": {}})


@pytest.mark.parametrize(
    ("meta", "expected"),
    [(_meta({}), 0), (_meta({f"{MOD}.x_f__mutmut_1": 1, f"{MOD}.x_f__mutmut_2": None}), 2), (None, None)],
    ids=["empty_meta", "two_mutants", "no_meta"],
)
def test_generated_mutants_metaFile_countsMutantsOrNoneWhenMissing(
    meta: str | None, expected: int | None, tmp_path: Path
) -> None:
    # Arrange
    if meta is not None:
        _write_meta(tmp_path, MOD, meta)

    # Act
    result = mc.generated_mutants(tmp_path, MOD)

    # Assert
    assert result == expected


@pytest.mark.parametrize("meta", ["{not json", "[]", '{"durations_by_key": {}}'], ids=["bad_json", "list", "no_key"])
def test_generated_mutants_malformedMeta_raisesValueError(meta: str, tmp_path: Path) -> None:
    # Arrange
    _write_meta(tmp_path, MOD, meta)

    # Act / Assert
    with pytest.raises(ValueError):
        mc.generated_mutants(tmp_path, MOD)


@pytest.mark.parametrize(
    ("meta", "rc", "line"),
    [(_meta({}), 0, "mutation-changed: SKIP"), (None, 1, "mutation-changed: FAIL"), ("{bad", 2, "")],
    ids=["constants_only_module", "run_never_reached_module", "corrupt_meta"],
)
def test_main_summary_moduleWithoutResults_usesMutantsDir(
    meta: str | None,
    rc: int,
    line: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    if meta is not None:
        _write_meta(tmp_path, MOD, meta)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    # Act
    result = mc.main(["summary", f"{MOD}.*", "--mutants-dir", str(tmp_path)])

    # Assert
    out = capsys.readouterr()
    assert result == rc
    assert out.out.startswith(line)
    assert ("cannot read mutmut's meta files" in out.err) is (rc == 2)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('[tool.mutmut]\nsource_paths = ["src/clinical_analytics/"]\n', ("src/clinical_analytics/",)),
        ('[tool.mutmut]\nsource_paths = "src/pkg"\n', ("src/pkg",)),
        ("[project]\nname='x'\n", ("src/",)),
    ],
    ids=["list", "single_string", "default"],
)
def test_source_paths_pyproject_readsMutmutSourcePaths(text: str, expected: tuple[str, ...]) -> None:
    # Act / Assert
    assert mc.source_paths(text) == expected


@pytest.mark.parametrize(
    ("path", "kept"),
    [
        ("src/clinical_analytics/core/schema.py", True),
        ("src/other_pkg/x.py", False),
        ("src/clinical_analytics_extra/x.py", False),
        ("src/setup_helper.py", False),
    ],
    ids=["inside", "other_package", "name_prefix_only", "src_top_level"],
)
def test_module_patterns_sourcePaths_keepsOnlyModulesUnderThem(path: str, kept: bool) -> None:
    # Act
    result = mc.module_patterns([path], EXCLUDED, ["src/clinical_analytics/"])

    # Assert
    assert bool(result) is kept


def test_main_summary_readsStdinAndExitsOneOnSurvivor(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    monkeypatch.setattr("sys.stdin", io.StringIO(f"    {MOD}.x_f__mutmut_1: survived\n"))

    # Act
    rc = mc.main(["summary", f"{MOD}.*"])

    # Assert
    assert rc == 1
    assert f"survived: {MOD}.x_f__mutmut_1" in capsys.readouterr().out


def test_main_patterns_changedSrcModulesOnly(
    make_branch_repo: BranchRepo, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    pyproject = (
        '[tool.mutmut]\nsource_paths = ["src/clinical_analytics/"]\n'
        'do_not_mutate = ["src/clinical_analytics/ui/pages/*", "*/__init__.py"]\n'
    )
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
            "src/vendored/lib.py": "def g():\n    return 2\n",
            "tests/test_schema.py": "def test_f(): pass\n",
        },
    )
    monkeypatch.chdir(repo)

    # Act
    rc = mc.main(["patterns", "--base", "base"])

    # Assert
    assert rc == 0
    assert capsys.readouterr().out.split() == [f"{MOD}.*"]
