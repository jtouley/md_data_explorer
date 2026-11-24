"""Tests for scripts/mutation_changed.py (make mutation-changed scope and survivor summary)."""

import io
import json
import subprocess
from collections.abc import Callable
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
    assert text.splitlines()[0] == "mutation-changed: SKIP: no mutants in the changed functions"


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
    result = mc.generated_mutants(tmp_path, f"{MOD}.*")

    # Assert
    assert result == expected


@pytest.mark.parametrize("meta", ["{not json", "[]", '{"durations_by_key": {}}'], ids=["bad_json", "list", "no_key"])
def test_generated_mutants_malformedMeta_raisesValueError(meta: str, tmp_path: Path) -> None:
    # Arrange
    _write_meta(tmp_path, MOD, meta)

    # Act / Assert
    with pytest.raises(ValueError):
        mc.generated_mutants(tmp_path, f"{MOD}.*")


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


@pytest.mark.parametrize(
    ("flags", "expected"),
    [([], [f"{MOD}.x_f__mutmut_*"]), (["--whole-modules"], [f"{MOD}.*"])],
    ids=["functions", "whole_modules"],
)
def test_main_patterns_changedSrcModulesOnly(
    flags: list[str],
    expected: list[str],
    make_branch_repo: BranchRepo,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
    rc = mc.main(["patterns", "--base", "base", *flags])

    # Assert
    assert rc == 0
    assert capsys.readouterr().out.split() == expected


# --- Function-level scope: diff hunks -> changed functions -> mutmut patterns ---------------------

PYPROJECT = '[tool.mutmut]\nsource_paths = ["src/clinical_analytics/"]\ndo_not_mutate = ["*/__init__.py"]\n'
SCHEMA = "src/clinical_analytics/core/schema.py"
SOURCE = """\
LIMIT = 3


def keep(x):
    return x + 1


@decorate
def decorated(x):
    return x


class Schema:
    size = 2

    def validate(self, x):
        def inner(y):
            return y * 2

        return inner(x) > LIMIT

    class Column:
        async def name(self):
            return "c"


def last(x):
    return x - 1
"""
HEAD_A = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"


def _fc(old_path: str | None, old: set[int], new: set[int]) -> mc.FileChange:
    return mc.FileChange(old_path, frozenset(old), frozenset(new))


# (mangled name, first line, last line) of SOURCE's functions, as mutmut names them.
SPANS = [
    ("x_keep", 4, 5),
    ("x_decorated", 8, 10),
    ("xǁSchemaǁvalidate", 16, 20),
    ("xǁSchema.Columnǁname", 23, 24),
    ("x_last", 27, 28),
]


def test_function_spans_source_namesEveryFunctionAsMutmutMangles() -> None:
    # Act
    spans = mc.function_spans(SOURCE)

    # Assert
    assert spans == SPANS


def test_function_spans_syntaxError_raises() -> None:
    # Act / Assert
    with pytest.raises(SyntaxError):
        mc.function_spans("def broken(:\n")


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        ({1}, []),
        ({14}, []),
        ({5}, ["x_keep"]),
        ({8}, ["x_decorated"]),
        ({18}, ["xǁSchemaǁvalidate"]),
        ({24}, ["xǁSchema.Columnǁname"]),
        ({3, 6, 7, 11, 12, 21, 25, 26}, []),
        ({5, 28}, ["x_keep", "x_last"]),
        ({28, 5}, ["x_keep", "x_last"]),
        (set(), []),
        ({0, 999}, []),
    ],
    ids=[
        "module_constant",
        "class_attribute",
        "function_body",
        "decorator_line",
        "nested_function_counts_for_outer",
        "nested_class_method",
        "blank_lines_between",
        "two_functions",
        "order_free",
        "nothing",
        "out_of_range",
    ],
)
def test_function_patterns_changedLines_onlyFunctionsTheyTouch(lines: set[int], expected: list[str]) -> None:
    # Act
    patterns = mc.function_patterns(MOD, SOURCE, lines)

    # Assert
    assert patterns == [f"{MOD}.{name}__mutmut_*" for name in expected]


@given(a=st.sets(st.integers(0, 30)), b=st.sets(st.integers(0, 30)))
def test_function_patterns_anyLines_monotonicAndWithinSpans(a: set[int], b: set[int]) -> None:
    # Act
    small = mc.function_patterns(MOD, SOURCE, a)
    large = mc.function_patterns(MOD, SOURCE, a | b)

    # Assert: more changed lines never drop a function, and every function named holds a changed line.
    assert set(small) <= set(large)
    spans = {f"{MOD}.{name}__mutmut_*": (first, last) for name, first, last in SPANS}
    for pattern in small:
        first, last = spans[pattern]
        assert any(first <= n <= last for n in a)


def test_function_patterns_everyLine_everyFunction() -> None:
    # Act
    patterns = mc.function_patterns(MOD, SOURCE, set(range(1, SOURCE.count("\n") + 1)))

    # Assert
    assert patterns == sorted(f"{MOD}.{name}__mutmut_*" for name, _first, _last in SPANS)


@pytest.mark.parametrize(
    ("old", "new"),
    [("keep", "renamed"), ("Schema", "Contract")],
    ids=["function", "class"],
)
def test_function_patterns_renamedDefinition_patternFollowsNewName(old: str, new: str) -> None:
    # Arrange
    source = SOURCE.replace(old, new)

    # Act
    patterns = mc.function_patterns(MOD, source, {5, 18})

    # Assert
    assert not any(old in p for p in patterns)
    assert sum(new in p for p in patterns) == 1


@pytest.mark.parametrize(
    ("diff", "expected"),
    [
        (f"{HEAD_A}@@ -3 +3 @@ def f():\n-x\n+y\n", {"a.py": _fc("a.py", {3}, {3})}),
        (f"{HEAD_A}@@ -3,0 +4,2 @@\n+x\n+y\n", {"a.py": _fc("a.py", set(), {4, 5})}),
        (f"{HEAD_A}@@ -3,2 +2,0 @@\n-x\n-y\n", {"a.py": _fc("a.py", {3, 4}, set())}),
        (
            f"{HEAD_A}@@ -1 +1 @@\n-x\n+y\n@@ -7,0 +8 @@\n+w\n"
            "diff --git a/b.py b/b.py\n--- a/b.py\n+++ b/b.py\n@@ -9,0 +10 @@\n+z\n",
            {"a.py": _fc("a.py", {1}, {1, 8}), "b.py": _fc("b.py", set(), {10})},
        ),
        ("diff --git a/a.py b/a.py\n--- a/a.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-x\n-y\n", {}),
        (
            "diff --git a/n.py b/n.py\n--- /dev/null\n+++ b/n.py\n@@ -0,0 +1,2 @@\n+x\n+y\n",
            {"n.py": _fc(None, set(), {1, 2})},
        ),
        ("diff --git a/a.py b/c.py\nrename from a.py\nrename to c.py\n", {}),
        ("diff --git a/a.py b/c.py\n--- a/a.py\n+++ b/c.py\n@@ -2 +2 @@\n-x\n+y\n", {"c.py": _fc("a.py", {2}, {2})}),
        (f"{HEAD_A}@@ -1 +1 @@\n-@@ -1 +9 @@\n+@@ -2 +8 @@\n", {"a.py": _fc("a.py", {1}, {1})}),
        (f"{HEAD_A}@@ -1,0 +2,2 @@\n+++ b/evil.py\n+@@ -1 +7 @@\n", {"a.py": _fc("a.py", set(), {2, 3})}),
        ("", {}),
    ],
    ids=[
        "edit",
        "addition",
        "pure_deletion_marks_preceding_line",
        "two_files",
        "deleted_file",
        "added_file",
        "pure_rename_has_no_hunk",
        "rename_with_edit",
        "hunk_like_content_line",
        "added_line_like_file_header",
        "empty",
    ],
)
def test_changed_lines_unifiedZeroDiff_oldAndNewSideLines(diff: str, expected: dict[str, mc.FileChange]) -> None:
    # Act / Assert
    assert mc.changed_lines(diff) == expected


@pytest.mark.parametrize(
    ("pattern", "module"),
    [(f"{MOD}.*", MOD), (f"{MOD}.x_f__mutmut_*", MOD), (f"{MOD}.xǁSchema.Columnǁname__mutmut_*", MOD)],
    ids=["module", "function", "nested_class_method"],
)
def test_pattern_module_anyPattern_itsModule(pattern: str, module: str) -> None:
    # Act / Assert
    assert mc.pattern_module(pattern) == module


def test_generated_mutants_functionPattern_countsOnlyItsMutants(tmp_path: Path) -> None:
    # Arrange
    _write_meta(
        tmp_path, MOD, _meta({f"{MOD}.x_f__mutmut_1": 1, f"{MOD}.x_f__mutmut_2": 0, f"{MOD}.x_fg__mutmut_1": 1})
    )

    # Act
    counts = [mc.generated_mutants(tmp_path, f"{MOD}.{name}__mutmut_*") for name in ("x_f", "x_fg", "x_g")]

    # Assert
    assert counts == [2, 1, 0]


@pytest.mark.parametrize(
    ("results", "generated", "rc", "ok", "head"),
    [
        ({}, 0, 1, True, "mutation-changed: SKIP"),
        ({f"{MOD}.x_f__mutmut_1": "killed"}, 1, 1, False, "mutation-changed: FAIL"),
        ({}, None, 1, False, "mutation-changed: FAIL"),
        ({f"{MOD}.x_f__mutmut_1": "killed"}, 1, 0, True, "mutation-changed: PASS"),
    ],
    ids=["nothing_matched_is_skip", "run_failed_after_results", "run_never_generated", "clean_run"],
)
def test_summarise_runExitCode_failsUnlessEveryPatternIsSkip(
    results: dict[str, str], generated: int | None, rc: int, ok: bool, head: str
) -> None:
    # Act
    text, passed = mc.summarise(results, [f"{MOD}.x_f__mutmut_*"], lambda _pattern: generated, rc)

    # Assert
    assert passed is ok
    assert text.startswith(head)
    assert (f"mutmut run exited {rc}" in text) is (rc != 0 and not ok)


def test_summarise_functionPattern_reportsFunctionScope() -> None:
    # Act
    text, passed = mc.summarise({f"{MOD}.x_f__mutmut_1": "survived"}, [f"{MOD}.x_f__mutmut_*"])

    # Assert
    assert not passed
    assert f"  {MOD}.x_f: 1 mutants: 1 survived" in text


def _patterns(repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> list[str]:
    monkeypatch.chdir(repo)
    assert mc.main(["patterns", "--base", "base"]) == 0
    return capsys.readouterr().out.split()


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        (lambda s: s.replace("return x + 1", "return x + 2"), ["x_keep"]),
        (lambda s: s.replace("        return inner(x) > LIMIT\n", ""), ["xǁSchemaǁvalidate"]),
        (lambda s: s.replace("LIMIT = 3", "LIMIT = 4").replace("size = 2", "size = 5"), []),
        (lambda s: s + "\n\ndef added(x):\n    return x\n", ["x_added"]),
        (
            lambda s: s.replace('return "c"', 'return "d"').replace("return x - 1", "return x - 2"),
            ["x_last", "xǁSchema.Columnǁname"],
        ),
    ],
    ids=["edited_function", "deleted_line_in_method", "constants_only", "added_function", "two_functions"],
)
def test_main_patterns_editedModule_onlyChangedFunctions(
    edit: Callable[[str], str],
    expected: list[str],
    make_branch_repo: BranchRepo,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    repo = make_branch_repo({"pyproject.toml": PYPROJECT, SCHEMA: SOURCE}, {SCHEMA: edit(SOURCE)})

    # Act
    patterns = _patterns(repo, monkeypatch, capsys)

    # Assert
    assert patterns == [f"{MOD}.{name}__mutmut_*" for name in expected]


def test_main_patterns_newModule_everyFunction(
    make_branch_repo: BranchRepo, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    repo = make_branch_repo({"pyproject.toml": PYPROJECT}, {SCHEMA: SOURCE})

    # Act
    patterns = _patterns(repo, monkeypatch, capsys)

    # Assert
    assert patterns == sorted(f"{MOD}.{name}__mutmut_*" for name, _first, _last in SPANS)


def test_main_patterns_uncommittedEdit_counts(
    make_branch_repo: BranchRepo, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    repo = make_branch_repo({"pyproject.toml": PYPROJECT, SCHEMA: SOURCE}, {})
    (repo / SCHEMA).write_text(SOURCE.replace("return x - 1", "return x - 3"), encoding="utf-8")

    # Act
    patterns = _patterns(repo, monkeypatch, capsys)

    # Assert
    assert patterns == [f"{MOD}.x_last__mutmut_*"]


@pytest.mark.parametrize(
    "config", [["diff.noprefix", "true"], ["diff.mnemonicPrefix", "true"]], ids=["noprefix", "mnemonic"]
)
def test_main_patterns_userDiffPrefixConfig_sameResult(
    config: list[str],
    make_branch_repo: BranchRepo,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    repo = make_branch_repo({"pyproject.toml": PYPROJECT, SCHEMA: SOURCE}, {SCHEMA: SOURCE.replace("x + 1", "x + 9")})
    subprocess.run(["git", "config", *config], cwd=repo, check=True)

    # Act
    patterns = _patterns(repo, monkeypatch, capsys)

    # Assert
    assert patterns == [f"{MOD}.x_keep__mutmut_*"]


def test_main_patterns_renamedModule_patternUsesNewModule(
    make_branch_repo: BranchRepo, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    moved = "src/clinical_analytics/core/contract.py"
    repo = make_branch_repo(
        {"pyproject.toml": PYPROJECT, SCHEMA: SOURCE},
        {SCHEMA: None, moved: SOURCE.replace("return x - 1", "return x - 4")},
    )

    # Act
    patterns = _patterns(repo, monkeypatch, capsys)

    # Assert
    assert patterns == ["clinical_analytics.core.contract.x_last__mutmut_*"]


def test_main_patterns_unparsableChangedModule_exitsTwo(
    make_branch_repo: BranchRepo,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    repo = make_branch_repo({"pyproject.toml": PYPROJECT, SCHEMA: SOURCE}, {SCHEMA: "def broken(:\n"})
    monkeypatch.chdir(repo)

    # Act
    rc = mc.main(["patterns", "--base", "base"])

    # Assert
    assert rc == 2
    assert "cannot run" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("new_source", "old_lines", "expected"),
    [
        (SOURCE.replace("        return inner(x) > LIMIT\n", ""), {20}, ["xǁSchemaǁvalidate"]),
        (SOURCE.replace("def last(x):\n    return x - 1\n", ""), {27, 28}, []),
        (SOURCE.replace("LIMIT = 3\n", ""), {1}, []),
        (SOURCE, set(), []),
    ],
    ids=["last_line_of_method_removed", "function_removed_outright", "module_line_removed", "nothing_removed"],
)
def test_function_patterns_oldSideLines_namesSurvivingFunctionsTheyLeft(
    new_source: str, old_lines: set[int], expected: list[str]
) -> None:
    # Act
    patterns = mc.function_patterns(MOD, new_source, set(), SOURCE, old_lines)

    # Assert
    assert patterns == [f"{MOD}.{name}__mutmut_*" for name in expected]
