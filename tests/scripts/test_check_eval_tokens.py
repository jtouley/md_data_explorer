"""Tests for scripts/check_eval_tokens.py (evaluation-token leak guard), on a synthetic hash list."""

import os
import subprocess
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scripts import check_eval_tokens as guard
from tests.scripts.conftest import GIT_IDENTITY, BranchRepo

SALT = "test-salt-1"
# Synthetic protected tokens: never the real list, which stays hashed in config/.
TOKENS = ("zebra_cohort", "held-out-set", "quokka")


def _hash_text(tokens: tuple[str, ...] = TOKENS, salt: str = SALT) -> str:
    lines = [f"# sha256(salt + lowercase identifier); salt={salt}"]
    lines += [guard.digest(salt, token) for token in tokens]
    return "\n".join(lines) + "\n"


@pytest.fixture
def hashes() -> guard.TokenHashes:
    return guard.load_hashes(_hash_text())


identifiers = st.from_regex(r"[a-z_][a-z0-9_]{0,20}", fullmatch=True)


def test_load_hashes_headerAndDigests_parsesSaltAndAllDigests() -> None:
    # Act
    result = guard.load_hashes(_hash_text())

    # Assert
    assert result.salt == SALT
    assert result.digests == {guard.digest(SALT, t) for t in TOKENS}


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("# no salt here\n" + "a" * 64 + "\n", "no 'salt=<salt>'"),
        ("# salt=s\n", "no digests"),
        ("", "no 'salt=<salt>'"),
        ("# salt=s\n" + "A" * 64 + "\n", "line 2 is not"),
        ("# salt=s\n" + "a" * 63 + "\n", "line 2 is not"),
        ("# salt=s\nquokka\n", "line 2 is not"),
    ],
    ids=["missing_salt", "no_digests", "empty", "uppercase_digest", "short_digest", "plaintext_token"],
)
def test_load_hashes_malformedFile_raisesHashFileError(text: str, error: str) -> None:
    # Act / Assert
    with pytest.raises(guard.HashFileError, match=error):
        guard.load_hashes(text)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("x = zebra_cohort + 1", True),
        ("x = Zebra_Cohort", True),
        ('name = "ZEBRA_COHORT"', True),
        ("see the held-out-set results", True),
        ("path = 'data/quokka'", True),
        ('name = "  Quokka  "', True),
        ("dataset quokka.csv", True),
        ("x = zebra_cohort_v2", False),
        ("x = zebra", False),
        ("held out set", False),
        ("", False),
        ("load data/held-out-set.parquet", True),
        ("held-out-set.csv", True),
        ("x-held-out-set", True),
        ("old-held-out-set-v2.tar.gz", True),
        ("see zebra_cohort.v2", True),
        ("held-out-settings", False),
        ("held-out", False),
    ],
    ids=[
        "identifier",
        "identifier_mixed_case",
        "string_upper",
        "hyphen_word",
        "path_segment",
        "padded_string",
        "dotted_file_name",
        "longer_identifier",
        "prefix_only",
        "spaced_words",
        "empty",
        "joined_in_file_name",
        "joined_with_extension",
        "joined_with_prefix",
        "joined_inside_longer_chain",
        "identifier_with_suffix_segment",
        "longer_last_segment",
        "partial_joined_word",
    ],
)
def test_find_matches_line_flagsOnlyWholeProtectedTokens(line: str, expected: bool, hashes: guard.TokenHashes) -> None:
    # Arrange
    lines = [guard.AddedLine("src/x.py", 7, line)]

    # Act
    result = guard.find_matches(lines, hashes)

    # Assert
    assert result == ([("src/x.py", 7)] if expected else [])


@given(
    token=identifiers, prefix=st.sampled_from(["", "x = ", "f(", "  # ", "- "]), quote=st.sampled_from(["", '"', "'"])
)
def test_find_matches_anyProtectedIdentifierInAnyContext_isFlagged(token: str, prefix: str, quote: str) -> None:
    # Arrange
    hashes = guard.load_hashes(_hash_text((token,)))
    line = f"{prefix}{quote}{token.upper()}{quote})"

    # Act
    result = guard.find_matches([guard.AddedLine("a.py", 1, line)], hashes)

    # Assert
    assert result == [("a.py", 1)]


segments = st.lists(st.from_regex(r"[a-z0-9_]{1,6}", fullmatch=True), min_size=0, max_size=3)
separators = st.sampled_from(["-", "."])


@given(token=st.lists(identifiers, min_size=2, max_size=3), sep=separators, before=segments, after=segments)
def test_find_matches_protectedJoinedWordInsideLongerChain_isFlagged(
    token: list[str], sep: str, before: list[str], after: list[str]
) -> None:
    # Arrange
    protected = sep.join(token)
    hashes = guard.load_hashes(_hash_text((protected,)))
    line = "path/" + "-".join([*before, protected.upper(), *after]) + " end"

    # Act
    result = guard.find_matches([guard.AddedLine("a.py", 1, line)], hashes)

    # Assert
    assert result == [("a.py", 1)]


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("a", ["a"]),
        ("a-b", ["a", "a-b", "b"]),
        ("a-b.c", ["a", "a-b", "a-b.c", "b", "b.c", "c"]),
    ],
    ids=["single", "two_segments", "mixed_separators"],
)
def test_joined_runs_word_everyContiguousRunWithItsSeparators(word: str, expected: list[str]) -> None:
    # Act
    result = sorted(guard.joined_runs(word))

    # Assert
    assert result == expected


@given(token=identifiers, text=st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=40))
def test_candidate_tokens_anyText_areLowercaseAndIncludeEmbeddedIdentifier(token: str, text: str) -> None:
    # Arrange
    line = f"{text} {token} {text}"

    # Act
    result = guard.candidate_tokens(line)

    # Assert
    assert token in result
    assert all(t == t.lower() for t in result)


@given(salt=st.text(min_size=1, max_size=10), token=identifiers)
def test_digest_anyAsciiToken_isCaseInsensitive(salt: str, token: str) -> None:
    # Act / Assert
    assert guard.digest(salt, token.upper()) == guard.digest(salt, token)


def test_added_lines_multiHunkDiff_numbersNewFileLinesAndSkipsDeletions() -> None:
    # Arrange
    diff = "\n".join(
        [
            "diff --git a/src/a.py b/src/a.py",
            "--- a/src/a.py",
            "+++ b/src/a.py",
            "@@ -3,0 +4,2 @@ def f():",
            "+first",
            "+second",
            "@@ -10 +12 @@",
            "-old",
            "+third",
            "diff --git a/src/gone.py b/src/gone.py",
            "--- a/src/gone.py",
            "+++ /dev/null",
            "@@ -1 +0,0 @@",
            "-removed",
        ]
    )

    # Act
    result = list(guard.added_lines(diff))

    # Assert
    assert result == [
        guard.AddedLine("src/a.py", 4, "first"),
        guard.AddedLine("src/a.py", 5, "second"),
        guard.AddedLine("src/a.py", 12, "third"),
    ]


diff_lines = st.text(alphabet=st.characters(blacklist_characters="\n", blacklist_categories=("Cs",)), max_size=15)


@given(texts=st.lists(diff_lines, min_size=1, max_size=8), start=st.integers(1, 10_000))
def test_added_lines_anyHunk_numbersLinesConsecutivelyFromHunkStart(texts: list[str], start: int) -> None:
    # Arrange
    diff = "\n".join(["+++ b/f.txt", f"@@ -0,0 +{start},{len(texts)} @@", *(f"+{t}" for t in texts)])

    # Act
    result = list(guard.added_lines(diff))

    # Assert
    assert [line.line_no for line in result] == list(range(start, start + len(texts)))
    assert [line.text for line in result] == texts


def test_added_lines_contentLookingLikeHeaders_staysContent() -> None:
    # Arrange: a removed "-- x" and an added "++ y" render as "--- x" and "+++ y" inside the hunk
    diff = "\n".join(["--- a/f.md", "+++ b/f.md", "@@ -2 +2,2 @@", "--- x", "+++ y", "+@@ -1 +1 @@"])

    # Act
    result = list(guard.added_lines(diff))

    # Assert
    assert result == [guard.AddedLine("f.md", 2, "++ y"), guard.AddedLine("f.md", 3, "@@ -1 +1 @@")]


@pytest.mark.parametrize(
    ("header", "path"),
    [
        ("+++ b/docs/my notes.md\t", "docs/my notes.md"),
        ("+++ b/docs/notes.md", "docs/notes.md"),
        ('+++ "b/docs/tab\\there.md"', "docs/tab\there.md"),
        ("+++ /dev/null", None),
    ],
    ids=["space_trailing_tab", "plain", "quoted", "deleted"],
)
def test_diff_path_header_newSidePathWithoutTrailingTab(header: str, path: str | None) -> None:
    # Act / Assert
    assert guard._diff_path(header) == path


def _run(repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    hash_file = repo.parent / "hashes.txt"
    hash_file.write_text(_hash_text(), encoding="utf-8")
    monkeypatch.chdir(repo)
    rc = guard.main(["--base", "base", "--hashes", str(hash_file)])
    captured = capsys.readouterr()
    return rc, captured.out + captured.err


def test_main_addedLineWithToken_failsWithFileLineAndNeverPrintsToken(
    make_branch_repo: BranchRepo, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    repo = make_branch_repo(
        {"src/app.py": "a = 1\n"},
        {"src/app.py": "a = 1\nb = 2\nds = 'Zebra_Cohort'\n", "docs/notes.md": "uses the held-out-set\n"},
    )

    # Act
    rc, output = _run(repo, monkeypatch, capsys)

    # Assert
    assert rc == 1
    assert f"src/app.py:3: {guard.MATCH_MESSAGE}" in output
    assert f"docs/notes.md:1: {guard.MATCH_MESSAGE}" in output
    assert "zebra" not in output.lower()
    assert "held-out" not in output.lower()


def test_main_tokenInNewFileName_isReportedAsFileName(
    make_branch_repo: BranchRepo, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    repo = make_branch_repo({"src/app.py": "a = 1\n"}, {"tests/eval/quokka.yaml": "q: 1\n"})

    # Act
    rc, output = _run(repo, monkeypatch, capsys)

    # Assert
    assert rc == 1
    assert f"tests/eval/quokka.yaml (file name): {guard.MATCH_MESSAGE}" in output


@pytest.mark.parametrize(
    "branch",
    [
        {"src/app.py": "a = 1\n"},
        {"src/app.py": None},
        {"src/app.py": "a = 2\n"},
        {"uv.lock": "quokka\n"},
        {"electron/package-lock.json": '"quokka": 1\n'},
        {guard.HASH_FILE: "# salt=x\nquokka\n"},
    ],
    ids=["no_change", "token_line_deleted", "clean_edit", "uv_lock", "package_lock", "hash_file_itself"],
)
def test_main_noProtectedTokenAdded_passes(
    branch: dict[str, str | None],
    make_branch_repo: BranchRepo,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    repo = make_branch_repo({"src/app.py": "quokka = 1\n"}, branch)

    # Act
    rc, output = _run(repo, monkeypatch, capsys)

    # Assert
    assert rc == 0, output
    assert "no protected token" in output


@pytest.mark.parametrize(
    "path", ["README.md", ".claude/CLAUDE.md", ".cursor/rules/brief.mdc", "Makefile", ".github/ci.yml"]
)
def test_main_tokenOutsideSourceTrees_isFlagged(
    path: str, make_branch_repo: BranchRepo, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    repo = make_branch_repo({"src/app.py": "a = 1\n"}, {path: "rule: avoid the quokka trap\n"})

    # Act
    rc, output = _run(repo, monkeypatch, capsys)

    # Assert
    assert rc == 1
    assert f"{path}:1: {guard.MATCH_MESSAGE}" in output


def test_main_tokenInPathWithSpace_reportsPathWithoutTab(
    make_branch_repo: BranchRepo, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    repo = make_branch_repo({"src/app.py": "a = 1\n"}, {"docs/my notes.md": "intro\nthe quokka run\n"})

    # Act
    rc, output = _run(repo, monkeypatch, capsys)

    # Assert
    assert rc == 1
    assert f"docs/my notes.md:2: {guard.MATCH_MESSAGE}" in output
    assert "\t" not in output


def test_main_symlinkReplacedByFileWithToken_isChecked(
    make_branch_repo: BranchRepo, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange: base holds a symlink; the branch turns it into a regular file (git diff status T)
    repo = make_branch_repo({"src/app.py": "a = 1\n"}, {})
    git = ["git", *GIT_IDENTITY]
    subprocess.run(["git", "checkout", "-q", "base"], cwd=repo, check=True)
    (repo / "notes.md").symlink_to("src/app.py")
    subprocess.run(["git", "add", "notes.md"], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "link"], cwd=repo, check=True)
    subprocess.run(["git", "checkout", "-q", "-B", "work"], cwd=repo, check=True)
    (repo / "notes.md").unlink()
    (repo / "notes.md").write_text("the quokka run\n", encoding="utf-8")
    subprocess.run(["git", "add", "notes.md"], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "type change"], cwd=repo, check=True)
    status = subprocess.run(
        ["git", "diff", "--name-status", "base", "work"], cwd=repo, capture_output=True, text=True, check=True
    )
    assert status.stdout.startswith("T\tnotes.md")

    # Act
    rc, output = _run(repo, monkeypatch, capsys)

    # Assert
    assert rc == 1
    assert f"notes.md:1: {guard.MATCH_MESSAGE}" in output


def test_main_uncommittedEdit_isChecked(
    make_branch_repo: BranchRepo, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    repo = make_branch_repo({"src/app.py": "a = 1\n"}, {})
    (repo / "src/app.py").write_text("a = 1\nquokka()\n", encoding="utf-8")

    # Act
    rc, output = _run(repo, monkeypatch, capsys)

    # Assert
    assert rc == 1
    assert f"src/app.py:2: {guard.MATCH_MESSAGE}" in output


def test_main_missingBaseRef_exitsTwoNamingTheRef(
    make_branch_repo: BranchRepo, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    repo = make_branch_repo({"src/app.py": "a = 1\n"}, {})
    hash_file = repo.parent / "hashes.txt"
    hash_file.write_text(_hash_text(), encoding="utf-8")
    monkeypatch.chdir(repo)

    # Act
    rc = guard.main(["--base", "origin/nope", "--hashes", str(hash_file)])

    # Assert
    assert rc == 2
    assert "origin/nope" in capsys.readouterr().err


def test_load_hashes_committedHashFile_hasSaltAndDigests() -> None:
    # Arrange
    text = (Path(__file__).resolve().parents[2] / guard.HASH_FILE).read_text(encoding="utf-8")

    # Act
    result = guard.load_hashes(text)

    # Assert
    assert result.salt
    assert result.digests


REPO_ROOT = Path(__file__).resolve().parents[2]
# The last 12 first-parent commits of main when the curated hash list was committed. Each must pass
# the real list: a list that flags ordinary merged work would fail every PR.
MAIN_REPLAY_COMMITS = (
    "af063e7529bd81cf229ad92285ee47a5f545649f",
    "c8f55e614bc9524f14298e0fec793a947401040f",
    "13443ab45abc4e6ffcf63f6a138a64d070a8a6be",
    "89e9f5259dc7fd0af4179b572dd3c6a85d8b1c5d",
    "a5d321efa1baa3b829bb267e1a1a58d2b3d21164",
    "5c309978015e7c157df708749c43e1cd60754fe9",
    "9eb009f28c71a358ef433eeacd2000b674891ea7",
    "99d6ae46a35ad5b05e78a07bf686fe05a5eae8a1",
    "ccdd5235a709552a02416caba89ece7f64a583ae",
    "121f6a20611bac98aa3834951bb47ff69d2991e9",
    "16c0bf86fe06b5a2bad5601f1a919ea4bc920998",
    "564f84d6c63b231818efcd0c7d00d5027177bfad",
)
# make check-eval-token-replay sets this (CI quality job, full history): a missing commit fails there
# instead of skipping. Shallow checkouts (tests-fast) skip.
REQUIRE_HISTORY_ENV = "EVAL_TOKEN_REPLAY_REQUIRE_HISTORY"


def _has_commit(rev: str) -> bool:
    probe = ["git", "cat-file", "-e", f"{rev}^{{commit}}"]
    return subprocess.run(probe, cwd=REPO_ROOT, capture_output=True, check=False).returncode == 0


@pytest.mark.parametrize("commit", MAIN_REPLAY_COMMITS, ids=[c[:7] for c in MAIN_REPLAY_COMMITS])
def test_find_matches_replayRecentMainCommitsWithRealList_flagsNothing(commit: str) -> None:
    # Arrange
    if not (_has_commit(commit) and _has_commit(f"{commit}^")):
        if os.environ.get(REQUIRE_HISTORY_ENV) == "1":
            pytest.fail(f"commit {commit[:7]} or its parent is not in this checkout (needs full history)")
        pytest.skip("shallow checkout: replay needs main's history (make check-eval-token-replay)")
    hashes = guard.load_hashes((REPO_ROOT / guard.HASH_FILE).read_text(encoding="utf-8"))
    lines = guard.collect_added(REPO_ROOT, f"{commit}^", commit)

    # Act
    matches = guard.find_matches(lines, hashes)

    # Assert
    assert matches == []
