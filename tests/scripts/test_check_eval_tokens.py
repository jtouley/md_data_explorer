"""Tests for scripts/check_eval_tokens.py (evaluation-token leak guard), on a synthetic hash list."""

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scripts import check_eval_tokens as guard

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


def test_load_hashes_headerAndDigests_parsesSaltAndAllDigests():
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
def test_load_hashes_malformedFile_raisesHashFileError(text, error):
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
    ],
)
def test_find_matches_line_flagsOnlyWholeProtectedTokens(line, expected, hashes):
    # Arrange
    lines = [guard.AddedLine("src/x.py", 7, line)]

    # Act
    result = guard.find_matches(lines, hashes)

    # Assert
    assert result == ([("src/x.py", 7)] if expected else [])


@given(
    token=identifiers, prefix=st.sampled_from(["", "x = ", "f(", "  # ", "- "]), quote=st.sampled_from(["", '"', "'"])
)
def test_find_matches_anyProtectedIdentifierInAnyContext_isFlagged(token, prefix, quote):
    # Arrange
    hashes = guard.load_hashes(_hash_text((token,)))
    line = f"{prefix}{quote}{token.upper()}{quote})"

    # Act
    result = guard.find_matches([guard.AddedLine("a.py", 1, line)], hashes)

    # Assert
    assert result == [("a.py", 1)]


@given(token=identifiers, text=st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=40))
def test_candidate_tokens_anyText_areLowercaseAndIncludeEmbeddedIdentifier(token, text):
    # Arrange
    line = f"{text} {token} {text}"

    # Act
    result = guard.candidate_tokens(line)

    # Assert
    assert token in result
    assert all(t == t.lower() for t in result)


@given(salt=st.text(min_size=1, max_size=10), token=identifiers)
def test_digest_anyAsciiToken_isCaseInsensitive(salt, token):
    # Act / Assert
    assert guard.digest(salt, token.upper()) == guard.digest(salt, token)


def test_added_lines_multiHunkDiff_numbersNewFileLinesAndSkipsDeletions():
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
def test_added_lines_anyHunk_numbersLinesConsecutivelyFromHunkStart(texts, start):
    # Arrange
    diff = "\n".join(["+++ b/f.txt", f"@@ -0,0 +{start},{len(texts)} @@", *(f"+{t}" for t in texts)])

    # Act
    result = list(guard.added_lines(diff))

    # Assert
    assert [line.line_no for line in result] == list(range(start, start + len(texts)))
    assert [line.text for line in result] == texts


def test_added_lines_contentLookingLikeHeaders_staysContent():
    # Arrange: a removed "-- x" and an added "++ y" render as "--- x" and "+++ y" inside the hunk
    diff = "\n".join(["--- a/f.md", "+++ b/f.md", "@@ -2 +2,2 @@", "--- x", "+++ y", "+@@ -1 +1 @@"])

    # Act
    result = list(guard.added_lines(diff))

    # Assert
    assert result == [guard.AddedLine("f.md", 2, "++ y"), guard.AddedLine("f.md", 3, "@@ -1 +1 @@")]


def _run(repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    hash_file = repo.parent / "hashes.txt"
    hash_file.write_text(_hash_text(), encoding="utf-8")
    monkeypatch.chdir(repo)
    rc = guard.main(["--base", "base", "--hashes", str(hash_file)])
    captured = capsys.readouterr()
    return rc, captured.out + captured.err


def test_main_addedLineWithToken_failsWithFileLineAndNeverPrintsToken(make_branch_repo, monkeypatch, capsys):
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


def test_main_tokenInNewFileName_isReportedAsFileName(make_branch_repo, monkeypatch, capsys):
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
        {"README.md": "quokka\n"},
        {guard.HASH_FILE: "# salt=x\nquokka\n"},
    ],
    ids=["no_change", "token_line_deleted", "clean_edit", "outside_scanned_paths", "hash_file_itself"],
)
def test_main_noProtectedTokenAdded_passes(branch, make_branch_repo, monkeypatch, capsys):
    # Arrange
    repo = make_branch_repo({"src/app.py": "quokka = 1\n"}, branch)

    # Act
    rc, output = _run(repo, monkeypatch, capsys)

    # Assert
    assert rc == 0, output
    assert "no protected token" in output


def test_main_uncommittedEdit_isChecked(make_branch_repo, monkeypatch, capsys):
    # Arrange
    repo = make_branch_repo({"src/app.py": "a = 1\n"}, {})
    (repo / "src/app.py").write_text("a = 1\nquokka()\n", encoding="utf-8")

    # Act
    rc, output = _run(repo, monkeypatch, capsys)

    # Assert
    assert rc == 1
    assert f"src/app.py:2: {guard.MATCH_MESSAGE}" in output


def test_main_missingBaseRef_exitsTwoNamingTheRef(make_branch_repo, monkeypatch, capsys):
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


def test_load_hashes_committedHashFile_hasSaltAndDigests():
    # Arrange
    text = (Path(__file__).resolve().parents[2] / guard.HASH_FILE).read_text(encoding="utf-8")

    # Act
    result = guard.load_hashes(text)

    # Assert
    assert result.salt
    assert result.digests
