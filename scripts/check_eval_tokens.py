#!/usr/bin/env python3
"""Fail if lines added since the base branch name a protected evaluation token.

Evaluation-dataset identifiers (sealed and held-out sets) must not leak into the repository's source, prompts, tests
or docs: a rule or prompt written around them overfits the evaluation. ``config/eval_token_hashes.txt``
holds ``sha256(salt + lowercase identifier)`` for each one, with the salt in its header, so the
list can be committed without naming the tokens.

For every line added against the merge base with ``--base`` (default ``origin/main``) under
any path except ``EXEMPT_PATHSPECS`` (the hash list itself and generated lock files), and for
every added or renamed path, this script takes the identifiers, the
hyphen/dot-joined words and the string literals, lowercases them, hashes each with the salt and
reports ``path:line`` on a match. The token itself is never printed. Uncommitted edits to tracked
files are checked too; untracked files only once they are added.

    python scripts/check_eval_tokens.py                    # make check-eval-tokens (in check-fast)
    python scripts/check_eval_tokens.py --base HEAD~3

Exit codes: 0 clean, 1 a protected token was added, 2 the check could not run (bad hash file,
missing base ref, git failure).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NamedTuple

if __package__:  # imported as scripts.check_eval_tokens (tests)
    from scripts.git_tracked import GitQueryError, diff_against_base, repo_root
else:  # run as a file: scripts/ is sys.path[0]
    from git_tracked import GitQueryError, diff_against_base, repo_root

HASH_FILE: Final = "config/eval_token_hashes.txt"
DEFAULT_BASE: Final = "origin/main"
# Every tracked path is scanned: rules, prompts and briefs live in .claude/, .cursor/, root docs and the
# Makefile as much as in src/. Only the hash list and generated lock files are left out.
EXEMPT_PATHSPECS: Final[tuple[str, ...]] = (
    f":(exclude){HASH_FILE}",
    ":(exclude)uv.lock",
    ":(exclude,glob)**/package-lock.json",
)
SCANNED_PATHS: Final[tuple[str, ...]] = (".", *EXEMPT_PATHSPECS)
MATCH_MESSAGE: Final = "matched a protected evaluation token"

SALT_RE: Final = re.compile(r"\bsalt=(\S+)")
DIGEST_RE: Final = re.compile(r"[0-9a-f]{64}")
IDENTIFIER_RE: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Words joined by "-" or "." ("held-out-set", "cohort.v2"), which IDENTIFIER_RE splits apart.
WORD_RE: Final = re.compile(r"[A-Za-z0-9_]+(?:[-.][A-Za-z0-9_]+)*")
# Splits a joined word into segments, keeping the separators: "a-b.c" -> ["a", "-", "b", ".", "c"].
JOIN_SPLIT_RE: Final = re.compile(r"([-.])")
# Single-line '...', "..." and `...` literals, honouring backslash escapes.
STRING_RE: Final = re.compile(r"""(["'`])((?:\\.|(?!\1).)*)\1""")
HUNK_RE: Final = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class HashFileError(ValueError):
    """The hash list is missing its salt, has no digests, or holds a line that is not a digest."""


def digest(salt: str, token: str) -> str:
    return hashlib.sha256((salt + token.lower()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TokenHashes:
    salt: str
    digests: frozenset[str]

    def matches(self, token: str) -> bool:
        return digest(self.salt, token) in self.digests


def load_hashes(text: str) -> TokenHashes:
    """Parse the hash list: '#' comment lines (one carries ``salt=<salt>``), then one digest per line."""
    salt: str | None = None
    digests: set[str] = set()
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            found = SALT_RE.search(line)
            if found and salt is None:
                salt = found.group(1)
            continue
        if not DIGEST_RE.fullmatch(line):
            raise HashFileError(f"line {number} is not a lowercase sha256 hex digest")
        digests.add(line)
    if salt is None:
        raise HashFileError("no 'salt=<salt>' in the header comments")
    if not digests:
        raise HashFileError("no digests")
    return TokenHashes(salt, frozenset(digests))


def joined_runs(word: str) -> Iterator[str]:
    """Every contiguous run of a joined word's segments, with their original separators.

    ``WORD_RE`` is greedy, so a protected "held-out-set" inside "data/held-out-set.parquet" or
    "old-held-out-set-v2" only comes back as the whole chain; the runs include it.
    """
    parts = JOIN_SPLIT_RE.split(word)  # segments at even indexes, separators at odd ones
    count = len(parts) // 2 + 1
    for start in range(count):
        for end in range(start + 1, count + 1):
            yield "".join(parts[2 * start : 2 * end - 1])


def candidate_tokens(text: str) -> set[str]:
    """Lowercased identifiers, joined words (and every run of their segments) and string-literal contents."""
    tokens = {m.group(0) for m in IDENTIFIER_RE.finditer(text)}
    tokens |= {run for m in WORD_RE.finditer(text) for run in joined_runs(m.group(0))}
    tokens |= {m.group(2).strip() for m in STRING_RE.finditer(text)}
    return {t.lower() for t in tokens if t}


class AddedLine(NamedTuple):
    path: str
    line_no: int  # 1-based line in the new file; 0 for the path itself
    text: str


def _diff_path(header: str) -> str | None:
    """New-side path from a '+++ ' header, or None for a deleted file."""
    # git ends the header with a tab when the path holds a space ("+++ b/my notes.md\t").
    name = header[4:].rstrip("\n").removesuffix("\t")
    if name == "/dev/null":
        return None
    if name.startswith('"') and name.endswith('"'):  # git quotes names with control chars or quotes
        name = name[1:-1].encode("latin-1", "backslashreplace").decode("unicode_escape")
    return name[2:] if name.startswith("b/") else name


def added_lines(diff_text: str) -> Iterator[AddedLine]:
    """The '+' lines of a ``git diff -U0`` with their new-file line numbers.

    Hunk headers give the old and new line counts, so a content line that happens to start with
    "++ " or "-- " (shown as "+++ " / "--- ") is never mistaken for a file header.
    """
    path: str | None = None
    line_no = old_left = new_left = 0
    for raw in diff_text.split("\n"):
        if old_left > 0 or new_left > 0:
            if raw.startswith("+"):
                if path is not None:
                    yield AddedLine(path, line_no, raw[1:])
                line_no += 1
                new_left -= 1
            elif raw.startswith("-"):
                old_left -= 1
            elif raw.startswith(" "):
                old_left, new_left, line_no = old_left - 1, new_left - 1, line_no + 1
            # "\ No newline at end of file" and anything else: not counted
        elif raw.startswith("+++ "):
            path = _diff_path(raw)
        elif hunk := HUNK_RE.match(raw):
            old_left = int(hunk.group(2) or 1)
            line_no = int(hunk.group(3))
            new_left = int(hunk.group(4) or 1)


def find_matches(lines: Iterable[AddedLine], hashes: TokenHashes) -> list[tuple[str, int]]:
    """(path, line) of every line holding a protected token, in input order."""
    return [(line.path, line.line_no) for line in lines if any(map(hashes.matches, candidate_tokens(line.text)))]


def collect_added(root: Path, base: str, head: str | None = None) -> list[AddedLine]:
    """Added lines and added/renamed paths from the merge base with ``base`` to ``head``.

    Type changes (``T``: a symlink replaced by a regular file) are scanned like modifications.

    ``head=None`` diffs against the working tree (committed and uncommitted changes).
    """
    rev = [head] if head else []
    diff = diff_against_base(root, base, ["-U0", "--diff-filter=AMRT", *rev], SCANNED_PATHS)
    names = diff_against_base(root, base, ["--name-only", "-z", "--diff-filter=AR", *rev], SCANNED_PATHS)
    return [*added_lines(diff), *(AddedLine(p, 0, p) for p in names.split("\0") if p)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default=DEFAULT_BASE, help=f"ref to diff against (default {DEFAULT_BASE})")
    parser.add_argument("--hashes", type=Path, help=f"hash list (default <repo>/{HASH_FILE})")
    args = parser.parse_args(argv)

    try:
        root = repo_root()
        hashes = load_hashes((args.hashes or root / HASH_FILE).read_text(encoding="utf-8"))
        lines = collect_added(root, args.base)
    except (GitQueryError, HashFileError, OSError) as exc:
        print(f"check_eval_tokens: cannot run: {exc}", file=sys.stderr)
        return 2

    matches = find_matches(lines, hashes)
    for path, line_no in matches:
        where = f"{path} (file name)" if line_no == 0 else f"{path}:{line_no}"
        print(f"{where}: {MATCH_MESSAGE}", file=sys.stderr)
    if matches:
        print(
            f"{len(matches)} added line(s) name an evaluation token. Remove them: evaluation identifiers "
            "never go into source, prompts, tests or docs.",
            file=sys.stderr,
        )
        return 1
    print(f"check_eval_tokens: {len(lines)} added line(s) and path(s) since {args.base}: no protected token.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
