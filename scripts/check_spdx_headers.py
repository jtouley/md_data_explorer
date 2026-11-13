#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail if a first-party source file lacks an SPDX licence identifier line.

Covers tracked ``src/**/*.py`` (``# SPDX-License-Identifier: MIT``) and ``electron/src/**/*.js``
(``// SPDX-License-Identifier: MIT``), the format from https://spdx.dev/learn/handling-license-info/.
The line must sit in the file's first lines (after a BOM, shebang or PEP 263 encoding line, if any).
Paths are taken from the repository's top level, so the script can be run from any directory.
A covered file that is not UTF-8 is reported by path and fails the run.

Third-party material keeps its own licence and is not covered: the bundled embedding model
(fetched at build time, see THIRD_PARTY_NOTICES.md) and the OFL fonts in ``electron/src/fonts``
are not .py or .js files. Add a path to ``EXCLUDED`` if vendored code is ever copied in.

    python scripts/check_spdx_headers.py          # check (make check-spdx, part of make check-fast)
    python scripts/check_spdx_headers.py --fix    # insert the missing lines
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final, NamedTuple

if __package__:  # imported as scripts.check_spdx_headers (tests)
    from scripts.git_tracked import GitQueryError, repo_root, tracked_files
else:  # run as a file: scripts/ is sys.path[0]
    from git_tracked import GitQueryError, repo_root, tracked_files

LICENSE_ID: Final = "MIT"
TAG: Final = "SPDX-License-Identifier:"
# (glob over repo-relative tracked paths, comment prefix)
RULES: Final[tuple[tuple[str, str], ...]] = (
    ("src/*.py", "#"),
    ("electron/src/*.js", "//"),
)
EXCLUDED: Final[frozenset[str]] = frozenset()
# The identifier must appear within this many lines of the top of the file.
HEAD_LINES: Final = 5
BOM: Final = "\ufeff"
# PEP 263's regex, with a word boundary so prose ("# decoding: ...", "# recoding=...") is not a
# declaration. "encoding" and vim's "fileencoding" are allowed explicitly: \b alone would drop them.
ENCODING_LINE: Final = re.compile(r"^[ \t\f]*#.*?\b(?:file)?(?:en)?coding[:=][ \t]*([-_.a-zA-Z0-9]+)")
# A shebang must be line 1 and an encoding declaration line 1 or 2, so the identifier goes after.
PREAMBLE_LINES: Final = 2


def header_line(prefix: str) -> str:
    return f"{prefix} {TAG} {LICENSE_ID}"


def comment_prefix(path: str) -> str | None:
    """The comment prefix for a covered path, or None when the path is not covered."""
    if path in EXCLUDED:
        return None
    for pattern, prefix in RULES:
        # fnmatch's * crosses "/", so "src/*.py" matches every depth under src/.
        if fnmatch.fnmatch(path, pattern):
            return prefix
    return None


def has_header(text: str, prefix: str) -> bool:
    expected = header_line(prefix)
    return any(line.strip() == expected for line in text.removeprefix(BOM).splitlines()[:HEAD_LINES])


def _insert_index(lines: list[str]) -> int:
    """Index after the shebang (line 1) and a PEP 263 declaration (line 1 or 2), if present."""
    insert_at = 1 if lines and lines[0].startswith("#!") else 0
    for index in range(insert_at, min(len(lines), PREAMBLE_LINES)):
        if ENCODING_LINE.match(lines[index]):
            insert_at = index + 1
    return insert_at


def add_header(text: str, prefix: str) -> str:
    """Return text with the identifier line inserted after any BOM, shebang and encoding line."""
    if has_header(text, prefix):
        return text
    bom = BOM if text.startswith(BOM) else ""
    lines = text.removeprefix(BOM).splitlines(keepends=True)
    # Match the file's line endings so a CRLF file does not become mixed.
    newline = "\r\n" if lines and lines[0].endswith("\r\n") else "\n"
    lines.insert(_insert_index(lines), header_line(prefix) + newline)
    return bom + "".join(lines)


class ScanResult(NamedTuple):
    missing: list[str]  # covered paths whose text lacks the identifier line
    undecodable: list[tuple[str, str]]  # (path, reason) for covered files that are not UTF-8


def scan(paths: list[str], read_text: Callable[[str], str]) -> ScanResult:
    """Check every covered path; a file that is not UTF-8 is reported, never skipped."""
    result = ScanResult([], [])
    for path in paths:
        prefix = comment_prefix(path)
        if prefix is None:
            continue
        try:
            text = read_text(path)
        except UnicodeDecodeError as exc:
            result.undecodable.append((path, f"{exc.reason} at byte {exc.start}"))
            continue
        if not has_header(text, prefix):
            result.missing.append(path)
    return result


def _read(path: Path) -> str:
    # newline="" keeps CRLF files byte-for-byte when --fix rewrites them; a BOM stays in the text.
    with path.open(encoding="utf-8", newline="") as f:
        return f.read()


def _write(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fix", action="store_true", help="insert missing SPDX lines in place")
    args = parser.parse_args(argv)

    try:
        root = repo_root()
        paths = [p for p in tracked_files(root) if (root / p).is_file()]
    except GitQueryError as exc:
        print(f"check_spdx_headers: {exc}", file=sys.stderr)
        return 2
    result = scan(paths, lambda p: _read(root / p))

    for path, reason in result.undecodable:
        print(f"{path}: not valid UTF-8 ({reason}); save it as UTF-8, then re-run.", file=sys.stderr)
    if args.fix:
        for path in result.missing:
            prefix = comment_prefix(path)
            if prefix is None:  # scan() only returns covered paths
                raise RuntimeError(f"{path} is not covered by RULES")
            _write(root / path, add_header(_read(root / path), prefix))
        print(f"Added SPDX lines to {len(result.missing)} file(s).")
        return 1 if result.undecodable else 0
    if result.missing:
        print(f"Source files without an '{TAG} {LICENSE_ID}' line:", file=sys.stderr)
        for path in result.missing:
            print(f"  {path}", file=sys.stderr)
        print("Run `uv run python scripts/check_spdx_headers.py --fix` to add them.", file=sys.stderr)
    if result.missing or result.undecodable:
        return 1
    print("All covered source files carry an SPDX licence identifier.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
