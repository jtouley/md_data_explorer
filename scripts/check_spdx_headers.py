#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail if a first-party source file lacks an SPDX licence identifier line.

Covers tracked ``src/**/*.py`` (``# SPDX-License-Identifier: MIT``) and ``electron/src/**/*.js``
(``// SPDX-License-Identifier: MIT``), the format from https://spdx.dev/learn/handling-license-info/.
The line must sit in the file's first lines (after a shebang or encoding line, if any).

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
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final

LICENSE_ID: Final = "MIT"
TAG: Final = "SPDX-License-Identifier:"
# (glob over tracked paths, comment prefix)
RULES: Final[tuple[tuple[str, str], ...]] = (
    ("src/*.py", "#"),
    ("electron/src/*.js", "//"),
)
EXCLUDED: Final[frozenset[str]] = frozenset()
# The identifier must appear within this many lines of the top of the file.
HEAD_LINES: Final = 5


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
    return any(line.strip() == expected for line in text.splitlines()[:HEAD_LINES])


# PEP 263 encoding declaration; it and a shebang must stay in the first two lines.
ENCODING_LINE: Final = re.compile(r"^[ \t\f]*#.*?coding[:=]")


def _is_preamble(index: int, line: str) -> bool:
    return (index == 0 and line.startswith("#!")) or bool(ENCODING_LINE.match(line))


def add_header(text: str, prefix: str) -> str:
    """Return text with the identifier line inserted after any shebang/encoding preamble."""
    if has_header(text, prefix):
        return text
    lines = text.splitlines(keepends=True)
    insert_at = 0
    while insert_at < min(len(lines), 2) and _is_preamble(insert_at, lines[insert_at]):
        insert_at += 1
    # Match the file's line endings so a CRLF file does not become mixed.
    newline = "\r\n" if lines and lines[0].endswith("\r\n") else "\n"
    lines.insert(insert_at, header_line(prefix) + newline)
    return "".join(lines)


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True).stdout
    return [p for p in out.decode().split("\0") if p]


def missing(paths: list[str], read_text: Callable[[str], str]) -> list[str]:
    """Covered paths whose text lacks the identifier line."""
    bad = []
    for path in paths:
        prefix = comment_prefix(path)
        if prefix is not None and not has_header(read_text(path), prefix):
            bad.append(path)
    return bad


def _read(path: str) -> str:
    # newline="" keeps CRLF files byte-for-byte when --fix rewrites them.
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fix", action="store_true", help="insert missing SPDX lines in place")
    args = parser.parse_args(argv)

    paths = [p for p in tracked_files() if Path(p).is_file()]
    bad = missing(paths, _read)
    if args.fix:
        for path in bad:
            prefix = comment_prefix(path)
            if prefix is None:  # missing() only returns covered paths
                raise RuntimeError(f"{path} is not covered by RULES")
            _write(path, add_header(_read(path), prefix))
        print(f"Added SPDX lines to {len(bad)} file(s).")
        return 0
    if bad:
        print(f"Source files without an '{TAG} {LICENSE_ID}' line:", file=sys.stderr)
        for path in bad:
            print(f"  {path}", file=sys.stderr)
        print("Run `uv run python scripts/check_spdx_headers.py --fix` to add them.", file=sys.stderr)
        return 1
    print("All covered source files carry an SPDX licence identifier.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
