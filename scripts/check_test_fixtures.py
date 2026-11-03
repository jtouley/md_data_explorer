#!/usr/bin/env python3
"""
Pre-commit hook to detect duplicate test setup code.

Checks for common violations:
1. Inline UserDatasetStorage creation (should use fixture)
2. Duplicate DataFrame creation patterns
3. Missing fixture usage for known patterns

Usage:
    python scripts/check_test_fixtures.py <file1> [file2] ...

Exit codes:
    0: No violations found
    1: Violations detected
"""

import re
import sys
from pathlib import Path


def find_inline_storage_creation(content: str, filepath: Path) -> list[tuple[int, str]]:
    """Find inline UserDatasetStorage creation in test functions."""
    violations = []
    lines = content.split("\n")

    for i, line in enumerate(lines, 1):
        # Check for UserDatasetStorage( in test functions
        if "UserDatasetStorage(" in line:
            # Get context (previous 15 lines to check for test function and fixture)
            context_start = max(0, i - 15)
            context = "\n".join(lines[context_start:i])

            # Skip if it's in a fixture definition
            if "@pytest.fixture" in context:
                continue

            # Check if we're in a test function
            if "def test_" in context:
                violations.append(
                    (
                        i,
                        f"Line {i}: Inline UserDatasetStorage creation - extract to fixture",
                    )
                )

    return violations


def find_duplicate_dataframe_creation(content: str, filepath: Path) -> list[tuple[int, str]]:
    """Find duplicate pl.DataFrame creation patterns copy-pasted across different tests.

    Only flags a column-set repeated across 2+ *different* test functions -- that's
    copy-pasted setup that belongs in a fixture. Multiple DataFrames with the same
    columns inside a single test (e.g. comparing two differently-ordered or
    differently-typed inputs to test an invariant) are not duplication; they're the
    test's actual input data, so are intentionally not flagged.
    """
    violations = []
    lines = content.split("\n")

    # Map each line to its enclosing test function (last `def test_...` at/before it).
    test_def_re = re.compile(r"^\s*def (test_\w+)")
    line_to_test_fn: list[str | None] = [None] * len(lines)
    current_fn: str | None = None
    for idx, line in enumerate(lines):
        match = test_def_re.match(line)
        if match:
            current_fn = match.group(1)
        line_to_test_fn[idx] = current_fn

    # Find all pl.DataFrame( calls with their column definitions
    dataframe_creations = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "pl.DataFrame(" in line:
            # Collect multi-line DataFrame definition (up to 10 lines)
            df_lines = [line]
            j = i + 1
            brace_count = line.count("{") - line.count("}")
            while j < len(lines) and j < i + 10 and brace_count > 0:
                df_lines.append(lines[j])
                brace_count += lines[j].count("{") - lines[j].count("}")
                j += 1

            context = "\n".join(df_lines)
            # Extract column names from DataFrame
            cols = re.findall(r'"([^"]+)":', context)
            if cols:
                dataframe_creations.append((i + 1, tuple(sorted(cols)), line_to_test_fn[i]))
        i += 1

    # If same DataFrame pattern appears in 2+ *different* test functions, flag it
    if len(dataframe_creations) >= 2:
        patterns: dict[tuple[str, ...], list[tuple[int, str | None]]] = {}
        for line_num, cols, owning_fn in dataframe_creations:
            patterns.setdefault(cols, []).append((line_num, owning_fn))

        for cols, occurrences in patterns.items():
            distinct_fns = {fn for _, fn in occurrences}
            if len(distinct_fns) >= 2:
                line_nums = [line_num for line_num, _ in occurrences]
                violations.append(
                    (
                        line_nums[0],
                        f"Lines {line_nums}: Duplicate DataFrame creation pattern across "
                        f"{sorted(f for f in distinct_fns if f)} - extract to fixture",
                    )
                )

    return violations


def _blank_triple_quoted_strings(content: str) -> str:
    """Replace the interior of triple-quoted string literals with spaces.

    Preserves newlines (and thus line numbers) so violation line numbers stay
    accurate. Needed because some files -- notably this checker's own test
    suite -- embed example "bad code" as string literals to test the checker
    itself; without this, those examples are mistaken for real production
    code in the file being linted.
    """

    def _blank(match: re.Match) -> str:
        return "".join(c if c == "\n" else " " for c in match.group(0))

    return re.sub(r'""".*?"""|\'\'\'.*?\'\'\'', _blank, content, flags=re.DOTALL)


def check_file(filepath: Path) -> list[tuple[int, str]]:
    """Check a single test file for violations."""
    violations = []

    try:
        content = filepath.read_text()

        # Skip if file doesn't contain test functions
        if "def test_" not in content:
            return violations

        scan_content = _blank_triple_quoted_strings(content)

        # Check for inline storage creation
        violations.extend(find_inline_storage_creation(scan_content, filepath))

        # Check for duplicate DataFrame creation
        violations.extend(find_duplicate_dataframe_creation(scan_content, filepath))

    except Exception as e:
        print(f"Error checking {filepath}: {e}", file=sys.stderr)

    return violations


def main():
    """Main entry point for pre-commit hook."""
    if len(sys.argv) < 2:
        print("Usage: check_test_fixtures.py <file1> [file2] ...")
        sys.exit(1)

    all_violations = []
    for filepath_str in sys.argv[1:]:
        filepath = Path(filepath_str)
        if filepath.suffix == ".py" and "test" in filepath.name:
            violations = check_file(filepath)
            if violations:
                all_violations.append((filepath, violations))

    if all_violations:
        print("❌ Test fixture violations detected:")
        print()
        for filepath, violations in all_violations:
            print(f"  {filepath}:")
            for line_num, message in violations:
                print(f"    Line {line_num}: {message}")
        print()
        print("💡 Fix: Extract duplicate setup code to fixtures.")
        print("   See: .cursor/rules/105-test-fixture-enforcement.mdc")
        print()
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
