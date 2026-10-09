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


_FRAME_CALLS = ("pl.DataFrame(", "_frame(")


def _line_starts_frame_call(line: str) -> bool:
    """True when the line constructs a frame directly or through the local alias."""
    return any(call in line for call in _FRAME_CALLS)


def find_duplicate_dataframe_creation(content: str, filepath: Path) -> list[tuple[int, str]]:
    """Find duplicate frame-construction patterns in the same file.

    `_frame({...})` counts as a constructor. A wrapper whose only job is to
    call `pl.DataFrame` must not hide repeated column sets. Text inside
    triple-quoted docstrings is not a constructor.
    """
    violations = []
    lines = content.split("\n")

    dataframe_creations = []
    in_docstring = False
    in_frame_helper = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("def _frame("):
            in_frame_helper = True
        elif in_frame_helper and line and not line.startswith((" ", "\t")):
            in_frame_helper = False

        quote_count = line.count('"""')
        constructor = _line_starts_frame_call(line) and not in_docstring and not in_frame_helper
        if quote_count % 2 == 1:
            in_docstring = not in_docstring

        if constructor:
            df_lines = [line]
            j = i + 1
            brace_count = line.count("{") - line.count("}")
            while j < len(lines) and j < i + 10 and brace_count > 0:
                df_lines.append(lines[j])
                brace_count += lines[j].count("{") - lines[j].count("}")
                j += 1

            context = "\n".join(df_lines)
            cols = re.findall(r'"([^"]+)":', context)
            if cols:
                dataframe_creations.append((i + 1, tuple(sorted(cols))))
        i += 1

    # If same DataFrame pattern appears 2+ times, flag it
    if len(dataframe_creations) >= 2:
        patterns = {}
        for line_num, cols in dataframe_creations:
            if cols not in patterns:
                patterns[cols] = []
            patterns[cols].append(line_num)

        # Flag if same pattern appears 2+ times
        for cols, line_nums in patterns.items():
            if len(line_nums) >= 2:
                violations.append(
                    (
                        line_nums[0],
                        f"Lines {line_nums}: Duplicate DataFrame creation pattern - extract to fixture",
                    )
                )

    return violations


def check_file(filepath: Path) -> list[tuple[int, str]]:
    """Check a single test file for violations."""
    violations = []

    try:
        content = filepath.read_text()

        # Skip if file doesn't contain test functions
        if "def test_" not in content:
            return violations

        # Check for inline storage creation
        violations.extend(find_inline_storage_creation(content, filepath))

        # Check for duplicate DataFrame creation
        violations.extend(find_duplicate_dataframe_creation(content, filepath))

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
