"""The fixture hook must see repeated frames hidden behind a local alias."""

import importlib.util
import sys
from pathlib import Path

script_path = Path(__file__).parent.parent.parent / "scripts" / "check_test_fixtures.py"
spec = importlib.util.spec_from_file_location("check_test_fixtures_alias", script_path)
check_test_fixtures = importlib.util.module_from_spec(spec)
sys.modules["check_test_fixtures_alias"] = check_test_fixtures
spec.loader.exec_module(check_test_fixtures)

find_duplicate_dataframe_creation = check_test_fixtures.find_duplicate_dataframe_creation


def _aliased_frame_source(alias: str, column: str) -> str:
    """Build two test functions that construct the same column set."""
    call = alias + f'({{"{column}": [1, 2]}})'
    return f"def test_one():\n    first = {call}\n\ndef test_two():\n    second = {call}\n"


class TestFrameAliasFixtureHook:
    """Duplicate column sets stay visible when tests wrap pl.DataFrame."""

    def test_frame_alias_duplicate_columns_are_flagged(self):
        """Two _frame calls with the same columns are a duplicate setup."""
        content = "def _frame(data):\n    return pl.DataFrame(data)\n\n" + _aliased_frame_source("_frame", "age")

        violations = find_duplicate_dataframe_creation(content, Path("test_alias.py"))

        assert any("Duplicate DataFrame" in message for _, message in violations)

    def test_frame_alias_inside_docstring_is_ignored(self):
        """Prose that mentions the alias twice is not a constructor."""
        column = "age"
        mention = f'_frame({{"{column}": [1]}})'
        content = 'def test_docs():\n    """\n    ' + mention + "\n    " + mention + '\n    """\n    return 1\n'

        violations = find_duplicate_dataframe_creation(content, Path("test_docs.py"))

        assert violations == []
