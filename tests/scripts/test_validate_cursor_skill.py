"""Tests for scripts/validate_cursor_skill.py."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.validate_cursor_skill import validate_skill


def test_validate_skill_valid_minimal(tmp_path: Path) -> None:
    d = tmp_path / "s"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: A valid short description for testing.\n---\n\n# Body\n",
        encoding="utf-8",
    )
    ok, msg = validate_skill(d)
    assert ok is True
    assert msg == "ok"


def test_validate_skill_rejects_angle_brackets_in_description(tmp_path: Path) -> None:
    d = tmp_path / "s"
    d.mkdir()
    (d / "SKILL.md").write_text(
        '---\nname: bad-skill\ndescription: "Uses <tags> which are forbidden."\n---\n\n# Body\n',
        encoding="utf-8",
    )
    ok, msg = validate_skill(d)
    assert ok is False
    assert "angle brackets" in msg


def test_validate_skill_multiline_description(tmp_path: Path) -> None:
    d = tmp_path / "s"
    d.mkdir()
    fm = (
        "---\nname: multi-skill\ndescription: >\n"
        "  Line one of description.\n  Line two continues here.\n---\n\n# Body\n"
    )
    (d / "SKILL.md").write_text(fm, encoding="utf-8")
    ok, msg = validate_skill(d)
    assert ok is True
    assert msg == "ok"
