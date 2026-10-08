"""Tests for scripts/sync_volt_cursor_skills.py (packaged skill copy)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.sync_volt_cursor_skills import copy_packaged_cursor_skills


def test_copy_packaged_cursor_skills_copies_skill_tree(tmp_path: Path) -> None:
    """Arrange: repo .cursor/skills/foo with SKILL.md and references. Act: copy. Assert: tree preserved."""
    repo_skills = tmp_path / ".cursor" / "skills"
    foo = repo_skills / "foo-skill"
    foo.mkdir(parents=True)
    (foo / "SKILL.md").write_text(
        "---\nname: foo-skill\ndescription: Test.\n---\n\n# Foo\n",
        encoding="utf-8",
    )
    ref = foo / "references"
    ref.mkdir()
    (ref / "note.md").write_text("# Ref\n", encoding="utf-8")

    dest = tmp_path / "global_skills"
    dest.mkdir()

    names = copy_packaged_cursor_skills(repo_skills, dest)

    assert names == ["foo-skill"]
    assert (dest / "foo-skill" / "SKILL.md").is_file()
    assert (dest / "foo-skill" / "references" / "note.md").read_text(encoding="utf-8") == "# Ref\n"


def test_copy_packaged_cursor_skills_skips_readme_only_dir(tmp_path: Path) -> None:
    """Directories without SKILL.md are not copied as skills."""
    repo_skills = tmp_path / ".cursor" / "skills"
    repo_skills.mkdir(parents=True)
    (repo_skills / "README.md").write_text("# Index\n", encoding="utf-8")
    stray = repo_skills / "not-a-skill"
    stray.mkdir()
    (stray / "readme.txt").write_text("x", encoding="utf-8")

    dest = tmp_path / "out"
    dest.mkdir()
    assert copy_packaged_cursor_skills(repo_skills, dest) == []


def test_copy_packaged_cursor_skills_missing_dir_returns_empty(tmp_path: Path) -> None:
    dest = tmp_path / "out"
    dest.mkdir()
    assert copy_packaged_cursor_skills(tmp_path / "nope", dest) == []
