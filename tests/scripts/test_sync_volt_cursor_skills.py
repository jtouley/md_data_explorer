"""Tests for scripts/sync_volt_cursor_skills.py (packaged skill copy)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.sync_volt_cursor_skills as sync_skills


def _set_mtime(path: Path, t: float) -> None:
    os.utime(path, (t, t))


def _valid_skill_md(body: str = "# Body\n") -> str:
    return f"---\nname: test-skill\ndescription: Test skill for sync script tests.\n---\n\n{body}"


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

    outcome = sync_skills.copy_packaged_cursor_skills(repo_skills, dest)

    assert outcome.copied == ("foo-skill",)
    assert outcome.skipped_global_newer == ()
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
    outcome = sync_skills.copy_packaged_cursor_skills(repo_skills, dest)
    assert outcome.copied == ()
    assert outcome.skipped_global_newer == ()


def test_copy_packaged_cursor_skills_missing_dir_returns_empty(tmp_path: Path) -> None:
    dest = tmp_path / "out"
    dest.mkdir()
    outcome = sync_skills.copy_packaged_cursor_skills(tmp_path / "nope", dest)
    assert outcome.copied == ()
    assert outcome.skipped_global_newer == ()


def test_copy_packaged_skips_when_global_skill_md_newer_than_repo(tmp_path: Path) -> None:
    """Do not overwrite packaged install when ~/.cursor copy of SKILL.md is newer than repo (local edits)."""
    base = time.time()
    repo_skills = tmp_path / ".cursor" / "skills"
    pkg = repo_skills / "plan-skill"
    pkg.mkdir(parents=True)
    src_skill = pkg / "SKILL.md"
    src_skill.write_text("---\nname: x\ndescription: y\n---\n\nrepo\n", encoding="utf-8")
    _set_mtime(src_skill, base)

    dest = tmp_path / "global"
    dest.mkdir()
    dst_pkg = dest / "plan-skill"
    dst_pkg.mkdir(parents=True)
    dst_skill = dst_pkg / "SKILL.md"
    dst_skill.write_text("---\nname: x\ndescription: y\n---\n\nglobal newer\n", encoding="utf-8")
    _set_mtime(dst_skill, base + 100.0)

    outcome = sync_skills.copy_packaged_cursor_skills(repo_skills, dest, force=False)
    assert outcome.copied == ()
    assert outcome.skipped_global_newer == ("plan-skill",)
    assert dst_skill.read_text(encoding="utf-8") == "---\nname: x\ndescription: y\n---\n\nglobal newer\n"


def test_write_generated_skips_when_global_skill_newer_than_source(tmp_path: Path) -> None:
    """Generated mdde-* install: skip when global SKILL.md mtime > agent .md mtime."""
    base = time.time()
    src = tmp_path / "mdde-python.md"
    src.write_text("---\nname: mdde-python\ndescription: d\n---\n\nbody\n", encoding="utf-8")
    _set_mtime(src, base)

    out_dir = tmp_path / "global" / "mdde-python"
    out_dir.mkdir(parents=True)
    skill = out_dir / "SKILL.md"
    skill.write_text("---\nname: mdde-python\ndescription: d\n---\n\nglobal only\n", encoding="utf-8")
    _set_mtime(skill, base + 200.0)

    log: list[str] = []
    sync_skills._write_generated_skill(
        dir_path=out_dir,
        name="mdde-python",
        description="d",
        body="from gen",
        source_path=src,
        force=False,
        log_skipped=log,
    )
    assert log == ["mdde-python"]
    assert "global only" in skill.read_text(encoding="utf-8")


def test_promote_packaged_copies_to_global_and_removes_repo_dir(tmp_path: Path) -> None:
    """Promote: repo packaged tree → global, then rmtree repo package."""
    repo_skills = tmp_path / ".cursor" / "skills"
    pkg = repo_skills / "plan-to-pr"
    pkg.mkdir(parents=True)
    (pkg / "SKILL.md").write_text("skill body\n", encoding="utf-8")
    (pkg / "references").mkdir()
    (pkg / "references" / "workflow.md").write_text("# W\n", encoding="utf-8")

    global_root = tmp_path / "global_skills"
    global_root.mkdir()

    names = sync_skills.promote_packaged_repo_skills_to_global(repo_skills, global_root)

    assert names == ("plan-to-pr",)
    assert not pkg.exists()
    assert (global_root / "plan-to-pr" / "SKILL.md").read_text(encoding="utf-8") == "skill body\n"
    assert (global_root / "plan-to-pr" / "references" / "workflow.md").read_text(encoding="utf-8") == "# W\n"


def test_promote_packaged_empty_when_no_skill_dirs(tmp_path: Path) -> None:
    repo_skills = tmp_path / ".cursor" / "skills"
    repo_skills.mkdir(parents=True)
    (repo_skills / "README.md").write_text("# x\n", encoding="utf-8")
    global_root = tmp_path / "g"
    global_root.mkdir()
    assert sync_skills.promote_packaged_repo_skills_to_global(repo_skills, global_root) == ()


def test_copy_packaged_force_overwrites_when_global_newer(tmp_path: Path) -> None:
    """--force (force=True) copies from repo even if global SKILL.md is newer."""
    base = time.time()
    repo_skills = tmp_path / ".cursor" / "skills"
    pkg = repo_skills / "plan-skill"
    pkg.mkdir(parents=True)
    src_skill = pkg / "SKILL.md"
    src_skill.write_text("---\nname: x\ndescription: y\n---\n\nfrom repo\n", encoding="utf-8")
    _set_mtime(src_skill, base)

    dest = tmp_path / "global"
    dest.mkdir()
    dst_pkg = dest / "plan-skill"
    dst_pkg.mkdir(parents=True)
    dst_skill = dst_pkg / "SKILL.md"
    dst_skill.write_text("old global", encoding="utf-8")
    _set_mtime(dst_skill, base + 100.0)

    outcome = sync_skills.copy_packaged_cursor_skills(repo_skills, dest, force=True)
    assert outcome.copied == ("plan-skill",)
    assert outcome.skipped_global_newer == ()
    assert "from repo" in dst_skill.read_text(encoding="utf-8")


def test_iter_packaged_skill_dirs_maps_only_dirs_with_skill_md(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    root.mkdir()
    good = root / "good-skill"
    good.mkdir()
    (good / "SKILL.md").write_text(_valid_skill_md().replace("test-skill", "good-skill"), encoding="utf-8")
    bad = root / "no-skill-md"
    bad.mkdir()
    (bad / "readme.txt").write_text("x", encoding="utf-8")
    (root / "README.md").write_text("# i\n", encoding="utf-8")

    m = sync_skills.iter_packaged_skill_dirs(root)
    assert list(m.keys()) == ["good-skill"]
    assert m["good-skill"] == good


def test_format_packaged_skills_diff_report_identical_no_content_diff(tmp_path: Path) -> None:
    repo = tmp_path / "repo" / ".cursor" / "skills"
    glob = tmp_path / "global"
    for base in (repo, glob):
        d = base / "same-skill"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            _valid_skill_md().replace("test-skill", "same-skill"),
            encoding="utf-8",
        )
    report, content_diff = sync_skills.format_packaged_skills_diff_report(repo, glob)
    assert "same-skill" in report
    assert "identical" in report
    assert content_diff is False


def test_format_packaged_skills_diff_report_content_diff_true(tmp_path: Path) -> None:
    repo = tmp_path / "repo" / ".cursor" / "skills"
    glob = tmp_path / "global"
    text = _valid_skill_md("repo line\n").replace("test-skill", "diff-skill")
    rdir = repo / "diff-skill"
    rdir.mkdir(parents=True)
    (rdir / "SKILL.md").write_text(text, encoding="utf-8")
    gdir = glob / "diff-skill"
    gdir.mkdir(parents=True)
    (gdir / "SKILL.md").write_text(text.replace("repo line", "global line"), encoding="utf-8")

    report, content_diff = sync_skills.format_packaged_skills_diff_report(repo, glob)
    assert "DIFFERS" in report
    assert content_diff is True


def test_pull_packaged_from_global_to_repo_copies_trees(tmp_path: Path) -> None:
    global_root = tmp_path / "global"
    g = global_root / "pull-me"
    g.mkdir(parents=True)
    (g / "SKILL.md").write_text(
        _valid_skill_md().replace("test-skill", "pull-me"),
        encoding="utf-8",
    )
    ref = g / "references"
    ref.mkdir()
    (ref / "note.md").write_text("# N\n", encoding="utf-8")

    repo_skills = tmp_path / ".cursor" / "skills"
    names = sync_skills.pull_packaged_from_global_to_repo(repo_skills, global_root)

    assert names == ("pull-me",)
    dst = repo_skills / "pull-me"
    assert (dst / "SKILL.md").is_file()
    assert (dst / "references" / "note.md").read_text(encoding="utf-8") == "# N\n"


def test_main_diff_packaged_fail_on_diff_exits_0_when_mirrors_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo" / ".cursor" / "skills"
    glob = tmp_path / "global"
    for base in (repo, glob):
        d = base / "match-skill"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            _valid_skill_md().replace("test-skill", "match-skill"),
            encoding="utf-8",
        )
    monkeypatch.setattr(sync_skills, "REPO_CURSOR_SKILLS", repo)
    monkeypatch.setattr(sync_skills, "GLOBAL_SKILLS", glob)
    assert sync_skills.main(["--diff-packaged", "--fail-on-diff"]) == 0


def test_main_diff_packaged_fail_on_diff_exits_1_on_skew(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo" / ".cursor" / "skills"
    glob = tmp_path / "global"
    repo.mkdir(parents=True)
    gdir = glob / "only-global"
    gdir.mkdir(parents=True)
    (gdir / "SKILL.md").write_text(
        _valid_skill_md().replace("test-skill", "only-global"),
        encoding="utf-8",
    )
    monkeypatch.setattr(sync_skills, "REPO_CURSOR_SKILLS", repo)
    monkeypatch.setattr(sync_skills, "GLOBAL_SKILLS", glob)
    assert sync_skills.main(["--diff-packaged", "--fail-on-diff"]) == 1


def test_main_diff_packaged_fail_on_diff_exits_1_on_content_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo" / ".cursor" / "skills"
    glob = tmp_path / "global"
    name = "mismatch-skill"
    rdir = repo / name
    rdir.mkdir(parents=True)
    (rdir / "SKILL.md").write_text(
        _valid_skill_md("a\n").replace("test-skill", name),
        encoding="utf-8",
    )
    gdir = glob / name
    gdir.mkdir(parents=True)
    (gdir / "SKILL.md").write_text(
        _valid_skill_md("b\n").replace("test-skill", name),
        encoding="utf-8",
    )
    monkeypatch.setattr(sync_skills, "REPO_CURSOR_SKILLS", repo)
    monkeypatch.setattr(sync_skills, "GLOBAL_SKILLS", glob)
    assert sync_skills.main(["--diff-packaged", "--fail-on-diff"]) == 1
