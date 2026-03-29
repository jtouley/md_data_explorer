#!/usr/bin/env python3
"""Sync Cursor skills to ~/.cursor/skills.

Default run: generate mdde-* + mdde-context from ``.claude/agents/``, copy ``mcp-workbench`` ref.
Packaged folders under ``.cursor/skills/<name>/`` are **not** copied to global on that path; for those,
``~/.cursor/skills/<name>/`` is canonical. Use ``--diff-packaged`` / ``--pull-packaged-from-global`` /
``--push-packaged-to-global`` / ``--promote-packaged-to-global``, or ``make cursor-packaged-skills``
with ``CURSOR_PACKAGED_ARGS=...``.

Generated skills + mcp ref: skip overwrite when global is newer than repo source unless ``--force``.
"""

from __future__ import annotations

import argparse
import difflib
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.validate_cursor_skill import validate_skill

AGENTS = REPO / ".claude" / "agents"
REPO_CURSOR_SKILLS = REPO / ".cursor" / "skills"
GLOBAL_SKILLS = Path.home() / ".cursor" / "skills"
MCP_REF_SRC = REPO / ".claude" / "skills-references" / "mcp-workbench.md"


@dataclass(frozen=True)
class PackagedSyncOutcome:
    copied: tuple[str, ...]
    skipped_global_newer: tuple[str, ...]


def iter_packaged_skill_dirs(skills_root: Path) -> dict[str, Path]:
    """Map skill name → directory for each immediate child containing SKILL.md."""
    out: dict[str, Path] = {}
    if not skills_root.is_dir():
        return out
    for item in sorted(skills_root.iterdir()):
        if not item.is_dir() or item.name.startswith("."):
            continue
        if (item / "SKILL.md").is_file():
            out[item.name] = item
    return out


def _file_text_map(skill_dir: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for p in sorted(skill_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(skill_dir).as_posix()
        try:
            files[rel] = p.read_text(encoding="utf-8")
        except OSError:
            files[rel] = f"<unreadable: {p}>"
        except UnicodeDecodeError:
            files[rel] = f"<non-utf8 binary, {p.stat().st_size} bytes>"
    return files


def _validate_line(label: str, skill_dir: Path) -> str:
    ok, msg = validate_skill(skill_dir)
    state = "ok" if ok else "fail"
    return f"  validate_cursor_skill ({label}): {state} — {msg}"


def format_packaged_skills_diff_report(repo_skills: Path, global_skills: Path) -> tuple[str, bool]:
    """Human-readable diff + validation hints. Returns (report, has_content_differences)."""
    repo_map = iter_packaged_skill_dirs(repo_skills)
    global_map = iter_packaged_skill_dirs(global_skills)
    names = sorted(set(repo_map) | set(global_map))
    lines: list[str] = []
    any_diff = False
    lines.append("Packaged: repo .cursor/skills/<pkg>/ vs ~/.cursor/skills/ (global is canonical).")
    lines.append("")
    if not names:
        lines.append("(no packaged skills in either tree)")
        return "\n".join(lines), False

    for name in names:
        lines.append(f"## {name}")
        rdir = repo_map.get(name)
        gdir = global_map.get(name)
        if rdir is not None:
            lines.append(f"  repo:    {rdir}")
            lines.append(_validate_line("repo", rdir))
        else:
            lines.append("  repo:    (absent)")
        if gdir is not None:
            lines.append(f"  global:  {gdir}")
            lines.append(_validate_line("global", gdir))
        else:
            lines.append("  global:  (absent)")

        if rdir is None and gdir is not None:
            lines.append("  → Global-only: run --pull-packaged-from-global to mirror into repo, or ignore.")
            continue
        if rdir is not None and gdir is None:
            lines.append("  → Repo-only: --push-packaged-to-global or install under ~/.cursor/skills/.")
            any_diff = True
            continue

        assert rdir is not None and gdir is not None
        rf = _file_text_map(rdir)
        gf = _file_text_map(gdir)
        if rf == gf:
            lines.append("  content: identical")
            lines.append("")
            continue

        any_diff = True
        lines.append("  content: DIFFERS")
        all_rels = sorted(set(rf) | set(gf))
        for rel in all_rels:
            if rel not in rf:
                lines.append(f"  + only in global: {rel}")
                continue
            if rel not in gf:
                lines.append(f"  - only in repo: {rel}")
                continue
            if rf[rel] == gf[rel]:
                continue
            lines.append(f"  --- unified diff: {rel} ---")
            diff = difflib.unified_diff(
                rf[rel].splitlines(keepends=True),
                gf[rel].splitlines(keepends=True),
                fromfile=f"repo/{rel}",
                tofile=f"global/{rel}",
                n=3,
            )
            lines.extend("  " + ln.rstrip("\n") for ln in diff)
        lines.append("")

    return "\n".join(lines), any_diff


def pull_packaged_from_global_to_repo(repo_skills: Path, global_skills: Path) -> tuple[str, ...]:
    """Copy each global packaged skill into repo (global wins). Creates repo_skills if needed."""
    pulled: list[str] = []
    repo_skills.mkdir(parents=True, exist_ok=True)
    for name, gdir in sorted(iter_packaged_skill_dirs(global_skills).items()):
        dst = repo_skills / name
        shutil.copytree(gdir, dst, dirs_exist_ok=True)
        pulled.append(name)
    return tuple(pulled)


def _should_skip_dst_newer(*, dst_skill: Path, src_mtime: float, force: bool) -> bool:
    if force or not dst_skill.is_file():
        return False
    return dst_skill.stat().st_mtime > src_mtime


def copy_packaged_cursor_skills(
    repo_skills: Path,
    global_skills: Path,
    *,
    force: bool = False,
) -> PackagedSyncOutcome:
    """Copy each repo packaged skill to global (explicit push only).

    Skips a package when global ``SKILL.md`` is newer than repo (unless force).
    """
    copied: list[str] = []
    skipped: list[str] = []
    if not repo_skills.is_dir():
        return PackagedSyncOutcome((), ())
    for item in sorted(repo_skills.iterdir()):
        if not item.is_dir() or item.name.startswith("."):
            continue
        src_skill = item / "SKILL.md"
        if not src_skill.is_file():
            continue
        dst = global_skills / item.name
        dst_skill = dst / "SKILL.md"
        src_mtime = src_skill.stat().st_mtime
        if _should_skip_dst_newer(dst_skill=dst_skill, src_mtime=src_mtime, force=force):
            skipped.append(item.name)
            continue
        shutil.copytree(item, dst, dirs_exist_ok=True)
        copied.append(item.name)
    return PackagedSyncOutcome(tuple(copied), tuple(skipped))


def promote_packaged_repo_skills_to_global(repo_skills: Path, global_skills: Path) -> tuple[str, ...]:
    """Copy each repo packaged skill to global, then remove the repo directory."""
    promoted: list[str] = []
    if not repo_skills.is_dir():
        return ()
    for item in sorted(repo_skills.iterdir()):
        if not item.is_dir() or item.name.startswith("."):
            continue
        if not (item / "SKILL.md").is_file():
            continue
        dst = global_skills / item.name
        shutil.copytree(item, dst, dirs_exist_ok=True)
        shutil.rmtree(item)
        promoted.append(item.name)
    return tuple(promoted)


def parse_agent(md: str) -> tuple[str, str, str]:
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", md, re.DOTALL)
    if not m:
        raise ValueError("expected YAML frontmatter")
    fm, body = m.group(1), m.group(2).strip()
    name = ""
    desc = ""
    for line in fm.splitlines():
        if line.startswith("name:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("description:"):
            raw = line.split(":", 1)[1].strip()
            if raw.startswith('"') and raw.endswith('"'):
                raw = raw[1:-1].replace('\\"', '"')
            desc = raw
    if not name or not desc:
        raise ValueError("missing name/description in frontmatter")
    return name, desc, body


def yaml_desc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def write_skill(dir_path: Path, name: str, description: str, body: str) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    content = f'---\nname: {name}\ndescription: "{yaml_desc(description)}"\n---\n\n{body}'
    (dir_path / "SKILL.md").write_text(content, encoding="utf-8")


def _write_generated_skill(
    *,
    dir_path: Path,
    name: str,
    description: str,
    body: str,
    source_path: Path,
    force: bool,
    log_skipped: list[str],
) -> None:
    dst_skill = dir_path / "SKILL.md"
    src_mtime = source_path.stat().st_mtime
    if _should_skip_dst_newer(dst_skill=dst_skill, src_mtime=src_mtime, force=force):
        log_skipped.append(name)
        return
    write_skill(dir_path, name, description, body)


def _copy_mcp_ref(*, force: bool, log_skipped: list[str]) -> None:
    ref_dst_dir = GLOBAL_SKILLS / "references"
    ref_dst_dir.mkdir(parents=True, exist_ok=True)
    dst = ref_dst_dir / "mcp-workbench.md"
    if not MCP_REF_SRC.is_file():
        return
    src_mtime = MCP_REF_SRC.stat().st_mtime
    if _should_skip_dst_newer(dst_skill=dst, src_mtime=src_mtime, force=force):
        log_skipped.append("references/mcp-workbench.md")
        return
    shutil.copy2(MCP_REF_SRC, dst)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite global generated skills / mcp-workbench when global is newer; same for push.",
    )
    p.add_argument(
        "--promote-packaged-to-global",
        action="store_true",
        help="Copy repo packaged dirs to ~/.cursor/skills/ then delete repo copies (migration).",
    )
    p.add_argument(
        "--diff-packaged",
        action="store_true",
        help="Print repo vs global diff and validate_cursor_skill per side.",
    )
    p.add_argument(
        "--fail-on-diff",
        action="store_true",
        help="With --diff-packaged, exit 1 when any file content differs or repo-only/global-only skew.",
    )
    p.add_argument(
        "--pull-packaged-from-global",
        action="store_true",
        help="Copy every ~/.cursor/skills/<pkg>/ with SKILL.md into .cursor/skills/ (global wins).",
    )
    p.add_argument(
        "--push-packaged-to-global",
        action="store_true",
        help="Copy .cursor/skills/<pkg>/ → ~/.cursor/skills/ (use after merge; global is default SOT).",
    )
    args = p.parse_args(argv)
    force: bool = args.force

    if args.promote_packaged_to_global:
        global_skills = GLOBAL_SKILLS
        global_skills.mkdir(parents=True, exist_ok=True)
        promoted = promote_packaged_repo_skills_to_global(REPO_CURSOR_SKILLS, global_skills)
        if promoted:
            print("Promoted to global and removed from repo:", ", ".join(promoted))
        else:
            print("No packaged skills under .cursor/skills/ to promote.")
        print(f"Global skills root: {global_skills}")
        return 0

    if args.diff_packaged:
        report, content_diff = format_packaged_skills_diff_report(REPO_CURSOR_SKILLS, GLOBAL_SKILLS)
        print(report)
        repo_map = iter_packaged_skill_dirs(REPO_CURSOR_SKILLS)
        global_map = iter_packaged_skill_dirs(GLOBAL_SKILLS)
        names = set(repo_map) | set(global_map)
        skew = any((n in repo_map) != (n in global_map) for n in names)
        if args.fail_on_diff and (content_diff or skew):
            return 1
        return 0

    if args.pull_packaged_from_global:
        GLOBAL_SKILLS.mkdir(parents=True, exist_ok=True)
        pulled = pull_packaged_from_global_to_repo(REPO_CURSOR_SKILLS, GLOBAL_SKILLS)
        if pulled:
            print("Pulled from global into repo:", ", ".join(pulled))
        else:
            print("No packaged skills under ~/.cursor/skills/ to pull.")
        return 0

    if args.push_packaged_to_global:
        GLOBAL_SKILLS.mkdir(parents=True, exist_ok=True)
        packaged = copy_packaged_cursor_skills(REPO_CURSOR_SKILLS, GLOBAL_SKILLS, force=force)
        if packaged.copied:
            print("Pushed packaged to global:", ", ".join(packaged.copied))
        if packaged.skipped_global_newer:
            print(
                "Skipped push (global SKILL.md newer; use --force):",
                ", ".join(packaged.skipped_global_newer),
                file=sys.stderr,
            )
        print(f"Global skills root: {GLOBAL_SKILLS}")
        return 0

    skipped_gen: list[str] = []

    context_path = AGENTS / "_mdde-repo-context.md"
    context_md = context_path.read_text(encoding="utf-8")
    ctx_desc = (
        "Polars-first clinical analytics stack for md_data_explorer: DuckDB, Ibis, Makefile tests, "
        "no new pandas, assert_frame_equal. Use before or with other mdde-* skills."
    )
    ctx_body = "## Instructions\n\nApply these constraints when working on md_data_explorer.\n\n" + context_md
    _write_generated_skill(
        dir_path=GLOBAL_SKILLS / "mdde-context",
        name="mdde-context",
        description=ctx_desc,
        body=ctx_body,
        source_path=context_path,
        force=force,
        log_skipped=skipped_gen,
    )

    global_hook = (
        "## Workspace\n\n"
        "Optimized for **md_data_explorer** (clinical analytics: Polars, DuckDB, Ibis, Streamlit/Electron). "
        "In other repos, adapt or ignore repo-specific paths.\n\n"
        "### Stack defaults\n\n"
        + "\n".join(ln for ln in context_md.splitlines() if not ln.startswith("# Shared context for"))
        + "\n\n---\n\n"
    )

    for path in sorted(AGENTS.glob("mdde-*.md")):
        name, desc, body = parse_agent(path.read_text(encoding="utf-8"))
        body = re.sub(
            r" Read `\.claude/agents/_mdde-repo-context\.md` first, then ",
            " ",
            body,
            count=1,
        )
        body = re.sub(
            r" Read `\.claude/agents/_mdde-repo-context\.md` and `tests/AGENTS\.md`(?: before changing tests)?\.",
            "",
            body,
            count=1,
        )
        body = re.sub(
            r" Read `\.claude/agents/_mdde-repo-context\.md` first\.",
            "",
            body,
            count=1,
        )
        body = body.replace(
            "You are a senior Python engineer for this repository. `.claude/CLAUDE.md` for detail.",
            "You are a senior Python engineer for this repository. See `.claude/CLAUDE.md` for project conventions.",
        )
        body = body.replace(
            "You are an LLM systems architect for this product. inspect ",
            "You are an LLM systems architect for this product. Inspect ",
        )
        _write_generated_skill(
            dir_path=GLOBAL_SKILLS / name,
            name=name,
            description=desc,
            body=global_hook + body,
            source_path=path,
            force=force,
            log_skipped=skipped_gen,
        )

    _copy_mcp_ref(force=force, log_skipped=skipped_gen)

    if skipped_gen:
        print(
            "Skipped generated / MCP ref (global newer than repo source; use --force):",
            ", ".join(skipped_gen),
            file=sys.stderr,
        )

    print(f"Wrote mdde-* / mdde-context / mcp-workbench under {GLOBAL_SKILLS}")
    print(
        "Packaged skills: not auto-synced (global is canonical). "
        "make cursor-packaged-skills (override CURSOR_PACKAGED_ARGS for pull/push/promote).",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
