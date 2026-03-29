#!/usr/bin/env python3
"""Sync Cursor skills to ~/.cursor/skills.

1. Generate volt-* (and volt-mdde-context) from .claude/agents/volt-*.md.
2. Copy every packaged skill from .cursor/skills/<name>/ (must contain SKILL.md)
   so repo-authored skills (e.g. ship-feature-spec-pr) stay aligned with global.
3. Copy MCP workbench reference for volt / MCP docs.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AGENTS = REPO / ".claude" / "agents"
REPO_CURSOR_SKILLS = REPO / ".cursor" / "skills"
GLOBAL_SKILLS = Path.home() / ".cursor" / "skills"
MCP_REF_SRC = REPO / ".claude" / "skills-references" / "mcp-workbench.md"


def copy_packaged_cursor_skills(repo_skills: Path, global_skills: Path) -> list[str]:
    """Copy each immediate child directory that contains SKILL.md. Returns copied names."""
    copied: list[str] = []
    if not repo_skills.is_dir():
        return copied
    for item in sorted(repo_skills.iterdir()):
        if not item.is_dir() or item.name.startswith("."):
            continue
        if not (item / "SKILL.md").is_file():
            continue
        dst = global_skills / item.name
        shutil.copytree(item, dst, dirs_exist_ok=True)
        copied.append(item.name)
    return copied


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


def main() -> None:
    context_md = (AGENTS / "_volt-repo-context.md").read_text(encoding="utf-8")
    ctx_desc = (
        "Polars-first clinical analytics stack for md_data_explorer: DuckDB, Ibis, Makefile tests, "
        "no new pandas, assert_frame_equal. Use before or with other volt-* skills."
    )
    ctx_body = "## Instructions\n\nApply these constraints when working on md_data_explorer.\n\n" + context_md
    write_skill(GLOBAL_SKILLS / "volt-mdde-context", "volt-mdde-context", ctx_desc, ctx_body)

    global_hook = (
        "## Workspace\n\n"
        "Optimized for **md_data_explorer** (clinical analytics: Polars, DuckDB, Ibis, Streamlit/Electron). "
        "In other repos, adapt or ignore repo-specific paths.\n\n"
        "### Stack defaults\n\n"
        + "\n".join(ln for ln in context_md.splitlines() if not ln.startswith("# Shared context for"))
        + "\n\n---\n\n"
    )

    for path in sorted(AGENTS.glob("volt-*.md")):
        name, desc, body = parse_agent(path.read_text(encoding="utf-8"))
        body = re.sub(
            r" Read `\.claude/agents/_volt-repo-context\.md` first, then ",
            " ",
            body,
            count=1,
        )
        body = re.sub(
            r" Read `\.claude/agents/_volt-repo-context\.md` and `tests/AGENTS\.md`(?: before changing tests)?\.",
            "",
            body,
            count=1,
        )
        body = re.sub(
            r" Read `\.claude/agents/_volt-repo-context\.md` first\.",
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
        write_skill(GLOBAL_SKILLS / name, name, desc, global_hook + body)

    ref_dst_dir = GLOBAL_SKILLS / "references"
    ref_dst_dir.mkdir(parents=True, exist_ok=True)
    if MCP_REF_SRC.is_file():
        shutil.copy2(MCP_REF_SRC, ref_dst_dir / "mcp-workbench.md")

    packaged = copy_packaged_cursor_skills(REPO_CURSOR_SKILLS, GLOBAL_SKILLS)
    if packaged:
        print("Packaged skills:", ", ".join(packaged))

    print(f"Wrote skills under {GLOBAL_SKILLS}")


if __name__ == "__main__":
    main()
