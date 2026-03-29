#!/usr/bin/env python3
"""Validate a Cursor skill folder (SKILL.md YAML frontmatter + naming). Skill-creator aligned."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml


def validate_skill(skill_dir: Path) -> tuple[bool, str]:
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return False, "SKILL.md not found"

    raw = skill_md.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        return False, "missing opening --- frontmatter"

    match = re.match(r"^---\n(.*?)\n---\s*\n", raw, re.DOTALL)
    if not match:
        return False, "invalid frontmatter (need closing --- on its own line)"

    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError as e:
        return False, f"YAML parse error: {e}"

    if not isinstance(meta, dict):
        return False, "frontmatter must be a mapping"

    name = meta.get("name")
    desc = meta.get("description")
    if not name or not isinstance(name, str):
        return False, "missing or invalid name:"
    if not desc or not isinstance(desc, str):
        return False, "missing or invalid description"
    if not re.match(r"^[a-z0-9-]+$", name):
        return False, f"name {name!r} must be hyphen-case (lowercase, digits, hyphens)"
    if name.startswith("-") or name.endswith("-") or "--" in name:
        return False, f"name {name!r} has invalid hyphen placement"
    if "<" in desc or ">" in desc:
        return False, "description must not contain angle brackets"

    return True, "ok"


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: uv run python scripts/validate_cursor_skill.py <skill-directory>", file=sys.stderr)
        return 2
    ok, msg = validate_skill(Path(sys.argv[1]).resolve())
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
