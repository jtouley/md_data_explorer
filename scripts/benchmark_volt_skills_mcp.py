#!/usr/bin/env python3
"""Structural benchmark: volt skills + MCP reference wiring (no LLM).

Exits 0 when required files mention the MCP workbench; prints a score table.
Skills are expected under ~/.cursor/skills/; MCP reference is versioned in-repo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GLOBAL = Path.home() / ".cursor" / "skills"
REF = REPO / ".claude" / "skills-references" / "mcp-workbench.md"
CONTEXT = REPO / ".claude" / "agents" / "_mdde-repo-context.md"
EXTRA_SKILLS = [
    GLOBAL / "read-memories" / "SKILL.md",
    GLOBAL / "plan-to-pr" / "SKILL.md",
]


def main() -> int:
    rows: list[dict[str, str | bool]] = []

    def check(label: str, path: Path, needle: str) -> None:
        ok = path.is_file() and needle in path.read_text(encoding="utf-8")
        rows.append({"artifact": label, "ok": ok})

    check("versioned references/mcp-workbench.md", REF, "playwright")
    check("_mdde-repo-context.md → MCP workbench", CONTEXT, "mcp-workbench.md")
    check("~/.cursor/skills/read-memories/SKILL.md → MCP", EXTRA_SKILLS[0], "mcp-workbench")
    check("~/.cursor/skills/plan-to-pr/SKILL.md → MCP", EXTRA_SKILLS[1], "mcp-workbench")

    for path in sorted(GLOBAL.glob("mdde-*/SKILL.md")):
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        ok = "mcp-workbench.md" in text and "MCP workbench" in text
        rows.append({"artifact": str(path), "ok": ok})

    passed = sum(1 for r in rows if r["ok"])
    total = len(rows)
    out = {
        "benchmark": "mdde-skills-mcp-structure",
        "passed": passed,
        "total": total,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "rows": rows,
    }
    print(json.dumps(out, indent=2))
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
