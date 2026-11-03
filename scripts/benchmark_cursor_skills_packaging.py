#!/usr/bin/env python3
"""Run skill-creator-style validation on each global Cursor skill (one-by-one).

Writes JSON + Markdown under .cursor/benchmarks/skill-packaging/iteration-1/
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GLOBAL = Path.home() / ".cursor" / "skills"
OUT_DIR = REPO / ".cursor" / "benchmarks" / "skill-packaging" / "iteration-1"

_vpath = REPO / "scripts" / "validate_cursor_skill.py"
_spec = importlib.util.spec_from_file_location("validate_cursor_skill", _vpath)
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load validate_cursor_skill")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
validate_skill = _mod.validate_skill


def main() -> int:
    rows: list[dict[str, str | bool]] = []
    if not GLOBAL.is_dir():
        print(f"no global skills dir: {GLOBAL}", file=sys.stderr)
        return 1

    for path in sorted(GLOBAL.iterdir()):
        if not path.is_dir():
            continue
        if path.name.startswith("."):
            continue
        skill_md = path / "SKILL.md"
        if not skill_md.is_file():
            continue
        ok, msg = validate_skill(path)
        rows.append({"skill": path.name, "ok": ok, "message": msg})

    passed = sum(1 for r in rows if r["ok"])
    total = len(rows)
    payload = {
        "benchmark": "skill-packaging-validate",
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "global_skills_root": str(GLOBAL),
        "passed": passed,
        "total": total,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "rows": rows,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Skill packaging benchmark (validation only)",
        "",
        f"**When:** {payload['timestamp']}",
        f"**Root:** `{GLOBAL}`",
        "",
        f"| Pass rate | {passed}/{total} |",
        "",
        "| Skill | OK | Message |",
        "|-------|----|---------|",
    ]
    for r in rows:
        ok = "yes" if r["ok"] else "no"
        msg = str(r["message"]).replace("|", "\\|")
        lines.append(f"| {r['skill']} | {ok} | {msg} |")
    lines.append("")
    lines.append(
        "Run again: `make benchmark-cursor-skills` or `uv run python scripts/benchmark_cursor_skills_packaging.py`"
    )
    (OUT_DIR / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
