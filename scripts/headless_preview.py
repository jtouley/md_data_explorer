#!/usr/bin/env python3
"""Thin wrapper: headless CSV → JSON preview (issue #51)."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python scripts/headless_preview.py` without install.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clinical_analytics.cli.headless_preview import main

if __name__ == "__main__":
    raise SystemExit(main())
