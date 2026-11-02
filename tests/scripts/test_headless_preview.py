"""Headless CSV → JSON preview smoke (issue #51)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "headless_preview.py"
DEMO_CSV = PROJECT_ROOT / "data" / "demo" / "sample_cohort.csv"


@pytest.mark.integration
def test_headless_preview_cli_demo_csv_prints_json_without_streamlit(tmp_path):
    """CLI loads a CSV and prints columns + row_count JSON; no Streamlit pages imported."""
    # Arrange
    assert SCRIPT.exists(), "scripts/headless_preview.py must exist"
    assert DEMO_CSV.exists(), "demo CSV must exist"

    # Act
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--csv",
            str(DEMO_CSV),
            "--upload-dir",
            str(tmp_path / "uploads"),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    # Assert
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "columns" in payload
    assert "row_count" in payload
    assert payload["row_count"] >= 50
    for col in ("patient_id", "outcome", "age", "sex"):
        assert col in payload["columns"]
    assert "streamlit" not in proc.stdout.lower()
    # Guard: entrypoint must not pull in Streamlit pages
    assert "ui/pages" not in proc.stderr
    assert "ui.pages" not in proc.stderr


def test_headless_preview_module_does_not_import_streamlit_pages():
    """Public entrypoint source must not import Streamlit or UI pages."""
    # Arrange / Act
    from clinical_analytics.cli import headless_preview as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")

    # Assert
    assert "import streamlit" not in src
    assert "from streamlit" not in src
    assert "ui.pages" not in src
    assert "ui/pages" not in src
