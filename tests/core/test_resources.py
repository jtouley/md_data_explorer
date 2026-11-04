"""Shipped read-only resources resolve from the package, not from a repo checkout."""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import clinical_analytics
from clinical_analytics.core import resources

SHIPPED_CONFIG_FILES = (
    "datasets.yaml",
    "golden_examples.yaml",
    "logging.yaml",
    "nl_query.yaml",
    "nl_query_patterns.yaml",
    "prompt_learning.yaml",
    "ui.yaml",
    "validation.yaml",
)
PACKAGE_DIR = Path(clinical_analytics.__file__).resolve().parent


@pytest.mark.parametrize("name", SHIPPED_CONFIG_FILES)
def test_config_file_shippedName_resolvesInsidePackage(name):
    # Act
    path = resources.config_file(name)

    # Assert
    assert path.is_file()
    assert path.resolve().is_relative_to(PACKAGE_DIR)


def test_demo_file_sampleCohort_resolvesInsidePackage():
    # Act
    path = resources.demo_file("sample_cohort.csv")

    # Assert
    assert path.is_file()
    assert path.resolve().is_relative_to(PACKAGE_DIR)


@pytest.mark.parametrize(
    "name",
    ["", ".", "..", "../secrets.yaml", "sub/ui.yaml", "/etc/passwd"],
    ids=["empty", "dot", "dotdot", "parent_traversal", "subdir", "absolute"],
)
def test_config_file_nonBareName_raisesValueError(name):
    # Act / Assert
    with pytest.raises(ValueError, match="bare file name"):
        resources.config_file(name)


def test_resources_root_zippedPackage_raisesResourceNotFound(monkeypatch, tmp_path):
    # Arrange: importlib.resources hands back a zip traversable instead of a directory
    archive = tmp_path / "pkg.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("clinical_analytics/resources/config/ui.yaml", "{}")
    monkeypatch.setattr(resources.resources, "files", lambda _pkg: zipfile.Path(archive, "clinical_analytics/"))

    # Act / Assert
    with pytest.raises(resources.ResourceNotFoundError, match="not installed as a directory"):
        resources.resources_root()


def test_loaders_packageCopiedOutsideRepo_readShippedConfig(tmp_path):
    """The package relocated away from the checkout (as in an install) still finds its config."""
    # Arrange: copy only the package; no config/ or pyproject.toml anywhere above it
    site = tmp_path / "site-packages"
    shutil.copytree(PACKAGE_DIR, site / "clinical_analytics", ignore=shutil.ignore_patterns("__pycache__"))
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    probe = (
        "import clinical_analytics, json\n"
        "from clinical_analytics.core.config_loader import load_nl_query_config, load_validation_config\n"
        "from clinical_analytics.core.mapper import get_global_config\n"
        "from clinical_analytics.core.prompt_optimizer import LearningConfig\n"
        "from clinical_analytics.datasets.demo import get_demo_csv_path\n"
        "print(json.dumps({'pkg': clinical_analytics.__file__, 'nl': sorted(load_nl_query_config())[:3],\n"
        "  'validation': bool(load_validation_config()), 'global': isinstance(get_global_config(), dict),\n"
        "  'intents': bool(LearningConfig.load().valid_intents), 'demo': get_demo_csv_path().is_file()}))\n"
    )

    # Act
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=cwd,
        env={"PYTHONPATH": str(site), "PATH": "", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=120,
    )

    # Assert
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = proc.stdout.strip().splitlines()[-1]
    assert str(site) in out
    assert '"validation": true' in out
    assert '"global": true' in out
    assert '"intents": true' in out
    assert '"demo": true' in out
