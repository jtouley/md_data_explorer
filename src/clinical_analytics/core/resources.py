"""One resolver for read-only files the app ships with.

Config YAML and the synthetic demo cohort live inside the package
(``clinical_analytics/resources``) and are found with ``importlib.resources``, so they resolve
the same way in a source checkout, an installed wheel, or a bundled runtime, whatever the cwd.
Nothing here is ever written: runtime data goes to ``core.data_home``.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

RESOURCES_PACKAGE = "clinical_analytics.resources"
CONFIG_SUBDIR = "config"
DEMO_SUBDIR = "demo"


class ResourceNotFoundError(FileNotFoundError):
    """A file that should ship with the app is missing from the installed package."""


def resources_root() -> Path:
    """Directory holding the shipped resources (must be a real directory, not a zip)."""
    root = resources.files(RESOURCES_PACKAGE)
    if not isinstance(root, Path):
        raise ResourceNotFoundError(
            f"{RESOURCES_PACKAGE} is not installed as a directory ({type(root).__name__}); "
            "install the app unpacked so its config files can be read"
        )
    return root


def shipped_config_dir() -> Path:
    """Directory of the shipped YAML config files."""
    return resources_root() / CONFIG_SUBDIR


def _bare_name(name: str) -> str:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"resource name must be a bare file name, got {name!r}")
    return name


def config_file(name: str) -> Path:
    """Path to one shipped config file (existence is checked by the caller's loader)."""
    return shipped_config_dir() / _bare_name(name)


def demo_file(name: str) -> Path:
    """Path to a file of the bundled synthetic demo cohort."""
    return resources_root() / DEMO_SUBDIR / _bare_name(name)
