"""Tests for scripts/make_synthetic_data.py: the synthetic kit is reproducible byte for byte."""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "make_synthetic_data.py"
_spec = importlib.util.spec_from_file_location("make_synthetic_data", _SCRIPT)
make_synthetic_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_synthetic_data)

KIT = sorted(make_synthetic_data.build_kit())


@pytest.fixture(scope="module")
def generated_kit(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-kit")
    make_synthetic_data.write_kit(root)
    return root


@pytest.mark.parametrize("rel", KIT, ids=[rel.name for rel in KIT])
def test_writeKit_freshGeneration_matchesCommittedBytes(rel, generated_kit, project_root):
    # Act
    generated = (generated_kit / rel).read_bytes()

    # Assert
    assert generated == (project_root / rel).read_bytes()


@pytest.mark.parametrize(
    "tamper",
    ["edited", "missing", "empty"],
)
def test_mismatches_committedFileChanged_reportsOnlyThatFile(tamper, generated_kit, tmp_path):
    # Arrange: a committed copy of the kit with one file changed
    make_synthetic_data.write_kit(tmp_path)
    target = tmp_path / KIT[0]
    if tamper == "edited":
        target.write_bytes(target.read_bytes().replace(b"\n", b"\r\n", 1))
    elif tamper == "missing":
        target.unlink()
    else:
        target.write_bytes(b"")

    # Act
    result = make_synthetic_data.mismatches(generated_kit, committed_root=tmp_path)

    # Assert
    assert result == [KIT[0]]


def test_buildKit_calledTwice_isDeterministic():
    # Act
    first, second = make_synthetic_data.build_kit(), make_synthetic_data.build_kit()

    # Assert
    for rel, frame in first.items():
        assert frame.equals(second[rel]), rel


def test_main_checkAgainstCommittedKit_returnsZero():
    # Act
    code = make_synthetic_data.main(["--check"])

    # Assert
    assert code == 0


def test_makeDemoCohort_plantedExceptions_preventPerfectSeparation():
    # Arrange
    demo = make_synthetic_data.make_demo_cohort()
    rule = (demo["age"] >= 55) | (demo["diabetes"] == 1)

    # Act
    breaks_rule = demo.filter(rule.cast(int) != demo["outcome"])["patient_id"].to_list()

    # Assert
    assert breaks_rule == ["P011", "P020"]
