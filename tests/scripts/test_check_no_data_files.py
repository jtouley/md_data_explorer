"""Tests for scripts/check_no_data_files.py (tracked-data guard)."""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_no_data_files.py"
_spec = importlib.util.spec_from_file_location("check_no_data_files", _SCRIPT)
check_no_data_files = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_no_data_files)


@pytest.mark.parametrize(
    ("path", "flagged"),
    [
        ("data/analytics.duckdb", True),
        ("data/clinical_analytics.db", True),
        ("data/uploads/raw/upl_1.csv", True),
        ("exports/cohort.parquet", True),
        ("notes/patients.xlsx", True),
        ("scratch/extract.csv", True),
        ("data/demo/sample_cohort.csv", False),
        ("data/dictionaries/Codebook v1.0.pdf", False),
        ("tests/fixtures/synthetic/cohort.csv", False),
        ("src/clinical_analytics/core/semantic.py", False),
        ("contracts/upl_1/abc.odcs.yaml", True),
        ("docs/examples/product.odps.yaml", True),
        ("exports/dq/upl_1/run.json", True),
        ("notes/lineage/graph.json", True),
        ("tests/eval/golden_real_cohort.yaml", True),
        ("tests/fixtures/synthetic/contracts/cohort.odcs.yaml", False),
        ("tests/eval/golden_questions.yaml", False),
        ("src/clinical_analytics/contracts/quality.py", False),
        ("tests/core/test_golden_question_generation.py", False),
    ],
    ids=[
        "duckdb_store",
        "session_db",
        "uploaded_csv",
        "parquet_anywhere",
        "excel_anywhere",
        "csv_outside_allowlist",
        "synthetic_demo_csv",
        "public_dictionary_pdf",
        "synthetic_test_fixture",
        "source_code",
        "contract_outside_fixtures",
        "data_product_contract",
        "dq_report_dir",
        "lineage_dir",
        "eval_golden_set",
        "synthetic_contract",
        "handwritten_golden_questions",
        "contracts_source_module",
        "golden_test_module",
    ],
)
def test_violations_path_flaggedUnlessAllowlisted(path, flagged):
    # Act
    result = check_no_data_files.violations([path])

    # Assert
    assert (result == [path]) is flagged


_HASH = "customProperties:\n  - property: sourceFileSha256\n    value: abc\n"


@pytest.mark.parametrize(
    ("path", "text", "flagged"),
    [
        ("contract.yaml", "kind: DataContract\n" + _HASH, True),
        ("tests/fixtures/synthetic/contracts/c.odcs.yaml", "kind: DataContract\n" + _HASH, True),
        ("c.yaml", "tags: [synthetic]\n" + _HASH, False),
        ("c.yaml", "sourceFileSha256: {broken", True),
        ("notes.yaml", "description: we record sourceFileSha256 in contracts\n", False),
        ("c.json", _HASH, False),
        ("deleted.yaml", None, False),
        ("multi.yaml", "tags: [synthetic]\n" + _HASH + "---\n" + _HASH, True),
    ],
    ids=[
        "untagged_hash",
        "untagged_hash_in_allowlisted_dir",
        "synthetic_tag",
        "unparseable_mentions_hash",
        "mentions_without_property",
        "not_yaml",
        "deleted_in_worktree",
        "second_document_untagged",
    ],
)
def test_contentViolations_sourceHashYaml_flaggedUnlessTaggedSynthetic(path, text, flagged):
    # Act
    result = check_no_data_files.content_violations([path], lambda _p: text)

    # Assert
    assert (result == [path]) is flagged


def test_contentViolations_committedSyntheticContract_passes():
    # Arrange
    path = "tests/fixtures/synthetic/contracts/cohort.odcs.yaml"
    root = Path(__file__).resolve().parents[2]

    # Act
    result = check_no_data_files.content_violations([path], lambda p: (root / p).read_text(encoding="utf-8"))

    # Assert
    assert result == []
