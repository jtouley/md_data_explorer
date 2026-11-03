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


@pytest.mark.parametrize(
    ("path", "flagged"),
    [
        ("src/clinical_analytics/lineage/__init__.py", False),
        ("src/clinical_analytics/lineage/graph.py", False),
        ("src/clinical_analytics/dq/checks.py", False),
        ("tests/lineage/test_graph.py", False),
        ("docs/lineage/overview.md", False),
        ("exports/lineage/graph.jsonl", True),
        ("exports/dq/u1/v1/run.yaml", True),
        ("exports/dq/u1/v1/run.parquet", True),
        ("contracts/upl_1/abc.odcs.json", True),
        ("exports/product.odps.json", True),
        ("tests/fixtures/synthetic/contracts/c.odcs.json", False),
    ],
    ids=[
        "lineage_package_init",
        "lineage_module",
        "dq_module",
        "lineage_test_module",
        "lineage_doc",
        "lineage_jsonl",
        "dq_yaml_report",
        "dq_parquet",
        "odcs_json_contract",
        "odps_json_contract",
        "synthetic_odcs_json",
    ],
)
def test_violations_derivedDirsAndJsonContracts_codeNotFlaggedDataFlagged(path, flagged):
    # Act
    result = check_no_data_files.violations([path])

    # Assert
    assert (result == [path]) is flagged


_ODCS = "apiVersion: v3.2.0\nkind: DataContract\nid: u1\nschema:\n  - name: unified\n    physicalName: unified\n"
_SCHEMA_ONLY = "schema:\n  - name: unified\n    physicalName: t\n    properties:\n      - name: age\n"


@pytest.mark.parametrize(
    ("path", "text", "flagged"),
    [
        ("notes/cohort.yaml", _ODCS, True),
        ("notes/cohort.yml", _ODCS.replace("v3.2.0", "v3.0.2"), True),
        ("notes/cohort.json", '{"apiVersion": "v3.1.0", "kind": "DataContract", "id": "u1"}', True),
        ("notes/cohort.yaml", _SCHEMA_ONLY, True),
        ("tests/fixtures/synthetic/contracts/c.yaml", "tags: [synthetic]\n" + _ODCS, False),
        ("tests/fixtures/synthetic/contracts/c.yaml", _ODCS, True),
        ("docs/openapi.yaml", "openapi: 3.1.0\nkind: Api\n", False),
        ("config/app.yaml", "apiVersion: v1\nkind: ConfigMap\n", False),
        ("electron/package.json", '{"name": "app", "version": "1.0.0"}', False),
        ("notes/broken.json", '{"apiVersion": "v3.2.0", "kind": ', False),
    ],
    ids=[
        "renamed_yaml_no_hash",
        "renamed_yml_v3_0",
        "json_contract",
        "schema_with_physical_name",
        "synthetic_tagged_in_fixtures",
        "untagged_in_fixtures",
        "other_yaml",
        "k8s_like_yaml",
        "package_json",
        "unparseable_json_without_hash",
    ],
)
def test_contentViolations_odcsShapedFile_flaggedOutsideTaggedSyntheticFixtures(path, text, flagged):
    # Act
    result = check_no_data_files.content_violations([path], lambda _p: text)

    # Assert
    assert (result == [path]) is flagged


def test_contentViolations_trackedRepoFiles_noFalsePositives():
    # Arrange: every file git tracks today (the committed synthetic contract included).
    root = Path(__file__).resolve().parents[2]
    paths = check_no_data_files.tracked_files()

    def read(p: str) -> str | None:
        try:
            return (root / p).read_text(encoding="utf-8")
        except (FileNotFoundError, UnicodeDecodeError, IsADirectoryError):
            return None

    # Act
    result = check_no_data_files.content_violations(paths, read) + check_no_data_files.violations(paths)

    # Assert
    assert result == []
