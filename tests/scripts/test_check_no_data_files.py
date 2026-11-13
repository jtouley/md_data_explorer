"""Tests for scripts/check_no_data_files.py (tracked-data guard)."""

import importlib.util

import pytest

from scripts import check_no_data_files, git_tracked


@pytest.mark.parametrize(
    ("path", "flagged"),
    [
        ("data/analytics.duckdb", True),
        ("data/clinical_analytics.db", True),
        ("data/uploads/raw/upl_1.csv", True),
        ("exports/cohort.parquet", True),
        ("notes/patients.xlsx", True),
        ("scratch/extract.csv", True),
        ("src/clinical_analytics/resources/demo/sample_cohort.csv", False),
        ("data/demo/sample_cohort.csv", True),
        ("src/clinical_analytics/resources/config/extract.csv", True),
        ("data/dictionaries/Codebook v1.0.pdf", True),
        ("tests/fixtures/synthetic/cohort.csv", False),
        ("src/clinical_analytics/core/semantic.py", False),
    ],
    ids=[
        "duckdb_store",
        "session_db",
        "uploaded_csv",
        "parquet_anywhere",
        "excel_anywhere",
        "csv_outside_allowlist",
        "synthetic_demo_csv",
        "old_demo_location_under_data",
        "csv_in_shipped_config",
        "dictionary_pdf_under_data",
        "synthetic_test_fixture",
        "source_code",
    ],
)
def test_violations_path_flaggedUnlessAllowlisted(path, flagged):
    # Act
    result = check_no_data_files.violations([path])

    # Assert
    assert (result == [path]) is flagged


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        # Synthetic kit: only the generated files
        ("tests/fixtures/synthetic/real_extract.csv", "not_in_synthetic_kit"),
        ("src/clinical_analytics/resources/demo/clinic_export.xlsx", "not_in_synthetic_kit"),
        ("tests/fixtures/synthetic/cohort.csv", None),
        ("src/clinical_analytics/resources/demo/sample_cohort.csv", None),
        # Golden sets derived from real tables
        ("tests/eval/golden/cohort_aggregates.yaml", "golden_outside_kit"),
        ("tests/eval/golden_real_site.yml", "golden_outside_kit"),
        ("golden_questions.json", "golden_outside_kit"),
        ("tests/eval/golden_questions.yaml", None),
        ("src/clinical_analytics/resources/config/golden_examples.yaml", None),
        ("config/golden_examples.yaml", "golden_outside_kit"),
        ("tests/fixtures/synthetic/golden/cohort.yaml", None),
        ("tests/analysis/test_golden_statistics.py", None),
        # Overlays and metadata patches
        ("exports/pending.json", "overlay_or_patch"),
        ("notes/resolved.json", "overlay_or_patch"),
        ("notes/patches.jsonl", "overlay_or_patch"),
        ("notes/metadata_patch_upl_1.json", "overlay_or_patch"),
        ("src/clinical_analytics/core/metadata_patch.py", None),
        # Settings and session databases
        ("settings.json", "settings_or_session"),
        ("electron/settings.json", "settings_or_session"),
        ("backup/clinical_analytics.db", "settings_or_session"),
        (".claude/settings.json", None),
        # Environment files
        (".env", "env_file"),
        ("electron/.env.production", "env_file"),
        (".envrc", "env_file"),
        (".env.example", None),
        # Local config overrides
        (".claude/settings.local.json", "local_override"),
        ("config/nl_query.local.yaml", "local_override"),
        # Store sidecars: WAL, engine marker, backups, compaction marker, SQLite journals
        ("analytics.duckdb.wal", "store_sidecar"),
        ("notes/analytics.duckdb.engine", "store_sidecar"),
        ("notes/analytics.duckdb.backup-v1.0.0", "store_sidecar"),
        ("analytics.duckdb.needs-compaction", "store_sidecar"),
        ("clinical.db-journal", "store_sidecar"),
        ("src/clinical_analytics/storage/duckdb_store.py", None),
        # Logs, prompt-optimizer output, embedding caches, log key
        ("logs/nl_query.log", "log_or_cache"),
        ("notes/upl_1_queries.jsonl", "log_or_cache"),
        ("prompt_overlay.txt", "log_or_cache"),
        ("cache/golden_embeddings.npy", "log_or_cache"),
        (".log-key", "log_or_cache"),
        # Anything under a data-home-like path
        ("scratch/data/uploads/metadata/upl_1.json", "data_home_path"),
        ("scratch/data/nl_query_learning/iteration_01.json", "data_home_path"),
        ("home/.local/share/clinical-analytics/notes.md", "data_home_path"),
        ("ClinicalAnalytics/data/readme.txt", "data_home_path"),
        ("docs/architecture/overview.md", None),
        ("electron/package.json", None),
        ("", None),
    ],
)
def test_localOnlyViolations_path_flaggedWithReason(path, reason):
    # Act
    result = check_no_data_files.local_only_violations([path])

    # Assert
    assert result == ([] if reason is None else [(path, reason)])


def test_localOnlyViolations_trackedRepoFiles_noFalsePositives():
    # Arrange
    paths = git_tracked.tracked_files(git_tracked.repo_root())

    # Act
    result = check_no_data_files.local_only_violations(paths) + check_no_data_files.violations(paths)

    # Assert
    assert result == []


def test_syntheticKit_guardAllowlist_matchesGeneratorOutputs(project_root):
    # Arrange
    spec = importlib.util.spec_from_file_location(
        "make_synthetic_data", project_root / "scripts" / "make_synthetic_data.py"
    )
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    # Act
    generated = {rel.as_posix() for rel in generator.build_kit()}

    # Assert: the guard allows exactly what the generator writes, nothing more
    assert generated == check_no_data_files.SYNTHETIC_KIT


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
    assert (result == [(path, "contract_or_report")]) is flagged


def test_contentViolations_committedSyntheticContract_passes():
    # Arrange
    path = "tests/fixtures/synthetic/contracts/cohort.odcs.yaml"
    root = git_tracked.repo_root()

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
def test_localOnlyViolations_derivedDirsAndJsonContracts_codeNotFlaggedDataFlagged(path, flagged):
    # Act
    result = check_no_data_files.local_only_violations([path])

    # Assert
    assert (result == [(path, "contract_or_report")]) is flagged


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
    assert (result == [(path, "contract_or_report")]) is flagged


def test_contentViolations_trackedRepoFiles_noFalsePositives():
    # Arrange: every file git tracks today (the committed synthetic contract included).
    root = git_tracked.repo_root()
    paths = git_tracked.tracked_files(root)

    def read(p: str) -> str | None:
        try:
            return (root / p).read_text(encoding="utf-8")
        except (FileNotFoundError, UnicodeDecodeError, IsADirectoryError):
            return None

    # Act
    result = check_no_data_files.content_violations(paths, read)

    # Assert
    assert result == []


_REPORT_JSON = '{"status": "ran", "upload_id": "u1", "row_count": 400, "summary": ["ldl: 5.8% missing"], "checks": []}'


@pytest.mark.parametrize(
    ("path", "text", "flagged"),
    [
        ("exports/run.json", _REPORT_JSON, True),
        ("notes/report.yaml", "row_count: 400\nsummary: [a]\nchecks: []\n", True),
        ("tests/fixtures/synthetic/dq/run.json", _REPORT_JSON, False),
        ("electron/fixture.json", '{"summary": "text", "checks": 3}', False),
    ],
    ids=["report_json_outside_dq", "report_yaml", "synthetic_fixture", "other_shape"],
)
def test_contentViolations_qualityReportShape_flaggedOutsideSyntheticFixtures(path, text, flagged):
    # Act
    result = check_no_data_files.content_violations([path], lambda _p: text)

    # Assert
    assert (result == [(path, "contract_or_report")]) is flagged


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        ("contracts/upl_1/abc.odcs.yaml", "contract_or_report"),
        ("docs/examples/product.odps.yaml", "contract_or_report"),
        ("exports/dq/upl_1/run.json", "contract_or_report"),
        ("notes/lineage/graph.json", "contract_or_report"),
        ("scratch/data/contracts/upl_1/v1.odcs.yaml", "contract_or_report"),
        ("scratch/data/dq/README.md", "data_home_path"),
        ("tests/fixtures/synthetic/contracts/cohort.odcs.yaml", None),
        ("src/clinical_analytics/contracts/quality.py", None),
    ],
    ids=[
        "contract_outside_fixtures",
        "data_product_contract",
        "dq_report_dir",
        "lineage_dir",
        "contract_in_data_home",
        "doc_in_data_home_dq",
        "synthetic_contract",
        "contracts_source_module",
    ],
)
def test_localOnlyViolations_contractsAndReports_flaggedWithReason(path, reason):
    # Act
    result = check_no_data_files.local_only_violations([path])

    # Assert
    assert result == ([] if reason is None else [(path, reason)])


# --- Run location: the guard must not depend on the current directory -----------------------

_CONTRACT_YAML = b"apiVersion: v3.0.0\nkind: DataContract\nid: real\n"


@pytest.mark.parametrize("cwd", [".", "src", "src/pkg", "docs"], ids=["root", "src", "nested", "sibling_dir"])
def test_trackedFiles_fromSubdirectory_returnsRepoRelativePaths(make_git_repo, cwd):
    # Arrange
    root = make_git_repo({"src/pkg/a.py": b"x = 1\n", "docs/index.md": b"# Docs\n", "data/x.csv": b"a\n1\n"})

    # Act
    found_root = git_tracked.repo_root(root / cwd)
    result = git_tracked.tracked_files(found_root)

    # Assert
    assert found_root == root.resolve()
    assert sorted(result) == ["data/x.csv", "docs/index.md", "src/pkg/a.py"]


def test_repoRoot_outsideRepository_raisesGitQueryError(tmp_path):
    # Arrange
    outside = tmp_path / "not_a_repo"
    outside.mkdir()

    # Act / Assert
    with pytest.raises(git_tracked.GitQueryError, match="rev-parse"):
        git_tracked.repo_root(outside)


@pytest.mark.parametrize("cwd", [".", "src", "src/pkg"], ids=["root", "src", "nested"])
@pytest.mark.parametrize(
    ("files", "flagged_path"),
    [
        ({"data/extract.csv": b"a\n1\n"}, "data/extract.csv"),
        ({"notes/real.yaml": _CONTRACT_YAML}, "notes/real.yaml"),
        ({}, None),
    ],
    ids=["data_file", "contract_by_content", "clean"],
)
def test_main_runFromAnyDirectory_flagsByRepoPath(make_git_repo, monkeypatch, capsys, cwd, files, flagged_path):
    # Arrange
    root = make_git_repo({"src/pkg/a.py": b"x = 1\n", **files})
    monkeypatch.chdir(root / cwd)

    # Act
    code = check_no_data_files.main()

    # Assert
    err = capsys.readouterr().err
    assert code == (0 if flagged_path is None else 1)
    if flagged_path is not None:
        assert f"  {flagged_path}  [" in err
