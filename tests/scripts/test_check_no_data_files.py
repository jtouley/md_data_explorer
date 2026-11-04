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
        ("data/demo/clinic_export.xlsx", "not_in_synthetic_kit"),
        ("tests/fixtures/synthetic/cohort.csv", None),
        ("data/demo/sample_cohort.csv", None),
        # Golden sets derived from real tables
        ("tests/eval/golden/cohort_aggregates.yaml", "golden_outside_kit"),
        ("tests/eval/golden_real_site.yml", "golden_outside_kit"),
        ("golden_questions.json", "golden_outside_kit"),
        ("tests/eval/golden_questions.yaml", None),
        ("config/golden_examples.yaml", None),
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
    paths = check_no_data_files.tracked_files()

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
