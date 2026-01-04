"""
Tests for upload normalization (Phase 1 - ADR007).

Tests normalize_upload_to_table_list() and helper functions.
"""

import io
import zipfile

import polars as pl
import pytest

from clinical_analytics.ui.storage.user_datasets import (
    SecurityError,
    UploadError,
    extract_zip_tables,
    load_single_file,
    normalize_upload_to_table_list,
)


class TestNormalizeUploadToTableList:
    """Test suite for normalize_upload_to_table_list() function."""

    def test_normalize_csv_to_table_list(self):
        """Test normalizing CSV upload returns single table with filename stem as name."""
        # Arrange
        df_data = {"patient_id": ["P001", "P002"], "age": [25, 30]}
        df = pl.DataFrame(df_data)
        file_bytes = df.write_csv().encode("utf-8")
        filename = "patient_outcomes.csv"

        # Act
        tables, metadata = normalize_upload_to_table_list(file_bytes, filename)

        # Assert
        assert len(tables) == 1
        assert tables[0]["name"] == "patient_outcomes"
        assert isinstance(tables[0]["data"], pl.DataFrame)
        assert tables[0]["data"].height == 2
        assert metadata["table_count"] == 1

    @pytest.mark.skip(reason="Excel writing requires xlsxwriter - test load_single_file directly instead")
    def test_normalize_excel_to_table_list(self):
        """Test normalizing Excel upload returns single table."""
        # Arrange
        df_data = {"patient_id": ["P001", "P002"], "age": [25, 30]}
        df = pl.DataFrame(df_data)
        # Write to Excel bytes
        buffer = io.BytesIO()
        df.write_excel(buffer)
        file_bytes = buffer.getvalue()
        filename = "clinical_data.xlsx"

        # Act
        tables, metadata = normalize_upload_to_table_list(file_bytes, filename)

        # Assert
        assert len(tables) == 1
        assert tables[0]["name"] == "clinical_data"
        assert isinstance(tables[0]["data"], pl.DataFrame)
        assert metadata["table_count"] == 1

    def test_normalize_zip_to_table_list(self):
        """Test normalizing ZIP upload returns multiple tables with ZIP entry names."""
        # Arrange
        patients_df = pl.DataFrame({"patient_id": ["P001", "P002"], "name": ["Alice", "Bob"]})
        admissions_df = pl.DataFrame({"admission_id": ["A001", "A002"], "patient_id": ["P001", "P002"]})

        # Create ZIP with two CSV files
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("patients.csv", patients_df.write_csv())
            zf.writestr("admissions.csv", admissions_df.write_csv())

        file_bytes = zip_buffer.getvalue()
        filename = "multi_table.zip"

        # Act
        tables, metadata = normalize_upload_to_table_list(file_bytes, filename)

        # Assert
        assert len(tables) == 2
        assert metadata["table_count"] == 2
        # Tables should be sorted by name
        table_names = [t["name"] for t in tables]
        assert "patients" in table_names
        assert "admissions" in table_names
        # Verify DataFrames
        for table in tables:
            assert isinstance(table["data"], pl.DataFrame)
            assert table["data"].height == 2

    def test_normalize_preserves_original_filename_stem(self):
        """Test that single-file normalization uses original filename stem, not 'table_0'."""
        # Arrange
        df = pl.DataFrame({"patient_id": ["P001"], "outcome": [1]})
        file_bytes = df.write_csv().encode("utf-8")
        filename = "viral_load_study.csv"

        # Act
        tables, metadata = normalize_upload_to_table_list(file_bytes, filename)

        # Assert
        assert tables[0]["name"] == "viral_load_study"
        assert tables[0]["name"] != "table_0"  # CRITICAL: not generic name

    def test_normalize_handles_gzip_compressed_csv_in_zip(self):
        """Test normalizing ZIP with .csv.gz files."""
        # Arrange
        import gzip

        df = pl.DataFrame({"patient_id": ["P001", "P002"], "age": [25, 30]})
        csv_bytes = df.write_csv().encode("utf-8")
        gzip_bytes = gzip.compress(csv_bytes)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("patients.csv.gz", gzip_bytes)

        file_bytes = zip_buffer.getvalue()
        filename = "compressed.zip"

        # Act
        tables, metadata = normalize_upload_to_table_list(file_bytes, filename)

        # Assert
        assert len(tables) == 1
        assert tables[0]["name"] == "patients"  # Stem without .csv.gz
        assert tables[0]["data"].height == 2


class TestExtractZipTables:
    """Test suite for extract_zip_tables() helper function."""

    def test_extract_valid_zip_returns_tables(self):
        """Test extracting valid ZIP returns table list."""
        # Arrange
        df1 = pl.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        df2 = pl.DataFrame({"id": [3, 4], "value": [10, 20]})

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("table1.csv", df1.write_csv())
            zf.writestr("table2.csv", df2.write_csv())

        file_bytes = zip_buffer.getvalue()

        # Act
        tables = extract_zip_tables(file_bytes)

        # Assert
        assert len(tables) == 2
        assert all(isinstance(t["data"], pl.DataFrame) for t in tables)

    def test_extract_zip_rejects_path_traversal(self):
        """Test that ZIP with path traversal is rejected."""
        # Arrange
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("../../../etc/passwd.csv", "malicious,data\n1,2")

        file_bytes = zip_buffer.getvalue()

        # Act & Assert
        with pytest.raises(SecurityError, match="Invalid path"):
            extract_zip_tables(file_bytes)

    def test_extract_zip_rejects_no_csv_files(self):
        """Test that ZIP with no CSV files raises UploadError."""
        # Arrange
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("readme.txt", "This is not a CSV")

        file_bytes = zip_buffer.getvalue()

        # Act & Assert
        with pytest.raises(UploadError, match="No CSV files in ZIP"):
            extract_zip_tables(file_bytes)

    def test_extract_zip_handles_duplicate_table_names(self):
        """Test that ZIP with duplicate table names raises UploadError."""
        # Arrange
        df1 = pl.DataFrame({"id": [1, 2]})
        df2 = pl.DataFrame({"id": [3, 4]})

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("patients.csv", df1.write_csv())
            zf.writestr("subfolder/patients.csv", df2.write_csv())  # Duplicate name!

        file_bytes = zip_buffer.getvalue()

        # Act & Assert
        with pytest.raises(UploadError, match="Duplicate table name"):
            extract_zip_tables(file_bytes)

    def test_extract_zip_skips_macosx_files(self):
        """Test that __MACOSX files are skipped."""
        # Arrange
        df = pl.DataFrame({"id": [1, 2]})

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("table.csv", df.write_csv())
            zf.writestr("__MACOSX/._table.csv", "garbage")

        file_bytes = zip_buffer.getvalue()

        # Act
        tables = extract_zip_tables(file_bytes)

        # Assert
        assert len(tables) == 1  # Only table.csv extracted
        assert tables[0]["name"] == "table"

    def test_extract_zip_handles_corrupted_file(self):
        """Test that corrupted ZIP raises UploadError."""
        # Arrange
        file_bytes = b"not a valid zip file"

        # Act & Assert
        with pytest.raises(UploadError, match="Corrupted ZIP file"):
            extract_zip_tables(file_bytes)


class TestLoadSingleFile:
    """Test suite for load_single_file() helper function."""

    def test_load_csv_returns_polars_dataframe(self):
        """Test loading CSV returns Polars DataFrame."""
        # Arrange
        df = pl.DataFrame({"patient_id": ["P001", "P002"], "age": [25, 30]})
        file_bytes = df.write_csv().encode("utf-8")
        filename = "patients.csv"

        # Act
        result = load_single_file(file_bytes, filename)

        # Assert
        assert isinstance(result, pl.DataFrame)
        assert result.height == 2
        assert "patient_id" in result.columns

    @pytest.mark.skip(reason="Excel writing requires xlsxwriter - manual testing needed")
    def test_load_excel_returns_polars_dataframe(self):
        """Test loading Excel returns Polars DataFrame."""
        # Arrange
        df = pl.DataFrame({"patient_id": ["P001", "P002"], "age": [25, 30]})
        buffer = io.BytesIO()
        df.write_excel(buffer)
        file_bytes = buffer.getvalue()
        filename = "patients.xlsx"

        # Act
        result = load_single_file(file_bytes, filename)

        # Assert
        assert isinstance(result, pl.DataFrame)
        assert result.height == 2

    def test_load_unsupported_file_raises_error(self):
        """Test that unsupported file type raises ValueError."""
        # Arrange
        file_bytes = b"some data"
        filename = "data.txt"

        # Act & Assert
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_single_file(file_bytes, filename)

    def test_load_preserves_column_types(self):
        """Test that column types are preserved during load."""
        # Arrange
        df = pl.DataFrame(
            {
                "patient_id": ["P001", "P002"],
                "age": [25, 30],
                "outcome": [0.5, 0.8],
            }
        )
        file_bytes = df.write_csv().encode("utf-8")
        filename = "data.csv"

        # Act
        result = load_single_file(file_bytes, filename)

        # Assert
        assert result.schema["patient_id"] == pl.Utf8
        assert result.schema["age"] == pl.Int64
        assert result.schema["outcome"] == pl.Float64
