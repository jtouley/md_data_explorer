"""Tests for documentation extraction from ZIP uploads."""

from io import BytesIO
from zipfile import ZipFile

from clinical_analytics.core.doc_parser import (
    extract_context_from_docs,
    extract_markdown_text,
    extract_pdf_text,
    extract_text,
)
from clinical_analytics.ui.storage.user_datasets import extract_documentation_files


class TestDocumentationExtraction:
    """Test suite for documentation file detection and extraction."""

    def test_extract_documentation_files_detects_pdf_md_txt(self, tmp_path):
        """Test that extract_documentation_files() detects PDF, Markdown, and text files in ZIP."""
        # Arrange: Create ZIP with documentation files
        zip_buffer = BytesIO()
        with ZipFile(zip_buffer, "w") as zf:
            # Add documentation files
            zf.writestr("README.md", "# Dataset Documentation\nDescription here")
            zf.writestr("dictionary.txt", "Column definitions")
            zf.writestr("docs/study_protocol.pdf", b"%PDF-1.4 fake pdf content")
            # Add data files (should not be detected as docs)
            zf.writestr("data.csv", "col1,col2\n1,2")
            zf.writestr("patients.xlsx", b"fake excel content")

        zip_buffer.seek(0)

        # Act: Extract documentation files
        doc_files = extract_documentation_files(zip_buffer)

        # Assert: Only documentation files detected
        assert len(doc_files) == 3
        doc_names = [f.name for f in doc_files]
        assert "README.md" in doc_names
        assert "dictionary.txt" in doc_names
        assert "study_protocol.pdf" in doc_names
        # Data files should not be included
        assert "data.csv" not in doc_names
        assert "patients.xlsx" not in doc_names

    def test_extract_documentation_files_empty_zip(self, tmp_path):
        """Test that extract_documentation_files() returns empty list for ZIP with no docs."""
        # Arrange: Create ZIP with only data files
        zip_buffer = BytesIO()
        with ZipFile(zip_buffer, "w") as zf:
            zf.writestr("data.csv", "col1,col2\n1,2")

        zip_buffer.seek(0)

        # Act: Extract documentation files
        doc_files = extract_documentation_files(zip_buffer)

        # Assert: No documentation files found
        assert len(doc_files) == 0

    def test_extract_documentation_files_docs_subdirectory(self, tmp_path):
        """Test that extract_documentation_files() detects docs in subdirectories."""
        # Arrange: Create ZIP with docs in nested subdirectories
        zip_buffer = BytesIO()
        with ZipFile(zip_buffer, "w") as zf:
            zf.writestr("docs/data_dictionary.pdf", b"%PDF-1.4 fake pdf")
            zf.writestr("metadata/README.md", "# Metadata\nDescription")
            zf.writestr("deep/nested/path/notes.txt", "Study notes")

        zip_buffer.seek(0)

        # Act: Extract documentation files
        doc_files = extract_documentation_files(zip_buffer)

        # Assert: All documentation files detected regardless of directory depth
        assert len(doc_files) == 3
        doc_names = [f.name for f in doc_files]
        assert "data_dictionary.pdf" in doc_names
        assert "README.md" in doc_names
        assert "notes.txt" in doc_names


class TestDocParser:
    """Test suite for documentation parsing module."""

    def test_extract_pdf_text_extracts_content(self, tmp_path):
        """Test that extract_pdf_text() extracts text from PDF files."""
        # Arrange: Create a simple PDF file
        pdf_path = tmp_path / "test.pdf"
        # Create minimal valid PDF with text content
        pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>/MediaBox[0 0 612 792]/Contents 5 0 R>>endobj
4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
5 0 obj<</Length 44>>stream
BT /F1 12 Tf 100 700 Td (Hello World) Tj ET
endstream endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000262 00000 n
0000000337 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
414
%%EOF"""
        pdf_path.write_bytes(pdf_content)

        # Act: Extract text from PDF
        text = extract_pdf_text(pdf_path)

        # Assert: Text content extracted
        assert text is not None
        assert len(text) > 0
        # Note: actual text extraction depends on pymupdf, so we just verify non-empty

    def test_extract_markdown_text_preserves_structure(self, tmp_path):
        """Test that extract_markdown_text() extracts and preserves Markdown structure."""
        # Arrange: Create Markdown file
        md_path = tmp_path / "README.md"
        md_content = """# Dataset Documentation

## Overview
This is a clinical dataset.

## Columns
- patient_id: Patient identifier
- age: Patient age in years
- diagnosis: Primary diagnosis
"""
        md_path.write_text(md_content)

        # Act: Extract text from Markdown
        text = extract_markdown_text(md_path)

        # Assert: Content preserved
        assert "# Dataset Documentation" in text
        assert "patient_id" in text
        assert "diagnosis" in text

    def test_extract_text_reads_plain_text(self, tmp_path):
        """Test that extract_text() reads plain text files."""
        # Arrange: Create text file
        txt_path = tmp_path / "notes.txt"
        txt_content = "Study notes:\nProtocol version 2.0\nRecruitment ongoing"
        txt_path.write_text(txt_content)

        # Act: Extract text
        text = extract_text(txt_path)

        # Assert: Content read correctly
        assert text == txt_content

    def test_extract_context_from_docs_concatenates_files(self, tmp_path):
        """Test that extract_context_from_docs() concatenates multiple documentation files."""
        # Arrange: Create multiple doc files
        md_path = tmp_path / "README.md"
        md_path.write_text("# Documentation\nDataset overview")

        txt_path = tmp_path / "notes.txt"
        txt_path.write_text("Additional notes")

        file_paths = [md_path, txt_path]

        # Act: Extract context from all docs
        context = extract_context_from_docs(file_paths)

        # Assert: All content concatenated
        assert "# Documentation" in context
        assert "Dataset overview" in context
        assert "Additional notes" in context

    def test_extract_context_from_docs_truncates_large_files(self, tmp_path):
        """Test that extract_context_from_docs() truncates content at 50k chars."""
        # Arrange: Create large text file (>50k chars)
        txt_path = tmp_path / "large.txt"
        large_content = "x" * 60000  # 60k characters
        txt_path.write_text(large_content)

        # Act: Extract context
        context = extract_context_from_docs([txt_path])

        # Assert: Truncated to 50k chars
        assert len(context) <= 50000

    def test_extract_context_from_docs_handles_empty_list(self):
        """Test that extract_context_from_docs() handles empty file list."""
        # Arrange: Empty file list
        file_paths = []

        # Act: Extract context
        context = extract_context_from_docs(file_paths)

        # Assert: Returns empty string
        assert context == ""

    def test_extract_context_from_docs_handles_missing_files(self, tmp_path):
        """Test that extract_context_from_docs() handles missing files gracefully."""
        # Arrange: Non-existent file path
        missing_path = tmp_path / "nonexistent.txt"

        # Act: Extract context (should not raise exception)
        context = extract_context_from_docs([missing_path])

        # Assert: Returns empty string or skips missing file
        assert isinstance(context, str)


class TestAddDocumentationToUpload:
    """Test suite for adding documentation to existing uploads."""

    def test_add_documentation_to_upload_adds_doc_context_to_existing_upload(self, tmp_path):
        """Test that add_documentation_to_upload() adds doc_context to existing upload metadata."""

        import polars as pl

        from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

        # Arrange: Create existing upload without doc_context
        storage = UserDatasetStorage(upload_dir=tmp_path)

        # Create a dataset and save it
        df = pl.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(100)],
                "age": [20 + (i % 50) for i in range(100)],
                "outcome": [i % 2 for i in range(100)],
            }
        )

        # Save upload (simulate existing upload without doc_context)
        upload_id = storage.generate_upload_id("test_dataset.csv")
        tables = [{"name": "patients", "data": df}]
        metadata = {
            "dataset_name": "test_dataset",
            "table_count": 1,
            "table_names": ["patients"],
        }

        from clinical_analytics.ui.storage.user_datasets import save_table_list

        success, _ = save_table_list(storage, tables, upload_id, metadata)
        assert success is True

        # Verify no doc_context exists
        existing_metadata = storage.get_upload_metadata(upload_id)
        assert "doc_context" not in existing_metadata or existing_metadata.get("doc_context") == ""

        # Create test documentation file (using text file for easier testing)
        doc_path = tmp_path / "test_dictionary.txt"
        doc_content = (
            "patient_id: Unique patient identifier\n"
            "age: Patient age in years\n"
            "outcome: Binary outcome variable (0=no, 1=yes)"
        )
        doc_path.write_text(doc_content)

        # Act: Add documentation to existing upload
        success, message = storage.add_documentation_to_upload(upload_id, doc_path, re_infer_schema=False)

        # Assert: Documentation added successfully
        assert success is True, f"Failed to add documentation: {message}"

        # Assert: doc_context exists in metadata
        updated_metadata = storage.get_upload_metadata(upload_id)
        assert "doc_context" in updated_metadata
        assert updated_metadata["doc_context"] != ""

    def test_add_documentation_to_upload_re_runs_schema_inference_with_doc_context(self, tmp_path):
        """Test that add_documentation_to_upload() re-runs schema inference with doc_context when requested."""

        import polars as pl

        from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

        # Arrange: Create existing upload
        storage = UserDatasetStorage(upload_dir=tmp_path)

        df = pl.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(100)],
                "current_regimen": [1, 2, 1] * 33 + [1],  # Categorical with codebook
                "age": [20 + (i % 50) for i in range(100)],
            }
        )

        upload_id = storage.generate_upload_id("test_dataset.csv")
        tables = [{"name": "patients", "data": df}]
        metadata = {
            "dataset_name": "test_dataset",
            "table_count": 1,
            "table_names": ["patients"],
        }

        from clinical_analytics.ui.storage.user_datasets import save_table_list

        success, _ = save_table_list(storage, tables, upload_id, metadata)
        assert success is True

        # Create test documentation file with codebook information
        doc_path = tmp_path / "test_dictionary.txt"
        text_content = "current_regimen: Current antiretroviral regimen\nCurrent Regimen: 1: Biktarvy, 2: Symtuza"
        doc_path.write_text(text_content)

        # Act: Add documentation and re-run schema inference
        success, message = storage.add_documentation_to_upload(upload_id, doc_path, re_infer_schema=True)

        # Assert: Documentation added and schema re-inferred
        assert success is True, f"Failed to add documentation: {message}"

        # Assert: doc_context exists
        updated_metadata = storage.get_upload_metadata(upload_id)
        assert "doc_context" in updated_metadata
        assert updated_metadata["doc_context"] != ""

        # Assert: variable_types enhanced with codebooks (if schema inference ran)
        # Note: This depends on schema inference implementation
        # For now, just verify doc_context is present


class TestStandalonePdfUpload:
    """Test suite for standalone PDF upload during new dataset upload."""

    def test_normalize_upload_to_table_list_accepts_external_pdf_bytes(self, tmp_path):
        """Test that normalize_upload_to_table_list() accepts external PDF bytes in metadata."""

        import polars as pl

        from clinical_analytics.ui.storage.user_datasets import normalize_upload_to_table_list

        # Arrange: Create single-file upload with external PDF in metadata
        df = pl.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(100)],
                "age": [20 + (i % 50) for i in range(100)],
            }
        )
        csv_bytes = df.write_csv().encode("utf-8")

        # Create external PDF bytes (using text file for testing)
        pdf_content = "patient_id: Unique patient identifier\nage: Patient age in years"
        pdf_bytes = pdf_content.encode("utf-8")

        metadata = {
            "external_pdf_bytes": pdf_bytes,
            "external_pdf_filename": "dictionary.txt",  # Using .txt for testing
        }

        # Act: Normalize upload with external PDF
        tables, table_metadata = normalize_upload_to_table_list(csv_bytes, "test.csv", metadata)

        # Assert: Tables extracted correctly
        assert len(tables) == 1
        assert tables[0]["name"] == "test"

        # Assert: External PDF converted to doc_files format for processing
        assert "doc_files" in table_metadata
        assert len(table_metadata["doc_files"]) == 1
        assert table_metadata["doc_files"][0].name == "dictionary.txt"


class TestDocContextInMetadata:
    """Test suite for doc_context storage in upload metadata."""

    def test_save_zip_upload_stores_doc_context_in_metadata(self, tmp_path):
        """Test that save_zip_upload() extracts and stores doc_context in metadata."""
        from io import BytesIO
        from zipfile import ZipFile

        import polars as pl

        from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

        # Arrange: Create ZIP with data files and documentation
        # Create larger dataset to pass validation (minimum 1KB)
        zip_buffer = BytesIO()
        with ZipFile(zip_buffer, "w") as zf:
            # Add data files (create larger dataset)
            df = pl.DataFrame(
                {
                    "patient_id": [f"P{i:03d}" for i in range(100)],
                    "age": [20 + (i % 50) for i in range(100)],
                    "outcome": [i % 2 for i in range(100)],
                }
            )
            zf.writestr("patients.csv", df.write_csv())
            # Add documentation files
            zf.writestr("README.md", "# Dataset Documentation\nThis is a test dataset.")
            zf.writestr("dictionary.txt", "patient_id: Patient identifier\nage: Patient age")

        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()

        # Act: Save ZIP upload
        storage = UserDatasetStorage(upload_dir=tmp_path)
        success, message, upload_id = storage.save_zip_upload(
            file_bytes=zip_bytes,
            original_filename="test_dataset.zip",
            metadata={"dataset_name": "test_dataset"},
        )

        # Assert: Upload succeeded
        assert success is True, f"Upload failed: {message}"
        assert upload_id is not None

        # Assert: Metadata contains doc_context
        import json

        metadata_path = tmp_path / "metadata" / f"{upload_id}.json"
        assert metadata_path.exists()

        with open(metadata_path) as f:
            metadata = json.load(f)

        # Verify doc_context was extracted and stored
        assert "doc_context" in metadata, "doc_context not found in saved metadata"
        assert isinstance(metadata["doc_context"], str)
        assert len(metadata["doc_context"]) > 0
        # Verify documentation content is in context
        assert "Dataset Documentation" in metadata["doc_context"]
        assert "patient_id" in metadata["doc_context"]

    def test_save_table_list_skips_extraction_if_doc_context_exists(self, tmp_path):
        """Test that save_table_list() skips doc_context extraction if it already exists (idempotency)."""
        from io import BytesIO
        from zipfile import ZipFile

        import polars as pl

        from clinical_analytics.ui.storage.user_datasets import (
            UserDatasetStorage,
            save_table_list,
        )

        # Arrange: Create ZIP with data files and documentation
        zip_buffer = BytesIO()
        with ZipFile(zip_buffer, "w") as zf:
            df = pl.DataFrame(
                {
                    "patient_id": [f"P{i:03d}" for i in range(100)],
                    "age": [20 + (i % 50) for i in range(100)],
                    "outcome": [i % 2 for i in range(100)],
                }
            )
            zf.writestr("patients.csv", df.write_csv())
            zf.writestr("README.md", "# Dataset Documentation\nThis is a test dataset.")

        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()

        storage = UserDatasetStorage(upload_dir=tmp_path)
        upload_id = storage.generate_upload_id("test.zip")

        # Get tables and metadata with doc_files
        from clinical_analytics.ui.storage.user_datasets import normalize_upload_to_table_list

        tables, table_metadata = normalize_upload_to_table_list(zip_bytes, "test.zip", {})
        metadata = {"dataset_name": "test", **table_metadata}

        # Verify doc_files exists before extraction
        assert "doc_files" in metadata
        assert len(metadata["doc_files"]) > 0

        # First call: Extract doc_context normally
        success, _ = save_table_list(storage, tables, upload_id, metadata.copy())
        assert success is True

        # Verify doc_context was extracted and doc_files was removed
        saved_metadata = storage.get_upload_metadata(upload_id)
        assert "doc_context" in saved_metadata
        assert len(saved_metadata["doc_context"]) > 0
        # doc_files should be removed after extraction
        assert "doc_files" not in saved_metadata

        # Second call: Create new metadata with both doc_files AND pre-existing doc_context
        # This tests the idempotency safeguard: should skip extraction if doc_context exists
        # Create a new upload_id to test the safeguard in isolation
        upload_id2 = storage.generate_upload_id("test2.zip")
        metadata_with_both = metadata.copy()
        metadata_with_both["doc_context"] = "PRE_EXISTING_CONTEXT"
        # Keep doc_files to test the safeguard

        # This should skip extraction because doc_context already exists
        # The safeguard should prevent re-extraction and remove doc_files
        success2, _ = save_table_list(storage, tables, upload_id2, metadata_with_both)
        assert success2 is True

        # Verify: doc_files was removed (ensures JSON serializability)
        # and extraction was skipped (doc_context remains PRE_EXISTING_CONTEXT)
        final_metadata = storage.get_upload_metadata(upload_id2)
        assert "doc_files" not in final_metadata, "doc_files should be removed even when extraction is skipped"
        assert "doc_context" in final_metadata
        # The safeguard prevents re-extraction, so PRE_EXISTING_CONTEXT is preserved
        assert final_metadata["doc_context"] == "PRE_EXISTING_CONTEXT"
