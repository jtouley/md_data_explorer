"""
Tests for PDF upload UI component in upload page.

Tests ensure that:
1. PDF upload component is visible in render_upload_step()
2. Session state is correctly managed when PDF is uploaded
3. External PDF bytes are passed to metadata during save
"""

from pathlib import Path


class TestPdfUploadUI:
    """Test suite for PDF upload UI component."""

    def test_pdf_upload_component_exists_in_upload_page(self):
        """Test that PDF upload component exists in render_upload_step()."""
        # Arrange: Read upload page
        upload_page = Path("src/clinical_analytics/ui/pages/01_📤_Add_Your_Data.py")
        assert upload_page.exists(), "Upload page should exist"

        with open(upload_page) as f:
            content = f.read()

        # Assert: PDF upload component exists
        assert 'key="doc_uploader"' in content, "PDF upload component (doc_uploader) should exist"
        assert "st.file_uploader" in content, "PDF upload should use st.file_uploader"

        # Assert: Component accepts correct file types
        assert 'type=["pdf", "txt", "md"]' in content or 'type=["pdf","txt","md"]' in content, (
            "PDF upload should accept PDF, TXT, and MD files"
        )

    def test_pdf_upload_sets_session_state_when_file_uploaded(self):
        """Test that PDF upload sets session state correctly when file is uploaded."""
        # Arrange: Read upload page
        upload_page = Path("src/clinical_analytics/ui/pages/01_📤_Add_Your_Data.py")

        with open(upload_page) as f:
            content = f.read()

        # Assert: Session state is set when file is uploaded
        assert "external_pdf_bytes" in content, "PDF upload should set external_pdf_bytes in session state"
        assert "external_pdf_filename" in content, "PDF upload should set external_pdf_filename in session state"
        assert 'st.session_state["external_pdf_bytes"]' in content, "PDF upload should store bytes in session state"
        assert 'st.session_state["external_pdf_filename"]' in content, (
            "PDF upload should store filename in session state"
        )

    def test_pdf_upload_clears_session_state_when_removed(self):
        """Test that PDF upload clears session state when file is removed."""
        # Arrange: Read upload page
        upload_page = Path("src/clinical_analytics/ui/pages/01_📤_Add_Your_Data.py")

        with open(upload_page) as f:
            content = f.read()

        # Assert: Session state is cleared when file is removed
        assert 'st.session_state.pop("external_pdf_bytes"' in content, (
            "PDF upload should clear external_pdf_bytes when file removed"
        )
        assert 'st.session_state.pop("external_pdf_filename"' in content, (
            "PDF upload should clear external_pdf_filename when file removed"
        )

    def test_pdf_upload_component_visible_before_data_file_upload(self):
        """Test that PDF upload component is visible even before data file is uploaded."""
        # Arrange: Read upload page
        upload_page = Path("src/clinical_analytics/ui/pages/01_📤_Add_Your_Data.py")

        with open(upload_page) as f:
            content = f.read()
            lines = content.split("\n")

        # Find where doc_uploader is defined
        doc_uploader_line_idx = None
        for i, line in enumerate(lines):
            if 'key="doc_uploader"' in line:
                doc_uploader_line_idx = i
                break

        assert doc_uploader_line_idx is not None, "PDF upload component should exist"

        # Find where main file uploader is defined
        main_uploader_line_idx = None
        for i, line in enumerate(lines):
            if 'key="file_uploader"' in line:
                main_uploader_line_idx = i
                break

        assert main_uploader_line_idx is not None, "Main file uploader should exist"

        # Assert: PDF upload component appears after main uploader (not nested in validation)
        # The component should be visible before the "if uploaded_file is not None" check
        uploaded_file_check_idx = None
        for i, line in enumerate(lines):
            if "if uploaded_file is not None:" in line:
                uploaded_file_check_idx = i
                break

        assert uploaded_file_check_idx is not None, "Main file upload check should exist"

        # Assert: doc_uploader is defined before the uploaded_file check
        # This ensures it's visible regardless of whether a file is uploaded
        assert doc_uploader_line_idx < uploaded_file_check_idx, (
            "PDF upload component should be visible before data file validation (not nested inside uploaded_file check)"
        )

    def test_pdf_upload_passed_to_metadata_during_save(self):
        """Test that external PDF bytes are passed to metadata during save operations."""
        # Arrange: Read upload page
        upload_page = Path("src/clinical_analytics/ui/pages/01_📤_Add_Your_Data.py")

        with open(upload_page) as f:
            content = f.read()

        # Assert: External PDF is added to metadata
        assert 'metadata["external_pdf_bytes"]' in content, "External PDF bytes should be added to metadata during save"
        assert 'metadata["external_pdf_filename"]' in content, (
            "External PDF filename should be added to metadata during save"
        )
        assert '"external_pdf_bytes" in st.session_state' in content, (
            "Code should check if external PDF exists before adding to metadata"
        )
