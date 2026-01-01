"""
Tests for upload progress indicators (Phase 6.3).

Ensures:
- Upload page shows real file processing progress
- Progress indicators exist (progress bar, status messages, logs)
- Progress callback is implemented
"""

from pathlib import Path


class TestUploadProgress:
    """Test upload page has real progress indicators."""

    def test_upload_page_has_progress_bar(self):
        """Upload page should have st.progress() for visual progress tracking."""
        # Arrange: Read upload page
        upload_page = Path("src/clinical_analytics/ui/pages/01_📤_Add_Your_Data.py")
        assert upload_page.exists(), "Upload page should exist"

        with open(upload_page) as f:
            content = f.read()

        # Assert: Page uses st.progress()
        assert "st.progress" in content, "Upload page should have progress bar (st.progress)"

    def test_upload_page_has_progress_callback(self):
        """Upload page should have progress_callback function for real-time updates."""
        # Arrange: Read upload page
        upload_page = Path("src/clinical_analytics/ui/pages/01_📤_Add_Your_Data.py")

        with open(upload_page) as f:
            content = f.read()

        # Assert: Page has progress_callback function
        assert "def progress_callback" in content, "Upload page should have progress_callback function"
        assert "progress_bar.progress" in content, "progress_callback should update progress bar"
        assert "status_text" in content, "progress_callback should show status messages"

    def test_upload_page_has_detailed_logging(self):
        """Upload page should log detailed processing information."""
        # Arrange: Read upload page
        upload_page = Path("src/clinical_analytics/ui/pages/01_📤_Add_Your_Data.py")

        with open(upload_page) as f:
            content = f.read()

        # Assert: Page has detailed logging
        assert "upload_log_messages" in content, "Upload page should store log messages"
        assert "table_name" in content, "Upload page should log table names"
        assert "relationships" in content, "Upload page should log relationship detection"

    def test_upload_page_uses_spinners(self):
        """Upload page should use st.spinner for async operations."""
        # Arrange: Read upload page
        upload_page = Path("src/clinical_analytics/ui/pages/01_📤_Add_Your_Data.py")

        with open(upload_page) as f:
            content = f.read()

        # Assert: Page uses spinners for quality checks and variable analysis
        assert "st.spinner" in content, "Upload page should use st.spinner for async operations"
        # Check for specific spinner messages (indicates real processing steps)
        assert "Running quality checks" in content or "quality checks" in content.lower(), (
            "Upload page should show quality check progress"
        )
