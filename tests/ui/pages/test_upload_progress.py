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

    def test_progress_calculation_capped_at_one(self):
        """Test that progress calculation is capped at 1.0 to prevent StreamlitAPIException."""
        # Arrange: Read upload page
        upload_page = Path("src/clinical_analytics/ui/pages/01_📤_Add_Your_Data.py")

        with open(upload_page) as f:
            content = f.read()
            lines = content.split("\n")

        # Assert: Progress calculation caps at 1.0
        # The bug was: progress = step / total_steps (can exceed 1.0)
        # The fix should be: progress = min(progress, 1.0) on a subsequent line
        # OR: progress = min(step / total_steps, 1.0) on same line

        # Find the progress calculation line
        progress_calc_found = False
        progress_capped = False

        for i, line in enumerate(lines):
            if "progress = " in line and "total_steps" in line:
                progress_calc_found = True
                # Check if capped on same line
                if "min(" in line:
                    progress_capped = True
                    break
                # Check if capped on next line (within 3 lines)
                for j in range(i + 1, min(i + 4, len(lines))):
                    if "min(progress" in lines[j] or "progress = min(" in lines[j]:
                        progress_capped = True
                        break
                if progress_capped:
                    break

        assert progress_calc_found, "Progress calculation line should exist"
        assert progress_capped, (
            "Progress calculation should be capped at 1.0 using min(progress, 1.0) "
            "to prevent StreamlitAPIException when step > total_steps"
        )
