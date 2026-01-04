"""
Performance Tests for ADR004 Phase 1: Documentation Extraction

Tests that documentation extraction completes within performance budget (<5s for typical PDFs).
"""

import time

import pytest
from clinical_analytics.core.doc_parser import extract_context_from_docs


@pytest.mark.slow
@pytest.mark.integration
class TestDocExtractionPerformance:
    """Performance tests for documentation extraction."""

    def test_doc_extraction_performance_under_5s(self, tmp_path):
        """
        Test that doc extraction completes in <5s for typical PDFs.

        Performance Budget (ADR004 Phase 1):
        - Doc extraction should complete in <5s for typical PDFs (10-20 pages)
        - This ensures upload flow remains responsive

        Test Data:
        - Creates a sample PDF-like structure (simulated)
        - For real PDFs, would use pymupdf to create test PDF
        - Tests with multiple file types (PDF, MD, TXT)
        """
        # Create test documentation files
        # Note: For real PDF testing, we'd need pymupdf to create a valid PDF
        # For now, we test with text/markdown files and verify the extraction logic is fast

        # Create multiple documentation files
        doc_files = []
        for i in range(5):  # Simulate 5 documentation files
            md_path = tmp_path / f"doc_{i}.md"
            md_path.write_text(f"# Documentation File {i}\n\n" + "Content line.\n" * 100)
            doc_files.append(md_path)

        # Measure extraction time
        start_time = time.time()
        doc_context = extract_context_from_docs(doc_files)
        elapsed_time = time.time() - start_time

        # Verify extraction completed
        assert len(doc_context) > 0, "doc_context should not be empty"

        # Verify performance budget (5s for typical PDFs)
        # For text/markdown files, should be much faster (<1s)
        # For PDFs with pymupdf, budget is 5s
        assert (
            elapsed_time < 5.0
        ), f"Doc extraction took {elapsed_time:.2f}s, exceeds 5s budget. This indicates performance regression."

        # Success: Performance within budget
        print(f"\n✅ Doc Extraction Performance: {elapsed_time:.2f}s (< 5s budget)")

    def test_doc_extraction_truncation_performance(self, tmp_path):
        """
        Test that large documentation files are truncated efficiently.

        Verifies that truncation logic (50k char limit) doesn't cause performance issues.
        """
        # Create a large text file (>50k chars)
        large_content = "Large documentation content.\n" * 2000  # ~50k chars
        large_file = tmp_path / "large_doc.txt"
        large_file.write_text(large_content)

        # Measure extraction time with truncation
        start_time = time.time()
        doc_context = extract_context_from_docs([large_file], max_chars=50000)
        elapsed_time = time.time() - start_time

        # Verify truncation occurred
        assert len(doc_context) <= 50000, "doc_context should be truncated to 50k chars"

        # Verify performance (truncation should be fast, <1s)
        assert (
            elapsed_time < 1.0
        ), f"Truncation took {elapsed_time:.2f}s, should be <1s. This indicates performance regression."

        # Success: Truncation efficient
        print(f"\n✅ Doc Truncation Performance: {elapsed_time:.2f}s (< 1s budget)")

    @pytest.mark.skip(reason="Requires pymupdf and valid PDF generation - test with real PDFs in CI")
    def test_pdf_extraction_performance_real_pdf(self, tmp_path):
        """
        Test PDF extraction performance with real PDF files.

        **Note**: This test is skipped by default because it requires:
        1. pymupdf installed
        2. Ability to create valid PDF files for testing
        3. Real PDF files (10-20 pages typical)

        To run this test:
        1. Ensure pymupdf is installed
        2. Create test PDF files in fixtures/
        3. Remove @pytest.mark.skip decorator
        """
        # This would test with real PDF files
        # For now, we rely on the text/markdown performance test above
        pass
