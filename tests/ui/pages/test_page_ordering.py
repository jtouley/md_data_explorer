"""
Tests for page ordering in V1 MVP mode (Phase 6.2).

Ensures:
- Pages are ordered: Upload → Summary → Ask Questions
- Core pages come first (1-3)
- Gated pages come after (20-24)
"""

from pathlib import Path


class TestPageOrdering:
    """Test page ordering matches V1 MVP requirements."""

    def test_core_pages_ordered_upload_summary_ask(self):
        """Core pages should be ordered: Upload (01) → Summary (02) → Ask Questions (03)."""
        # Arrange: Get page files
        pages_dir = Path("src/clinical_analytics/ui/pages")

        upload_page = pages_dir / "01_📤_Add_Your_Data.py"
        summary_page = pages_dir / "02_📊_Your_Dataset.py"
        ask_page = pages_dir / "03_💬_Ask_Questions.py"

        # Assert: Files exist in correct order
        assert upload_page.exists(), "Upload page (01) should exist"
        assert summary_page.exists(), "Summary page (02) should exist"
        assert ask_page.exists(), "Ask Questions page (03) should exist"

    def test_gated_pages_come_after_core_pages(self):
        """Gated pages (20-24) should come after core pages (01-03)."""
        # Arrange: Get all page files
        pages_dir = Path("src/clinical_analytics/ui/pages")
        all_pages = sorted([p.name for p in pages_dir.glob("*.py") if not p.name.startswith("_")])

        # Act: Separate core and gated pages
        core_pages = [p for p in all_pages if p.startswith(("01_", "02_", "03_"))]
        gated_pages = [p for p in all_pages if p.startswith(("20", "21", "22", "23", "24"))]

        # Assert: All core pages come before all gated pages
        assert len(core_pages) == 3, "Should have 3 core pages"
        assert len(gated_pages) == 5, "Should have 5 gated pages"

        # Check first core page comes before first gated page
        first_core_index = all_pages.index(core_pages[0])
        first_gated_index = all_pages.index(gated_pages[0])
        assert first_core_index < first_gated_index, "Core pages should come before gated pages"

    def test_page_numbering_creates_correct_sidebar_order(self):
        """Page numbering should create correct sidebar ordering (01-03, then 20-24)."""
        # Arrange: Get page files
        pages_dir = Path("src/clinical_analytics/ui/pages")
        all_pages = sorted([p.name for p in pages_dir.glob("*.py") if not p.name.startswith("_")])

        # Assert: Order is correct
        # Streamlit sorts pages alphabetically, so we need:
        # 01_*.py, 02_*.py, 03_*.py, 20_*.py, 21_*.py, 22_*.py, 23_*.py, 24_*.py

        assert len(all_pages) == 8, f"Should have 8 pages, found {len(all_pages)}"

        # Verify core pages are first 3
        assert all_pages[0].startswith("01_"), "First page should be Upload (01_)"
        assert all_pages[1].startswith("02_"), "Second page should be Summary (02_)"
        assert all_pages[2].startswith("03_"), "Third page should be Ask Questions (03_)"
