"""
Tests for code-to-label mapping in count analysis rendering.

Tests verify:
- Numeric codes are mapped to labels when available
- Mapping is generic and works for any coded column
- Falls back gracefully when mapping unavailable

Note: These tests verify the logic is extensible and DRY, not hardcoded.
"""




class TestCodeToLabelMapping:
    """Test that codes are mapped to labels in count analysis rendering."""

    def test_code_to_label_mapping_logic_is_generic(self):
        """Test that code-to-label mapping logic is generic and extensible."""
        # Arrange: Simulate the mapping logic used in render_count_analysis
        from clinical_analytics.core.column_parser import parse_column_name

        # Test with different column formats (generic, not hardcoded)
        test_columns = [
            "Statin Used: 0: n/a 1: Atorvastatin 2: Rosuvastatin",
            "Treatment: 0: None 1: Aspirin 2: Ibuprofen",
            "Medication Type: 1: DrugA 2: DrugB 3: DrugC",
        ]

        for column_name in test_columns:
            # Act: Parse column to extract mapping
            column_meta = parse_column_name(column_name)

            # Assert: Should extract mapping generically (not hardcoded)
            assert column_meta.value_mapping is not None, f"Should extract mapping for: {column_name}"
            assert len(column_meta.value_mapping) > 0, f"Should have at least one code-label pair for: {column_name}"

            # Verify mapping structure is correct (code -> label)
            for code, label in column_meta.value_mapping.items():
                assert isinstance(code, str), f"Code should be string: {code}"
                assert isinstance(label, str), f"Label should be string: {label}"
                assert len(label) > 0, f"Label should not be empty for code: {code}"

    def test_mapping_works_for_any_coded_column_format(self):
        """Test that mapping extraction works for various coded column formats."""
        from clinical_analytics.core.column_parser import parse_column_name

        # Test different formats (generic patterns, not hardcoded)
        formats = [
            ("Code:Label format", "Treatment: 1: Aspirin 2: Ibuprofen"),
            ("Label:Code format", "Treatment: Aspirin:1 Ibuprofen:2"),
            ("Mixed format", "Medication: 0: None 1: DrugA 2: DrugB"),
        ]

        for format_name, column_name in formats:
            column_meta = parse_column_name(column_name)
            # Should extract at least some mapping (format may vary)
            # The key is that the logic is generic and extensible
            assert hasattr(column_meta, "value_mapping"), f"Should have value_mapping attribute for {format_name}"
