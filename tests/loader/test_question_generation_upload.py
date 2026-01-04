"""
Tests for upload-time question generation integration (ADR004 Phase 4).

Tests verify that generate_upload_questions() is integrated into save_table_list()
and example questions are stored in metadata.
"""

import polars as pl
from clinical_analytics.ui.storage.user_datasets import save_table_list


class TestUploadQuestionGenerationIntegration:
    """Test suite for upload-time question generation in save_table_list()."""

    def test_save_table_list_stores_example_questions_in_metadata(self, upload_storage):
        """Test that save_table_list() generates and stores example_questions in metadata."""
        # Arrange: Create storage and tables
        storage = upload_storage
        upload_id = storage.generate_upload_id("test_dataset.csv")

        df = pl.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(100)],
                "age": [20 + (i % 50) for i in range(100)],
                "outcome": [i % 2 for i in range(100)],
                "treatment": ["A", "B", "A", "B"] * 25,
            }
        )

        tables = [{"name": "patients", "data": df}]
        metadata = {
            "dataset_name": "test_dataset",
            "table_count": 1,
            "table_names": ["patients"],
        }

        # Act: Save tables (should generate questions)
        success, message = save_table_list(storage, tables, upload_id, metadata)

        # Assert: Save succeeded
        assert success is True, f"save_table_list failed: {message}"

        # Assert: Metadata contains example_questions
        saved_metadata = storage.get_upload_metadata(upload_id)
        assert "example_questions" in saved_metadata, "example_questions not found in metadata"
        assert isinstance(saved_metadata["example_questions"], list)
        assert len(saved_metadata["example_questions"]) > 0, "example_questions should not be empty"
        assert all(isinstance(q, str) for q in saved_metadata["example_questions"])

    def test_save_table_list_skips_question_generation_if_already_exists(self, upload_storage):
        """Test that save_table_list() skips question generation if example_questions already exists (idempotency)."""
        # Arrange: Create storage and tables
        storage = upload_storage
        upload_id = storage.generate_upload_id("test_dataset.csv")

        df = pl.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(100)],
                "age": [20 + (i % 50) for i in range(100)],
            }
        )

        tables = [{"name": "patients", "data": df}]
        metadata = {
            "dataset_name": "test_dataset",
            "example_questions": ["PRE_EXISTING_QUESTION_1", "PRE_EXISTING_QUESTION_2"],
        }

        # Act: Save tables (should preserve existing questions)
        success, _ = save_table_list(storage, tables, upload_id, metadata)

        # Assert: Save succeeded
        assert success is True

        # Assert: Existing questions preserved
        saved_metadata = storage.get_upload_metadata(upload_id)
        assert "example_questions" in saved_metadata
        assert saved_metadata["example_questions"] == ["PRE_EXISTING_QUESTION_1", "PRE_EXISTING_QUESTION_2"]

    def test_save_table_list_generates_questions_with_doc_context(self, upload_storage):
        """Test that question generation uses doc_context when available."""
        # Arrange: Create storage with doc_context
        storage = upload_storage
        upload_id = storage.generate_upload_id("test_dataset.csv")

        df = pl.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(100)],
                "age": [20 + (i % 50) for i in range(100)],
                "treatment": [1, 2, 1, 2] * 25,  # Coded column
            }
        )

        tables = [{"name": "patients", "data": df}]
        metadata = {
            "dataset_name": "test_dataset",
            "doc_context": "treatment: Treatment group\n1: Control, 2: Treatment",
        }

        # Act: Save tables
        success, _ = save_table_list(storage, tables, upload_id, metadata)

        # Assert: Save succeeded and questions generated
        assert success is True
        saved_metadata = storage.get_upload_metadata(upload_id)
        assert "example_questions" in saved_metadata
        assert len(saved_metadata["example_questions"]) > 0
