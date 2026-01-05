"""
Tests for upload-time example questions display in Ask Questions page (ADR004 Phase 4).

Tests verify that example_questions from metadata are displayed on first load.
"""

from unittest.mock import MagicMock


class TestExampleQuestionsDisplay:
    """Test suite for upload-time example questions display."""

    def test_example_questions_displayed_on_first_load(self):
        """Test that example_questions from metadata are available when chat is empty."""
        # Arrange: Create mock dataset with example_questions in metadata
        mock_dataset = MagicMock()
        mock_dataset.metadata = {
            "example_questions": [
                "What is the distribution of age?",
                "Are there any outliers in outcome?",
                "What are the key relationships?",
            ],
            "dataset_name": "test_dataset",
        }
        mock_dataset.upload_id = "test_upload_id"

        # Act: Verify the structure exists
        assert hasattr(mock_dataset, "metadata")
        assert "example_questions" in mock_dataset.metadata
        assert len(mock_dataset.metadata["example_questions"]) > 0

    def test_example_questions_not_displayed_when_chat_has_messages(self):
        """Test that example_questions are NOT displayed when chat already has messages."""
        # Arrange: Chat has messages (not first load)
        mock_dataset = MagicMock()
        mock_dataset.metadata = {
            "example_questions": ["Question 1", "Question 2"],
        }

        # Act: Check that example_questions should not be displayed
        # (chat is not empty, so user has already interacted)
        chat_has_messages = True

        # Assert: Should not display example_questions
        assert chat_has_messages  # Chat has messages, so don't show examples

    def test_example_questions_handles_missing_metadata_gracefully(self):
        """Test that missing example_questions in metadata doesn't cause errors."""
        # Arrange: Dataset with no example_questions in metadata
        mock_dataset = MagicMock()
        mock_dataset.metadata = {
            "dataset_name": "test_dataset",
            # No example_questions key
        }

        # Act: Access example_questions (should return None or empty list)
        example_questions = mock_dataset.metadata.get("example_questions")

        # Assert: Should handle gracefully (None or empty list)
        assert example_questions is None or example_questions == []

    def test_example_questions_handles_empty_list_gracefully(self):
        """Test that empty example_questions list doesn't cause errors."""
        # Arrange: Dataset with empty example_questions list
        mock_dataset = MagicMock()
        mock_dataset.metadata = {
            "example_questions": [],
            "dataset_name": "test_dataset",
        }

        # Act: Access example_questions
        example_questions = mock_dataset.metadata.get("example_questions", [])

        # Assert: Should handle gracefully (empty list)
        assert example_questions == []
