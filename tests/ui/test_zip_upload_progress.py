"""
Tests for ZIP upload progress callback functionality.
"""

import io
import zipfile

from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage


class TestZipUploadProgress:
    """Test suite for progress callback in ZIP upload processing."""

    def test_save_zip_upload_calls_progress_callback(self, tmp_path):
        """Test that progress callback is called during ZIP upload."""
        # Create test ZIP file (must be >= 1KB)
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            # Create larger CSV files to meet 1KB minimum
            patients_data = "patient_id,age\n" + "\n".join([f"P{i:03d},{20 + i}" for i in range(50)])
            admissions_data = "patient_id,date\n" + "\n".join([f"P{i:03d},2020-01-{1 + i % 30:02d}" for i in range(50)])
            zip_file.writestr("patients.csv", patients_data)
            zip_file.writestr("admissions.csv", admissions_data)

        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()

        # Track progress calls
        progress_calls = []

        def progress_callback(step: int, total_steps: int, message: str, details: dict):
            progress_calls.append({"step": step, "total_steps": total_steps, "message": message, "details": details})

        storage = UserDatasetStorage(upload_dir=tmp_path)
        success, message, upload_id = storage.save_zip_upload(
            file_bytes=zip_bytes,
            original_filename="test.zip",
            metadata={"dataset_name": "test"},
            progress_callback=progress_callback,
        )

        if not success:
            print(f"Upload failed: {message}")
        assert success is True, f"Upload failed: {message}"
        assert len(progress_calls) > 0

        # Verify initial progress call
        assert any("Initializing" in call["message"] for call in progress_calls)

    def test_progress_callback_receives_table_loading_updates(self, tmp_path):
        """Test that progress callback receives updates for each table being loaded."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            # Create larger CSV files to meet 1KB minimum
            patients_data = "patient_id,age\n" + "\n".join([f"P{i:03d},{20 + i}" for i in range(50)])
            admissions_data = "patient_id,date\n" + "\n".join([f"P{i:03d},2020-01-{1 + i % 30:02d}" for i in range(50)])
            diagnoses_data = "patient_id,code\n" + "\n".join([f"P{i:03d},E11.9" for i in range(50)])
            zip_file.writestr("patients.csv", patients_data)
            zip_file.writestr("admissions.csv", admissions_data)
            zip_file.writestr("diagnoses.csv", diagnoses_data)

        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()

        table_loading_calls = []

        def progress_callback(step: int, total_steps: int, message: str, details: dict):
            if "table_name" in details:
                table_loading_calls.append(
                    {
                        "table_name": details["table_name"],
                        "rows": details.get("rows"),
                        "cols": details.get("cols"),
                        "message": message,
                    }
                )

        storage = UserDatasetStorage(upload_dir=tmp_path)
        success, _, _ = storage.save_zip_upload(
            file_bytes=zip_bytes,
            original_filename="test.zip",
            metadata={"dataset_name": "test"},
            progress_callback=progress_callback,
        )

        assert success is True
        # Should have received updates for each table
        assert len(table_loading_calls) >= 3  # At least one per table

        # Verify table names are in the calls
        table_names = {call["table_name"] for call in table_loading_calls}
        assert "patients" in table_names
        assert "admissions" in table_names
        assert "diagnoses" in table_names

    def test_progress_callback_receives_relationship_detection_update(self, tmp_path):
        """Test that progress callback receives relationship detection updates."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            # Create larger CSV files to meet 1KB minimum
            patients_data = "patient_id,age\n" + "\n".join([f"P{i:03d},{20 + i}" for i in range(50)])
            admissions_data = "patient_id,date\n" + "\n".join([f"P{i:03d},2020-01-{1 + i % 30:02d}" for i in range(50)])
            zip_file.writestr("patients.csv", patients_data)
            zip_file.writestr("admissions.csv", admissions_data)

        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()

        relationship_calls = []

        def progress_callback(step: int, total_steps: int, message: str, details: dict):
            if "relationships" in details:
                relationship_calls.append({"relationships": details["relationships"], "message": message})

        storage = UserDatasetStorage(upload_dir=tmp_path)
        success, _, _ = storage.save_zip_upload(
            file_bytes=zip_bytes,
            original_filename="test.zip",
            metadata={"dataset_name": "test"},
            progress_callback=progress_callback,
        )

        assert success is True
        # Should have received relationship detection update
        assert len(relationship_calls) > 0
        assert any("relationships" in call for call in relationship_calls)

    def test_progress_callback_without_callback_works(self, tmp_path):
        """Test that save_zip_upload works without progress callback."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            # Create larger CSV file to meet 1KB minimum (need more rows)
            patients_data = "patient_id,age\n" + "\n".join([f"P{i:03d},{20 + i}" for i in range(150)])
            zip_file.writestr("patients.csv", patients_data)

        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()

        storage = UserDatasetStorage(upload_dir=tmp_path)
        success, message, upload_id = storage.save_zip_upload(
            file_bytes=zip_bytes,
            original_filename="test.zip",
            metadata={"dataset_name": "test"},
            # No progress_callback provided
        )

        if not success:
            print(f"Upload failed: {message}")
        assert success is True, f"Upload failed: {message}"
        assert upload_id is not None

    def test_progress_callback_receives_correct_step_counts(self, tmp_path):
        """Test that progress callback receives correct step and total_steps values."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            # Create larger CSV files to meet 1KB minimum
            patients_data = "patient_id,age\n" + "\n".join([f"P{i:03d},{20 + i}" for i in range(50)])
            admissions_data = "patient_id,date\n" + "\n".join([f"P{i:03d},2020-01-{1 + i % 30:02d}" for i in range(50)])
            zip_file.writestr("patients.csv", patients_data)
            zip_file.writestr("admissions.csv", admissions_data)

        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()

        step_values = []

        def progress_callback(step: int, total_steps: int, message: str, details: dict):
            step_values.append((step, total_steps))
            # Verify step is always <= total_steps
            assert step <= total_steps, f"Step {step} exceeds total_steps {total_steps}"

        storage = UserDatasetStorage(upload_dir=tmp_path)
        success, _, _ = storage.save_zip_upload(
            file_bytes=zip_bytes,
            original_filename="test.zip",
            metadata={"dataset_name": "test"},
            progress_callback=progress_callback,
        )

        assert success is True
        assert len(step_values) > 0

        # Verify steps are monotonically increasing
        steps = [s[0] for s in step_values]
        assert steps == sorted(steps), "Steps should be monotonically increasing"

        # Verify all calls use same total_steps
        total_steps_values = {s[1] for s in step_values}
        assert len(total_steps_values) == 1, "All calls should use same total_steps"

    def test_progress_callback_receives_table_details(self, tmp_path):
        """Test that progress callback receives detailed table information."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            # Create larger CSV file to meet 1KB minimum (need more rows)
            patients_data = "patient_id,age,sex\n" + "\n".join(
                [f"P{i:03d},{20 + i},{['M', 'F'][i % 2]}" for i in range(100)]
            )
            zip_file.writestr("patients.csv", patients_data)

        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()

        table_details = []

        def progress_callback(step: int, total_steps: int, message: str, details: dict):
            if "table_name" in details and "rows" in details:
                table_details.append(details)

        storage = UserDatasetStorage(upload_dir=tmp_path)
        success, _, _ = storage.save_zip_upload(
            file_bytes=zip_bytes,
            original_filename="test.zip",
            metadata={"dataset_name": "test"},
            progress_callback=progress_callback,
        )

        assert success is True
        assert len(table_details) > 0

        # Verify table details structure
        for details in table_details:
            assert "table_name" in details
            assert "rows" in details
            assert "cols" in details
            assert isinstance(details["rows"], int)
            assert isinstance(details["cols"], int)
            assert details["rows"] > 0
            assert details["cols"] > 0
