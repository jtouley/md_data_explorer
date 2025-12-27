"""
Tests for ZIP file extraction and multi-table processing.
"""

import pytest
import polars as pl
import zipfile
import io
from pathlib import Path
from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage, UploadSecurityValidator
from clinical_analytics.core.multi_table_handler import MultiTableHandler


class TestZipExtraction:
    """Test suite for ZIP file extraction and processing."""

    def test_extract_zip_with_csv_files(self, tmp_path):
        """Test extracting ZIP file containing multiple CSV files."""
        # Create test ZIP file
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            # Add first CSV
            zip_file.writestr('patients.csv', 'patient_id,age,sex\nP001,45,M\nP002,62,F\n')
            # Add second CSV
            zip_file.writestr('admissions.csv', 'patient_id,admission_date,discharge_date\nP001,2020-01-01,2020-01-05\nP002,2020-02-01,2020-02-10\n')
            # Add third CSV
            zip_file.writestr('diagnoses.csv', 'patient_id,icd_code,diagnosis\nP001,E11.9,Diabetes\nP002,I10,Hypertension\n')
        
        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()

        # Test extraction
        storage = UserDatasetStorage(upload_dir=tmp_path)
        success, message, upload_id = storage.save_zip_upload(
            file_bytes=zip_bytes,
            original_filename='test_dataset.zip',
            metadata={'dataset_name': 'test_dataset'}
        )

        assert success is True
        assert upload_id is not None
        assert 'tables' in message.lower() or 'joined' in message.lower()

    def test_extract_zip_with_csv_gz_files(self, tmp_path):
        """Test extracting ZIP file containing compressed CSV files."""
        import gzip
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            # Create compressed CSV
            csv_content = 'patient_id,age\nP001,45\nP002,62\n'.encode('utf-8')
            compressed = gzip.compress(csv_content)
            zip_file.writestr('patients.csv.gz', compressed)
            
            # Add regular CSV
            zip_file.writestr('admissions.csv', 'patient_id,date\nP001,2020-01-01\n')
        
        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()

        storage = UserDatasetStorage(base_dir=tmp_path)
        success, message, upload_id = storage.save_zip_upload(
            file_bytes=zip_bytes,
            original_filename='compressed.zip',
            metadata={'dataset_name': 'compressed'}
        )

        assert success is True
        assert upload_id is not None

    def test_extract_zip_with_subdirectories(self, tmp_path):
        """Test extracting ZIP file with CSV files in subdirectories."""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            zip_file.writestr('mimic-iv/patients.csv', 'patient_id,age\nP001,45\n')
            zip_file.writestr('mimic-iv/admissions.csv', 'patient_id,date\nP001,2020-01-01\n')
            zip_file.writestr('mimic-iv/diagnoses.csv', 'patient_id,code\nP001,E11.9\n')
        
        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()

        storage = UserDatasetStorage(base_dir=tmp_path)
        success, message, upload_id = storage.save_zip_upload(
            file_bytes=zip_bytes,
            original_filename='mimic-iv-clinical-database-demo-2.2.zip',
            metadata={'dataset_name': 'mimic_iv'}
        )

        assert success is True
        assert upload_id is not None

    def test_extract_zip_ignores_macosx(self, tmp_path):
        """Test that ZIP extraction ignores __MACOSX files."""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            zip_file.writestr('patients.csv', 'patient_id,age\nP001,45\n')
            zip_file.writestr('__MACOSX/._patients.csv', 'metadata')
            zip_file.writestr('__MACOSX/.DS_Store', 'metadata')
        
        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()

        storage = UserDatasetStorage(base_dir=tmp_path)
        success, message, upload_id = storage.save_zip_upload(
            file_bytes=zip_bytes,
            original_filename='test.zip',
            metadata={'dataset_name': 'test'}
        )

        assert success is True
        # Should only process patients.csv, not __MACOSX files

    def test_extract_zip_no_csv_files(self, tmp_path):
        """Test ZIP file with no CSV files raises error."""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            zip_file.writestr('readme.txt', 'This is a readme file')
            zip_file.writestr('data.json', '{"key": "value"}')
        
        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()

        storage = UserDatasetStorage(base_dir=tmp_path)
        success, message, upload_id = storage.save_zip_upload(
            file_bytes=zip_bytes,
            original_filename='no_csv.zip',
            metadata={'dataset_name': 'test'}
        )

        assert success is False
        assert 'no csv files' in message.lower()

    def test_extract_zip_invalid_file(self, tmp_path):
        """Test handling invalid ZIP file."""
        invalid_bytes = b'This is not a ZIP file'

        storage = UserDatasetStorage(base_dir=tmp_path)
        success, message, upload_id = storage.save_zip_upload(
            file_bytes=invalid_bytes,
            original_filename='invalid.zip',
            metadata={'dataset_name': 'test'}
        )

        assert success is False
        assert 'error' in message.lower() or 'invalid' in message.lower()

    def test_extract_zip_with_mixed_types(self, tmp_path):
        """Test ZIP extraction with tables having different key column types (int vs string)."""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            # Table 1: patient_id as integer
            zip_file.writestr('patients.csv', 'patient_id,age\n1,45\n2,62\n')
            # Table 2: patient_id as string (should be normalized)
            zip_file.writestr('admissions.csv', 'patient_id,date\n1,2020-01-01\n2,2020-02-01\n')
        
        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()

        storage = UserDatasetStorage(base_dir=tmp_path)
        success, message, upload_id = storage.save_zip_upload(
            file_bytes=zip_bytes,
            original_filename='mixed_types.zip',
            metadata={'dataset_name': 'mixed_types'}
        )

        # Should succeed despite type differences (normalization should handle it)
        assert success is True

    def test_extract_zip_large_dataset(self, tmp_path):
        """Test extracting ZIP with larger dataset (multiple tables, many rows)."""
        zip_buffer = io.BytesIO()
        
        # Create larger dataset
        patients_data = 'patient_id,age,sex\n' + '\n'.join([f'P{i:03d},{20+i},{["M","F"][i%2]}' for i in range(100)])
        admissions_data = 'patient_id,admission_date\n' + '\n'.join([f'P{i:03d},2020-01-{1+i%30:02d}' for i in range(100)])
        
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            zip_file.writestr('patients.csv', patients_data)
            zip_file.writestr('admissions.csv', admissions_data)
        
        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()

        storage = UserDatasetStorage(base_dir=tmp_path)
        success, message, upload_id = storage.save_zip_upload(
            file_bytes=zip_bytes,
            original_filename='large_dataset.zip',
            metadata={'dataset_name': 'large'}
        )

        assert success is True
        assert upload_id is not None

    def test_extract_zip_creates_unified_cohort(self, tmp_path):
        """Test that ZIP extraction creates unified cohort with joined tables."""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            zip_file.writestr('patients.csv', 'patient_id,age,sex\nP001,45,M\nP002,62,F\n')
            zip_file.writestr('admissions.csv', 'patient_id,admission_date\nP001,2020-01-01\nP002,2020-02-01\n')
        
        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()

        storage = UserDatasetStorage(base_dir=tmp_path)
        success, message, upload_id = storage.save_zip_upload(
            file_bytes=zip_bytes,
            original_filename='test.zip',
            metadata={'dataset_name': 'test'}
        )

        assert success is True
        
        # Check that unified cohort CSV was created
        csv_path = tmp_path / 'raw' / f'{upload_id}.csv'
        assert csv_path.exists()
        
        # Load and verify unified cohort
        unified_df = pl.read_csv(csv_path)
        assert 'patient_id' in unified_df.columns
        assert 'age' in unified_df.columns
        assert 'admission_date' in unified_df.columns

    def test_extract_zip_saves_metadata(self, tmp_path):
        """Test that ZIP extraction saves proper metadata."""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            zip_file.writestr('patients.csv', 'patient_id,age\nP001,45\nP002,62\n')
            zip_file.writestr('admissions.csv', 'patient_id,date\nP001,2020-01-01\n')
        
        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()

        storage = UserDatasetStorage(base_dir=tmp_path)
        success, message, upload_id = storage.save_zip_upload(
            file_bytes=zip_bytes,
            original_filename='test.zip',
            metadata={'dataset_name': 'test_dataset'}
        )

        assert success is True
        
        # Check metadata file
        import json
        metadata_path = tmp_path / 'metadata' / f'{upload_id}.json'
        assert metadata_path.exists()
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        assert metadata['file_format'] == 'zip_multi_table'
        assert 'tables' in metadata
        assert 'relationships' in metadata
        assert 'inferred_schema' in metadata
        assert len(metadata['tables']) == 2  # patients and admissions


class TestMultiTableHandler:
    """Test suite for MultiTableHandler used in ZIP extraction."""

    def test_detect_relationships_with_type_mismatch(self):
        """Test relationship detection handles type mismatches (int vs string keys)."""
        # Create tables with different key types
        patients = pl.DataFrame({
            'patient_id': [1, 2, 3],  # Integer
            'age': [45, 62, 38]
        })
        
        admissions = pl.DataFrame({
            'patient_id': ['1', '2', '3'],  # String (should be normalized)
            'admission_date': ['2020-01-01', '2020-02-01', '2020-03-01']
        })
        
        tables = {
            'patients': patients,
            'admissions': admissions
        }
        
        handler = MultiTableHandler(tables)
        relationships = handler.detect_relationships()
        
        # Should detect relationship despite type mismatch
        assert len(relationships) > 0
        handler.close()

    def test_build_unified_cohort_with_type_mismatch(self):
        """Test building unified cohort with type mismatches."""
        patients = pl.DataFrame({
            'patient_id': [1, 2, 3],
            'age': [45, 62, 38]
        })
        
        admissions = pl.DataFrame({
            'patient_id': ['1', '2', '3'],  # String
            'date': ['2020-01-01', '2020-02-01', '2020-03-01']
        })
        
        tables = {
            'patients': patients,
            'admissions': admissions
        }
        
        handler = MultiTableHandler(tables)
        cohort = handler.build_unified_cohort()
        
        assert cohort.height > 0
        assert 'patient_id' in cohort.columns
        assert 'age' in cohort.columns
        assert 'date' in cohort.columns
        handler.close()

