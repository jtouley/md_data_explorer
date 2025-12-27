#!/usr/bin/env python3
"""
Test script to debug ZIP upload with logging enabled.
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage
import tempfile

def test_zip_upload():
    """Test ZIP upload with MIMIC-IV demo file."""
    zip_path = Path("data/raw/mimic/mimic-iv-clinical-database-demo-2.2.zip")
    
    if not zip_path.exists():
        print(f"ERROR: ZIP file not found at {zip_path}")
        print("Please ensure the MIMIC-IV demo ZIP file exists at that location")
        return
    
    print(f"Testing ZIP upload with: {zip_path}")
    print(f"File size: {zip_path.stat().st_size / (1024*1024):.2f} MB")
    print("-" * 80)
    
    # Read ZIP file
    with open(zip_path, 'rb') as f:
        zip_bytes = f.read()
    
    # Create temporary directory for uploads
    with tempfile.TemporaryDirectory() as tmp_dir:
        storage = UserDatasetStorage(upload_dir=Path(tmp_dir))
        
        print(f"Upload directory: {tmp_dir}")
        print("-" * 80)
        
        try:
            success, message, upload_id = storage.save_zip_upload(
                file_bytes=zip_bytes,
                original_filename=zip_path.name,
                metadata={'dataset_name': 'mimic_iv_demo'}
            )
            
            print("-" * 80)
            if success:
                print(f"✅ SUCCESS: {message}")
                print(f"Upload ID: {upload_id}")
            else:
                print(f"❌ FAILED: {message}")
                print(f"Upload ID: {upload_id}")
                
        except Exception as e:
            print("-" * 80)
            print(f"❌ EXCEPTION: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_zip_upload()

