import polars as pl
from pathlib import Path
from typing import Generator

def find_psv_files(root_path: Path) -> Generator[Path, None, None]:
    """Recursively find all .psv files in the directory."""
    return root_path.rglob("*.psv")

def load_patient_file(path: Path) -> pl.DataFrame:
    """Read a single patient PSV file using Polars."""
    return pl.read_csv(path, separator='|')

def load_and_aggregate(root_path: Path, limit: int = None) -> pl.DataFrame:
    """
    Load sepsis data from PSV files using Polars for efficient aggregation.

    Since the dataset is time-series (hourly rows per patient),
    we aggregate to patient-level for the unified cohort.

    Features:
    - SepsisLabel: Max (1 if ever septic)
    - Age: First value
    - Gender: First value
    - num_hours: Count of records
    """
    psv_files = list(find_psv_files(root_path))

    if not psv_files:
        raise FileNotFoundError(f"No .psv files found in {root_path}")

    print(f"Found {len(psv_files)} patient files. Processing...")

    records = []
    count = 0

    for psv in psv_files:
        if limit and count >= limit:
            break

        try:
            df = load_patient_file(psv)

            # Basic Aggregation using Polars
            patient_id = psv.stem  # Filename is usually patient ID (e.g. p00001)

            # Use Polars aggregation for efficient processing
            agg = df.select([
                pl.col('Age').first().alias('age') if 'Age' in df.columns else pl.lit(None).alias('age'),
                pl.col('Gender').first().alias('gender') if 'Gender' in df.columns else pl.lit(None).alias('gender'),
                pl.col('SepsisLabel').max().alias('sepsis_label') if 'SepsisLabel' in df.columns else pl.lit(0).alias('sepsis_label'),
                pl.len().alias('num_hours')
            ])

            # Extract values from aggregation
            row = agg.row(0, named=True)

            # Calculate onset_hour if patient had sepsis
            onset_hour = None
            if row['sepsis_label'] == 1 and 'SepsisLabel' in df.columns:
                sepsis_rows = df.filter(pl.col('SepsisLabel') == 1)
                if len(sepsis_rows) > 0:
                    onset_hour = sepsis_rows.select(pl.lit(0).cum_sum()).row(0)[0]

            records.append({
                'patient_id': patient_id,
                'age': row['age'],
                'gender': row['gender'],
                'sepsis_label': row['sepsis_label'],
                'onset_hour': onset_hour,
                'num_hours': row['num_hours']
            })

            count += 1

        except Exception as e:
            print(f"Error processing {psv}: {e}")

    return pl.DataFrame(records)

