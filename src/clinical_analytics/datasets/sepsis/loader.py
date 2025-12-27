import logging
import polars as pl
from pathlib import Path
from typing import Generator, Optional
from clinical_analytics.core.mapper import ColumnMapper

logger = logging.getLogger(__name__)

def find_psv_files(root_path: Path) -> Generator[Path, None, None]:
    """Recursively find all .psv files in the directory."""
    return root_path.rglob("*.psv")

def load_patient_file(path: Path) -> pl.DataFrame:
    """Read a single patient PSV file using Polars."""
    return pl.read_csv(path, separator='|')

def load_and_aggregate(
    root_path: Path,
    mapper: Optional[ColumnMapper] = None,
    limit: Optional[int] = None
) -> pl.DataFrame:
    """
    Load sepsis data from PSV files using Polars for efficient aggregation.

    Now config-driven: Aggregation logic comes from mapper config.
    Since the dataset is time-series (hourly rows per patient),
    we aggregate to patient-level for the unified cohort.

    Args:
        root_path: Path to directory containing PSV files
        mapper: ColumnMapper instance for config-driven aggregation
        limit: Optional limit on number of files to process

    Returns:
        Aggregated DataFrame with patient-level data
    """
    psv_files = list(find_psv_files(root_path))

    if not psv_files:
        raise FileNotFoundError(f"No .psv files found in {root_path}")

    logger.info(f"Found {len(psv_files)} patient files. Processing...")

    # Collect all time-series data with patient_id from filename
    all_data = []
    count = 0

    for psv in psv_files:
        if limit and count >= limit:
            break

        try:
            df = load_patient_file(psv)
            patient_id = psv.stem  # Filename is usually patient ID (e.g. p00001)
            
            # Add patient_id to each row
            df = df.with_columns([pl.lit(patient_id).alias('patient_id')])
            all_data.append(df)
            count += 1

        except Exception as e:
            logger.error(f"Error processing {psv}: {e}")

    if not all_data:
        return pl.DataFrame()

    # Concatenate all patient data
    combined_df = pl.concat(all_data)

    # Apply config-driven aggregation using mapper
    if mapper:
        aggregated_df = mapper.apply_aggregations(combined_df, group_by='patient_id')
    else:
        # Fallback to basic aggregation if no mapper provided
        aggregated_df = combined_df.group_by('patient_id').agg([
            pl.col('Age').first().alias('age') if 'Age' in combined_df.columns else pl.lit(None).alias('age'),
            pl.col('Gender').first().alias('gender') if 'Gender' in combined_df.columns else pl.lit(None).alias('gender'),
            pl.col('SepsisLabel').max().alias('sepsis_label') if 'SepsisLabel' in combined_df.columns else pl.lit(0).alias('sepsis_label'),
            pl.len().alias('num_hours')
        ])

    return aggregated_df

