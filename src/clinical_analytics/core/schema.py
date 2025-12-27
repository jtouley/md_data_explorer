"""
Unified Cohort Schema Definition.

Ensures all datasets output data in a harmonized format for the generic analysis layer.
"""


class UnifiedCohort:
    # Core Identifiers
    PATIENT_ID = "patient_id"  # String identifier

    # Temporal Anchors
    TIME_ZERO = "time_zero"  # Datetime of entry/infection/admission

    # Outcomes
    OUTCOME = "outcome"  # Binary (0/1) or Numeric target
    OUTCOME_LABEL = "outcome_label"  # String description (e.g., "mortality_30d", "sepsis_onset")

    # Flexible Features
    # Features can be flat columns in the dataframe, but this constant
    # reserves a name for a packed feature dictionary if needed.
    FEATURES_JSON = "features_json"

    REQUIRED_COLUMNS = [PATIENT_ID, TIME_ZERO, OUTCOME, OUTCOME_LABEL]
