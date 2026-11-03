"""Constants for data contracts and data-quality checks (one place, each with its source).

Thresholds here are reporting thresholds: a check above one is shown to the user as a warning.
They never block an upload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from clinical_analytics.core.nl_query_config import SMALL_CELL_THRESHOLD

# ODCS v3.2.0 (Bitol). The open-data-contract-standard package pins the same schema version.
ODCS_API_VERSION: Final = "v3.2.0"
ODCS_KIND: Final = "DataContract"
CONTRACT_SUFFIX: Final = ".odcs.yaml"
CONTRACTS_DIRNAME: Final = "contracts"
DQ_DIRNAME: Final = "dq"
GENERATOR: Final = "clinical_analytics.contracts.builder"
# First contract for a dataset version. A later revision (accepted metadata patches) bumps it.
INITIAL_CONTRACT_VERSION: Final = "1.0.0"
UNIFIED_OBJECT_NAME: Final = "unified"
SYNTHETIC_TAG: Final = "synthetic"

# Counts 1..threshold-1 are shown as "<threshold" everywhere in the app (config_loader default 11).
MIN_CELL_SIZE: Final = SMALL_CELL_THRESHOLD

# Missingness above 5% is flagged. Jakobsen et al., BMC Med Res Methodol 2017;17:162
# (doi:10.1186/s12874-017-0442-1): complete case analysis "may be used as the primary analysis if
# the proportions of missing data are below approximately 5%"; above that, the doctor should know.
MISSINGNESS_WARN_PERCENT: Final = 5.0

# Fewer rows than the small-cell threshold: every count would be suppressed.
MIN_ROW_COUNT: Final = MIN_CELL_SIZE

# Upload notice length: a short list; the full result is in the report's checks.
SUMMARY_MAX_LINES: Final = 8

Severity: TypeAlias = Literal["error", "warning"]
SEVERITY_ID: Final[Severity] = "error"
SEVERITY_CODEBOOK: Final[Severity] = "warning"
SEVERITY_RANGE: Final[Severity] = "warning"
SEVERITY_MISSING: Final[Severity] = "warning"
SEVERITY_ROW_COUNT: Final[Severity] = "warning"


@dataclass(frozen=True)
class PlausibilityRange:
    """Bounds outside which a value is almost certainly an entry error, not a rare patient.

    These are not reference ranges. A column matches by its normalized short name.
    """

    variable: str
    aliases: frozenset[str]
    minimum: float
    maximum: float
    unit: str
    source: str


PLAUSIBILITY_RANGES: Final[tuple[PlausibilityRange, ...]] = (
    PlausibilityRange(
        variable="age",
        aliases=frozenset({"age", "ageyears", "ageyrs", "ageatbaseline", "ageatenrollment", "ageatenrolment"}),
        minimum=0,
        maximum=122,
        unit="years",
        # Longest documented human lifespan, 122 years: Robine JM, Allard M. The oldest human.
        # Science 1998;279:1831.
        source="Robine & Allard, Science 1998;279:1831 (max documented lifespan 122 y)",
    ),
    PlausibilityRange(
        variable="bmi",
        aliases=frozenset({"bmi", "bodymassindex"}),
        # Survivors below BMI 12 are documented (Collins S, Nat Med 1995;1:810-814), so the lower
        # bound sits below that. The upper bound is above recorded extremes; owner to confirm.
        minimum=10,
        maximum=200,
        unit="kg/m2",
        source="Lower: Collins, Nat Med 1995;1:810 (survival below BMI 12). Upper: unverified, owner to confirm",
    ),
)

# Lipids and other labs are unit-dependent (mg/dL vs mmol/L differ ~39x for cholesterol), so they
# get no range unless a unit is known. Not configured yet: deferred until units are captured.
