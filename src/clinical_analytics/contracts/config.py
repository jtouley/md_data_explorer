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
# Every contract for a dataset version. Not bumped yet when a patch is accepted later: the
# re-contract trigger is deferred (docs/architecture/AS_BUILT_ARCHITECTURE.md, known limits).
INITIAL_CONTRACT_VERSION: Final = "1.0.0"
UNIFIED_OBJECT_NAME: Final = "unified"
SYNTHETIC_TAG: Final = "synthetic"

# Counts 1..threshold-1 are shown as "<threshold" everywhere in the app (config_loader default 11).
MIN_CELL_SIZE: Final = SMALL_CELL_THRESHOLD

# Missingness above 5% is flagged. Jakobsen et al., BMC Med Res Methodol 2017;17:162
# (doi:10.1186/s12874-017-0442-1): complete case analysis "may be used as the primary analysis if
# the proportions of missing data are below approximately 5%"; above that, the doctor should know.
MISSINGNESS_WARN_PERCENT: Final = 5.0

# A count hidden so that a small count cannot be worked out from the visible ones (secondary
# suppression), shown instead of a number. Small counts themselves show as "<11".
COUNT_HIDDEN: Final = "hidden"

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

# Without an identifier column the ID checks still appear in the report, as not run.
ID_NOT_NULL_RULE_ID: Final = "id_not_null"
ID_UNIQUE_RULE_ID: Final = "id_no_duplicates"
ID_RULE_DISPLAY: Final = "Identifier"
NO_ID_COLUMN_REASON: Final = "no ID column identified"

# ODCS allows minimum/maximum only on numeric logical types; for a plausibility column loaded as
# text the bounds go in this custom property so the range check is reported as not run.
PLAUSIBILITY_RANGE_PROPERTY: Final = "plausibilityRange"


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
        # Lower: survivors below BMI 12 are documented (Collins S, Nat Med 1995;1:810-814), so the
        # bound sits below that. Upper: adult EHR BMI outside 15-90 kg/m2 is treated as implausible
        # (Lin PD et al., JAMIA Open 2022;5(4):ooac089, PMC9629892). A warning, never a rejection.
        minimum=10,
        maximum=90,
        unit="kg/m2",
        source="Lower: Collins, Nat Med 1995;1:810 (survival below BMI 12). "
        "Upper: Lin et al., JAMIA Open 2022;5(4):ooac089 (adult EHR BMI 15-90)",
    ),
)

# Lipids and other labs are unit-dependent (mg/dL vs mmol/L differ ~39x for cholesterol), so they
# get no range unless a unit is known. Not configured yet: deferred until units are captured.
