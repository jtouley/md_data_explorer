# SPDX-License-Identifier: MIT
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

# Status of every check whose count is hidden, whatever the hidden value: a pass/warning would
# say which side of a threshold the hidden count is on, which with the row total can pin it.
HIDDEN_STATUS: Final = "review"

# The one label for every suppressed count in a quality report: small (1..10) and the others hidden
# so a small count cannot be worked out. It only says "at least 1" (zero is shown as 0); a second
# label for small counts would tell a reader the rest are 11 or more.
COUNT_HIDDEN: Final = "hidden"

# Tables of 1..MAX_ROWS_NO_COUNTS rows report no column counts (zeros included) and say "too few
# rows to report counts"; the row count is still shown, as it is in the overview and /preview.
# The cut-off is fixed, so applying it reveals nothing beyond the row count. From
# MIN_CELL_SIZE + 2 rows up, hiding a column's non-zero counts always leaves every small count
# able to be MIN_CELL_SIZE or more (rows - 2 >= 11: one hidden neighbour >= 1, and >= 1 distinct
# value when duplicates are counted).
MAX_ROWS_NO_COUNTS: Final = MIN_CELL_SIZE + 1

OVERLAPPING_CHECK_REASON: Final = "another check already counts problem values in this column"
TOO_FEW_ROWS_NOTE: Final = "too few rows to report counts"

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


# One list of text that means "no value", used by the quality report (nullValues on text columns)
# and by the analysis (numbers stored as text). Compared after strip() and lower(). "inf" is a
# non-finite value, missing like NaN, not a category label.
MISSING_VALUE_TOKENS: Final[tuple[str, ...]] = (
    "", "na", "n/a", "nan", "null", "none", "missing", ".", "-", "?", "inf", "+inf", "-inf", "infinity",
    "+infinity", "-infinity",
)  # fmt: skip

# A value recorded as a limit ("<5", "> 100", "<=0.1"): censored, never the limit itself.
CENSORED_TEXT_PATTERN: Final = r"^\s*(<=|>=|<|>|≤|≥)\s*-?\d[\d,]*(\.\d+)?\s*$"
CENSORED_RULE_SUFFIX: Final = "_censored"

# Missingness is flagged (above MISSINGNESS_WARN_PERCENT) only for the columns the upload names
# as analysis variables. Every other column's missingness is measured and listed, not flagged:
# an optional column (a second phone number) is not a data-quality problem (L1).
MISSINGNESS_MEASURE_ONLY_PERCENT: Final = 100.0

# Contract logicalType versus the data's type: a column checked as one type and queried as
# another is how a check passes on data the answers never read.
TYPE_RULE_SUFFIX: Final = "_type"

# Category spellings that are the same level once case, spacing and these synonyms are
# normalised (Male / male / M). Flagged at upload; a query on such a level is held until the
# doctor confirms the levels are one.
LEVEL_SYNONYMS: Final[dict[str, str]] = {
    "m": "male",
    "man": "male",
    "f": "female",
    "woman": "female",
    "y": "yes",
    "true": "yes",
    "n": "no",
    "false": "no",
}
LEVEL_VARIANT_MAX_LEVELS: Final = 50  # a column with more levels is free text, not a category
LEVEL_VARIANTS_RULE_SUFFIX: Final = "_spellings"


@dataclass(frozen=True)
class PlausibilityRange:
    """Bounds outside which a value is almost certainly an entry error, not a rare patient.

    These are not reference ranges. A column matches by its normalized short name. ``unit`` is
    the unit the bounds hold in; when a variable is stored in more than one unit and the column
    does not say which, the bounds are the widest across those units (so only values impossible
    in every unit are flagged) and ``unit`` names them all. ``maximum`` None: no upper bound is
    known, only the physical lower one.
    """

    variable: str
    aliases: frozenset[str]
    minimum: float
    maximum: float | None
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
    PlausibilityRange(
        variable="glucose",
        aliases=frozenset({"glucose", "bloodglucose", "plasmaglucose", "fastingglucose", "glu", "fbg", "fpg"}),
        # Lower: a concentration cannot be negative. Upper: 2,656 mg/dL (147.6 mmol/L), the
        # highest blood glucose on record (Guinness World Records, 2008). Unit unknown: the
        # mg/dL bound, which also admits every mmol/L value.
        minimum=0,
        maximum=2656,
        unit="mg/dL or mmol/L",
        source="Lower: physical (no negative concentration). Upper: Guinness World Records 2008, "
        "highest recorded blood glucose 2,656 mg/dL (147.6 mmol/L)",
    ),
    PlausibilityRange(
        variable="temperature",
        aliases=frozenset({"temperature", "temp", "bodytemperature", "bodytemp"}),
        # Lower: survival from accidental hypothermia at 13.7 C (Gilbert M et al., Lancet
        # 2000;355:375-376); the bound sits below it. Upper: 47 C (116.6 F), above the highest
        # recorded survived body temperature, 46.5 C (Guinness World Records, 1980). Unit
        # unknown: from the C lower bound to the F upper bound.
        minimum=10,
        maximum=117,
        unit="°C or °F",
        source="Lower: Gilbert et al., Lancet 2000;355:375 (survival at 13.7 °C). "
        "Upper: Guinness World Records 1980 (46.5 °C survived), as °F",
    ),
    PlausibilityRange(
        variable="spo2",
        aliases=frozenset({"spo2", "sao2", "oxygensaturation", "o2sat", "o2saturation"}),
        minimum=0,
        maximum=100,
        unit="%",
        source="Physical: a saturation is a percentage (0-100)",
    ),
    PlausibilityRange(
        variable="hba1c",
        aliases=frozenset({"hba1c", "a1c", "glycatedhaemoglobin", "glycatedhemoglobin"}),
        # % (NGSP) and mmol/mol (IFCC) share the header, so only the physical lower bound applies.
        minimum=0,
        maximum=None,
        unit="% or mmol/mol",
        source="Physical: a proportion cannot be negative (no upper bound: % and mmol/mol share a name)",
    ),
    *(
        PlausibilityRange(
            variable=name,
            aliases=frozenset(aliases),
            minimum=0,
            maximum=None,
            unit="concentration (unit unknown)",
            source="Physical: a concentration cannot be negative. No upper bound is configured: "
            "the unit (mg/dL, mmol/L, µmol/L) changes it by up to ~90x",
        )
        for name, aliases in (
            ("ldl", {"ldl", "ldlc", "ldlcholesterol"}),
            ("hdl", {"hdl", "hdlc", "hdlcholesterol"}),
            ("cholesterol", {"cholesterol", "totalcholesterol", "tc", "chol"}),
            ("triglycerides", {"triglycerides", "triglyceride", "tg", "trig"}),
            ("creatinine", {"creatinine", "creat", "scr", "serumcreatinine"}),
            ("crp", {"crp", "creactiveprotein", "hscrp"}),
            ("sodium", {"sodium", "na"}),
            ("potassium", {"potassium", "k"}),
            ("haemoglobin", {"haemoglobin", "hemoglobin", "hb", "hgb"}),
            ("albumin", {"albumin", "alb"}),
            ("bilirubin", {"bilirubin", "bili", "tbili"}),
            ("urea", {"urea", "bun"}),
        )
    ),
)

# Numeric columns with no configured range are marked so in the contract, never silently passed.
NO_RANGE_KNOWN: Final = "no range known"
