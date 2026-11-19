# SPDX-License-Identifier: MIT
"""Constants for data contracts and data-quality checks (one place, each with its source).

Thresholds here are reporting thresholds: a check above one is shown to the user as a warning.
They never block an upload.
"""

from __future__ import annotations

import re
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


def is_tiny_table(rows: int) -> bool:
    """A table of 1..MAX_ROWS_NO_COUNTS rows: no column count is reported, zeros included, and
    the contract states no grain. The one rule the builder and the report both use."""
    return 0 < rows <= MAX_ROWS_NO_COUNTS


OVERLAPPING_CHECK_REASON: Final = "another check already counts problem values in this column"
TOO_FEW_ROWS_NOTE: Final = "too few rows to report counts"

# Fewer rows than the small-cell threshold: every count would be suppressed.
MIN_ROW_COUNT: Final = MIN_CELL_SIZE

# Cell values that mean "missing" in clinical exports (R NA, SAS/Stata ".", Excel #N/A, ...),
# matched case-insensitively after strip. Staging nulls them only in a column that otherwise
# types as numbers or dates; a text column keeps every value ("None" can be a category level).
# Same constant as PR #96 (answer side) plus "#n/a".
MISSING_VALUE_TOKENS: Final[tuple[str, ...]] = (
    "",
    "na",
    "n/a",
    "nan",
    "null",
    "none",
    "missing",
    ".",
    "-",
    "?",
    "inf",
    "+inf",
    "-inf",
    "infinity",
    "+infinity",
    "-infinity",
    "#n/a",
)

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


# A value recorded as a limit ("<5", "> 100", "<=0.1", "< 0.5 mg/L"): censored, never the limit
# itself. An optional unit may follow the number.
CENSORED_TEXT_PATTERN: Final = (
    r"(?i)^\s*(<=|>=|<|>|≤|≥)\s*[+-]?(\d[\d,]*(\.\d*)?|\.\d+)([eE][+-]?\d+)?\s*([a-zµμ%][a-zµμ%/0-9.]*)?\s*$"
)
CENSORED_RULE_SUFFIX: Final = "_censored"
# Lab words for "below the limit of quantification / not detected": censored like "<5".
# Compared after strip() and lower(). Configurable: add a site's own words here.
CENSORED_TEXT_WORDS: Final[tuple[str, ...]] = ("blq", "bql", "nd", "<lod", "<loq", "not detected")
# A text column whose present values are at least this share plain numbers (the rest neither a
# limit nor a missing token, e.g. "pending") is numbers with unreadable entries: answers refuse
# and name the entries, never read the column as categories.
NEAR_NUMERIC_SHARE: Final = 0.95
NAMED_UNREADABLE_MAX: Final = 3

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
    # Aliases too short to trust alone ("na", "k", "tc": also "not applicable", "treatment code"):
    # they match only with a unit in the header ("Na (mmol/L)" -> "nammoll").
    hinted_aliases: frozenset[str] = frozenset()
    # Per-unit bounds chosen from the column's median (temperature: °C below 50, else °F). When
    # the median is unknown the wide bound applies and its text says the unit is unknown.
    by_median: tuple[UnitBounds, ...] = ()


@dataclass(frozen=True)
class UnitBounds:
    """Bounds in one unit, used when the column's median is below ``median_below`` (None: any)."""

    unit: str
    minimum: float
    maximum: float
    median_below: float | None


# Header suffixes that name a lab unit (normalised: lower case, letters and digits only).
UNIT_HINTS: Final[tuple[str, ...]] = ("mmoll", "mmol", "mgdl", "mgl", "meql", "meq", "gdl", "gl", "umoll", "µmoll")
WIDE_BOUND_NOTE: Final = "unit unknown, wide bound"


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
        aliases=frozenset({"glucose", "bloodglucose", "plasmaglucose", "fastingglucose", "fbg", "fpg"}),
        hinted_aliases=frozenset({"glu"}),
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
        # recorded survived body temperature, 46.5 C (Guinness World Records, 1980). The unit
        # is read from the median (a body temperature in °C is below 50, in °F above): 98.6 in a
        # °C column is flagged. Median unknown: the wide bound, from the °C lower to the °F upper.
        minimum=10,
        maximum=117,
        unit="°C or °F",
        source="Lower: Gilbert et al., Lancet 2000;355:375 (survival at 13.7 °C). "
        "Upper: Guinness World Records 1980 (46.5 °C survived), as °F",
        by_median=(
            UnitBounds(unit="°C", minimum=10, maximum=47, median_below=50),
            UnitBounds(unit="°F", minimum=50, maximum=117, median_below=None),
        ),
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
            hinted_aliases=frozenset(hinted),
            minimum=0,
            maximum=None,
            unit="concentration (unit unknown)",
            source="Physical: a concentration cannot be negative. No upper bound is configured: "
            "the unit (mg/dL, mmol/L, µmol/L) changes it by up to ~90x",
        )
        for name, aliases, hinted in (
            ("ldl", {"ldl", "ldlc", "ldlcholesterol"}, set()),
            ("hdl", {"hdl", "hdlc", "hdlcholesterol"}, set()),
            ("cholesterol", {"cholesterol", "totalcholesterol", "chol"}, {"tc"}),
            ("triglycerides", {"triglycerides", "triglyceride", "trig"}, {"tg"}),
            ("creatinine", {"creatinine", "creat", "scr", "serumcreatinine"}, set()),
            ("crp", {"crp", "creactiveprotein", "hscrp"}, set()),
            ("sodium", {"sodium"}, {"na"}),
            ("potassium", {"potassium"}, {"k"}),
            ("haemoglobin", {"haemoglobin", "hemoglobin", "hgb"}, {"hb"}),
            ("albumin", {"albumin"}, {"alb"}),
            ("bilirubin", {"bilirubin", "bili", "tbili"}, set()),
            ("urea", {"urea", "bun"}, set()),
        )
    ),
)

# Numeric columns with no configured range are marked so in the contract, never silently passed.
NO_RANGE_KNOWN: Final = "no range known"


# ---------------------------------------------------------------------------- aggregate-data mode
# ADR014: files whose rows are estimates (location x year x sex x age x cause x measure), not
# patients. Detection (core/estimates.py) only adds checks; the relaxations below need the upload
# to carry POPULATION_ESTIMATES_STATEMENT, which the person ticks at upload.

# What the person declares at upload; stored per dataset (declarations.json) and in the contract.
POPULATION_ESTIMATES_LABEL: Final = "Public population estimates – no patient-level data"
POPULATION_ESTIMATES_STATEMENT: Final = (
    "Every row is a published estimate for a place, period or group (for example a rate per 100,000 "
    "with its interval), and no row describes a single patient."
)
POPULATION_ESTIMATES_PRIVACY_CLASS: Final = "public_aggregate"

# Bound columns paired with a value column by name: <value><low suffix> and <value><high suffix>
# (normalised: lowercase, runs of other characters as "_"). WHO TB: e_inc_100k_lo / _hi.
ESTIMATE_BOUND_SUFFIXES: Final[tuple[tuple[str, str], ...]] = (
    ("_lo", "_hi"),
    ("_lower", "_upper"),
    ("_low", "_high"),
    ("_lci", "_uci"),
    ("_lcl", "_ucl"),
    ("_cilower", "_ciupper"),
    ("_lower_ci", "_upper_ci"),
    ("_lb", "_ub"),
)
# Standalone bound pairs beside one value column named in ESTIMATE_VALUE_NAMES (GBD: val, lower,
# upper; IHME county files: mx, lower, upper; CDC CDI: datavalue, low/highconfidencelimit).
ESTIMATE_STANDALONE_BOUNDS: Final[tuple[tuple[str, str], ...]] = (
    ("lower", "upper"),
    ("lo", "hi"),
    ("lowconfidencelimit", "highconfidencelimit"),
    ("low_confidence_limit", "high_confidence_limit"),
    ("lower_ci", "upper_ci"),
    ("low_ci", "high_ci"),
    ("lci", "uci"),
    ("lower_bound", "upper_bound"),
)
ESTIMATE_VALUE_NAMES: Final[tuple[str, ...]] = (
    "val",
    "best",
    "mx",
    "value",
    "mean",
    "estimate",
    "point_estimate",
    "datavalue",
    "data_value",
    "rate",
)
# A numeric column named for a rate, ratio or percent is an estimate even without bounds.
ESTIMATE_RATE_NAME: Final = re.compile(
    r"(^|_)(rate|rates|ratio|sir|pct|percent|percentage|prevalence|incidence|proportion|coverage|mx|"
    r"per_?\d+k?|\d+k)($|_)"
)
# A value column named for a count: values of 1-10 stay hidden even on a declared dataset.
ESTIMATE_COUNT_NAME: Final = re.compile(r"(^|_)(deaths?|count|counts|cases|number|num|n|notified|events)($|_)")
# Integer columns that key rows rather than measure (year, codes, FIPS); other integers are values.
ESTIMATE_KEY_NAME: Final = re.compile(
    r"year|season|(^|_)(yr|id|code|fips|iso|iso\d|numeric|week|month|level)($|_)"
    r"|(location|question|response|topic|cause|measure|metric|age|sex|type|category)id$"
)
# Level names that are the total of the other levels in their column (compared case-insensitively).
ESTIMATE_TOTAL_LEVELS: Final = frozenset(
    {
        "both",
        "both sexes",
        "all sexes",
        "all",
        "all ages",
        "all age groups",
        "all races",
        "all races/ethnicities",
        "all races and ethnicities",
        "total",
        "overall",
        "all causes",
        "entire network",
        "united states",
        "us",
        "u.s.",
        "usa",
        "national",
        "global",
        "world",
    }
)
# A level naming a county-like part of a state: the other levels in that column are its totals.
ESTIMATE_SUBAREA_NAME: Final = re.compile(r"\b(county|parish|borough|census area|municipality)\s*$", re.IGNORECASE)
ESTIMATE_MAX_LISTED_LEVELS: Final = 5000  # levels read to find totals; a key with more lists none
ESTIMATE_LEVELS_IN_HOLD: Final = 6

ESTIMATE_PATIENT_TEST_REFUSAL: Final = (
    "Each row of this file is an estimate for a group, not a patient, so patient-level tests (models, "
    "survival curves, correlations) don't apply. Ask for the estimates themselves, for example by year or "
    "by location."
)
ESTIMATE_SHARE_REFUSAL: Final = (
    "Each row of this file is an estimate, not a patient, so a share of rows is not a share of people. "
    "Ask for the value column that holds the percent or rate."
)
ESTIMATE_KEY_NOT_VALUE: Final = (
    "{column} is one of the columns that identify each estimate ({key}), not a value. Ask about a value "
    "column instead, by {column}."
)
ESTIMATE_WHICH_VALUE: Final = "Name the value column you want, for example {values}."
ESTIMATE_GROUP_NOT_KEY: Final = (
    "{column} does not identify the estimates ({key}), so a breakdown by it would mix several estimates "
    "in each group. Ask by one of those columns instead."
)
ESTIMATE_NO_ROWS: Final = "No estimate matches the question's limits. Check the place, year and group you named."
ESTIMATE_OPEN_REFUSAL: Final = (
    "This file holds estimates, {key}. The question doesn't fix {open}. Averaging across those rows would "
    "mix overlapping groups and unweighted rates. Name one value for each of {name}, or ask for a "
    "breakdown by one of them."
)
ESTIMATE_OPEN_HOLD: Final = (
    "This file holds estimates, {key}. The question doesn't fix {open}. Confirm to use {totals} and save "
    "that choice for this dataset, or name another value."
)
ESTIMATE_UNDECLARED_REFUSAL: Final = (
    "This file looks like population estimates: each row is a value for a group, not a patient. It "
    "wasn't marked '" + POPULATION_ESTIMATES_LABEL + "' at upload, so each row is treated as one person "
    "and the small-cell rule hides it. If the file holds only published estimates, upload it again with "
    "that box ticked."
)
ESTIMATE_TOTAL_LEFT_OUT: Final = "{levels} is the total of the other {column} values, so it is left out."
ESTIMATE_SAVED_CHOICE_NOTE: Final = "Uses {column} = {level}, the choice saved for this dataset."
ESTIMATE_SMALL_COUNT_NOTE: Final = "Counts from 1 to 10 are hidden."
ESTIMATE_NO_BOUNDS_NOTE: Final = "This file gives no intervals for these values."
ESTIMATE_DUPLICATE_ROWS: Final = (
    "More than one row matches a group of this answer, so it isn't shown. Name a value for every column "
    "that identifies the estimates."
)
# GX equivalent: expect_column_pair_values_a_to_be_greater_than_b (or_equal=True) for value/lower and
# upper/value, checked per shown row at answer time rather than as an upload-time suite.
ESTIMATE_BAD_INTERVAL_NOTE: Final = (
    "The file's interval doesn't contain the estimate for {groups}, so no interval is shown there."
)
ESTIMATE_NO_TEST_NOTE: Final = "Estimates are compared by their intervals; no significance test is run on them."
# Words an estimate question uses for the file's own structure, not for a column or value: removed
# before word accounting on an estimate table (core/estimates.py ``grounded_words``; compared after
# lowercasing and dropping a plural "s").
ESTIMATE_QUESTION_WORDS: Final = frozenset(
    {
        "rate",
        "mortality",
        "death",
        "incidence",
        "prevalence",
        "estimate",
        "estimated",
        "value",
        "per",
        "100000",
        "100",
        "000",
        "how",
        "did",
        "change",
        "changed",
        "trend",
        "over",
        "time",
        "year",
        "yearly",
        "annual",
        "each",
        "from",
        "to",
        "between",
        "and",
        "compare",
        "compared",
        "versu",
        "vs",
        "which",
        "highest",
        "lowest",
        "rank",
        "top",
        "county",
        "countie",
        "both",
        "sexe",
        "sex",
        "for",
        "in",
        "the",
        "of",
        "what",
        "wa",
        "is",
        "had",
        "by",
        "interval",
        "overlap",
        "uncertainty",
        "bound",
        "ci",
        "confidence",
        "do",
        "their",
        "with",
        "it",
        "show",
        "give",
        "me",
        "a",
        "an",
    }
)
