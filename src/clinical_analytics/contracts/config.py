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

# Markers a public source writes in place of a number it withholds (ADR014 §10). CDC WONDER:
# "Suppressed" (1-9 deaths, and rates built on them), "12.3 (Unreliable)" (a rate on fewer than 20
# deaths), "Not Applicable" (no rate). Staging moves them out of the numbers into flag columns.
# A cell the source withheld, matched on the lowercased, trimmed cell: "Suppressed", "Data
# suppressed", "*", "**", "(S)", "S", "(D)", "D", "<10", "< 5", "<=10", "≤10", "1-9", "1–10".
# Only in a column whose other cells are numbers; such a cell is never read as a number.
# The words alone, for any upload (a patient file's "<5" is a censored lab value, not suppression).
SOURCE_SUPPRESSED_WORDS: Final = r"^(\*?\(?suppressed\)?\*?|data suppressed|\(s\))$"
# The full set, for uploads declared public estimates (ADR014 §7):
SOURCE_SUPPRESSED_PATTERN: Final = (
    r"^(\*?\(?suppressed\)?\*?|data suppressed|\*{1,3}|\(?[sd]\)?|(<|<=|≤)\s*\d{1,2}|[01]\s*[-–]\s*(9|10))$"
)
SOURCE_NOT_APPLICABLE_TOKENS: Final[tuple[str, ...]] = ("not applicable", "not available")
SOURCE_UNRELIABLE_PATTERN: Final = r"(?i)\s*\(?\s*unreliable\s*\)?\s*$"
SUPPRESSED_FLAG_SUFFIX: Final = " suppressed"
UNRELIABLE_FLAG_SUFFIX: Final = " unreliable"

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

# A column with any empty value is a warning at upload, never "passed": an incomplete column
# reported as passed hides that answers on it leave rows out. (Answers still note missingness
# above MISSINGNESS_WARN_PERCENT for the columns a question uses: core.answer_checks.)
MISSINGNESS_ALLOWED_PERCENT: Final = 0.0

# A value and its interval bounds (core.identifier_rules.find_intervals): every row where all
# three are present must have lower <= value <= upper. A warning: the upload is kept.
SEVERITY_INTERVAL: Final[Severity] = "warning"
INTERVAL_RULE_SUFFIX: Final = "_interval"

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
# The one form value that declares it (POST /datasets/upload, field population_estimates). Not a
# boolean: "true", "on" or "1" from a checkbox default or a script flag is rejected, not read as yes.
POPULATION_ESTIMATES_FORM_VALUE: Final = "public_population_estimates_no_patient_rows"
# Upload saved, mark not: the dataset exists under its name, so "upload again" alone would be refused.
POPULATION_ESTIMATES_NOT_SAVED: Final = (
    "The dataset was saved, but its public-population-estimates mark wasn't, so it is answered under the "
    "patient rules. Delete it, then upload it again with the box ticked."
)
POPULATION_ESTIMATES_NOT_DETECTED: Final = (
    "You marked this file public population estimates, but its rows don't look like estimates for groups "
    "(a column comes close to one value per row, like a record ID, or nothing marks a rate, interval or total). "
    "The patient rules still apply to it."
)

# Bound columns paired with a value column by name: <value><low suffix> and <value><high suffix>
# (normalised: lowercase, runs of other characters as "_"). WHO TB: e_inc_100k_lo / _hi.
ESTIMATE_BOUND_SUFFIXES: Final[tuple[tuple[str, str], ...]] = (
    ("_lo", "_hi"),
    ("_lower", "_upper"),
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
    r"(^|_)(rate|rates|ratio|sir|pct|percent|percentage|prevalence|proportion|coverage|mx|"
    r"per_?\d+k?|\d+k)($|_)"
)
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
# Age levels that summarise every age by standardisation (normalised: lowercase, "_" for others).
# Never interchangeable with a crude "All ages" value.
ESTIMATE_STANDARDIZED_LEVELS: Final = frozenset(
    {"age_standardized", "age_standardised", "age_adjusted", "age_standardized_rate", "age_adjusted_rate"}
)
# Metric-like dimensions whose levels say what the value is (GBD metric_name: Number, Rate, Percent;
# CDC DataValueType: Number, Crude Prevalence). A count level there keeps the small-cell rule.
ESTIMATE_METRIC_DIMENSION: Final = re.compile(
    r"(^|_)(metric|unit|units|datavaluetype|data_value_type|metric_name)($|_)"
)
ESTIMATE_COUNT_LEVELS: Final = frozenset({"number", "count", "counts", "cases", "number_of_cases", "deaths"})
# A metric level that positively says its value is a rate or a share ("Rate", "Percent", "Crude
# Prevalence", "Age-adjusted Rate", "Per 100,000"). Any other level keeps the small-cell rule.
ESTIMATE_RATE_LEVEL: Final = re.compile(
    r"(^|_)(rate|rates|percent|percentage|pct|prevalence|proportion|ratio|coverage|per)($|_)"
)
ESTIMATE_SUBAREA_NESTING_UNKNOWN: Final = (
    "{source} has counties of several states in {column}, and the state of some counties is not known "
    "(no county FIPS code or 'County, ST' name). A ranking of one state's counties could leave some out "
    "or mix in another state's, so it is not answered. Add a state column or county FIPS codes."
)
ESTIMATE_SUBAREA_NAME: Final = re.compile(r"\b(county|parish|borough|census area|municipality)\s*$", re.IGNORECASE)
# ---- Patient-level files can never pass as public estimates (structural guard, fail closed).
# Name tokens of a column that identifies a person or an encounter (any one is enough).
ESTIMATE_PERSON_TOKENS: Final = frozenset(
    {
        "mrn", "patient", "pt", "subject", "participant", "pid", "person", "member", "mbr", "enc",
        "encounter", "client", "ssn", "hadm", "beneficiary", "enrollee", "respondent", "guarantor",
    }
)  # fmt: skip
ESTIMATE_PERSON_NAMES: Final = frozenset({"dob", "dateofbirth", "birthdate", "id", "uid", "uuid"})
# A label or attribute column with at least this share of values reading as people's names is a
# names column (ADR014 §7); one odd title-case label below it is not.
ESTIMATE_NAME_SHARE_LIMIT: Final = 0.3
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
ESTIMATE_SOURCE_SUPPRESSED_NOTE: Final = (
    'Cells the source suppressed (marked "Suppressed") stay hidden; they are never read as zero or filled in.'
)
ESTIMATE_UNRELIABLE_NOTE: Final = "The source marks the value for {groups} as unreliable (based on few events)."
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


# ---------------------------------------------------------------------------- joining public tables
# ADR014 §10 (core/estimate_join.py): questions across several declared estimate tables, joined on
# state FIPS x year x sex (x age). Wording is plain on purpose: the reader is a clinician.

# Question words expanded before matching published level names (lowercase; the expansion is added,
# the word itself kept). ID abbreviations a fellow types for GBD cause names.
ESTIMATE_QUESTION_SYNONYMS: Final[dict[str, str]] = {
    "hcv": "hepatitis c",
    "hbv": "hepatitis b",
    "hiv": "hiv/aids",
    "lri": "lower respiratory infections",
    "pneumonia": "lower respiratory infections",
    "mortality": "deaths",
    "died": "deaths",
    "new cases": "incidence",
    "legionnaires": "legionella",
    "c diff": "clostridioides difficile",
    "tb": "tuberculosis",
}
ESTIMATE_JOIN_PRECISION_WORDS: Final = (
    r"\b(how (precise|sure|certain)|precision|too uncertain|widest (uncertainty|interval)|width of the interval|"
    r"interval width|most uncertain|least certain)\b"
)
ESTIMATE_JOIN_DISAGREE_WORDS: Final = (
    r"\b(disagree|disagrees|disagreement|agree with|out of line|outside (the|its|their) interval|"
    r"falls? (inside|outside|within)|lies? (inside|outside|within)|"
    r"model(led)? (vs\.?|versus) (surveillance|reported)|(surveillance|reported) (vs\.?|versus) model(led)?|"
    r"differ most|gap between)\b"
)
ESTIMATE_JOIN_RANK_WORDS: Final = (
    r"\b(rank|ranks|ranking|ranked|where does|highest|lowest|top \d+|bottom \d+|which states?)\b"
)
ESTIMATE_JOIN_CHANGE_WORDS: Final = (
    r"\b(change|changed|changes|trend|trends|over time|rise|rising|rose|fall|fell|since)\b"
)
ESTIMATE_JOIN_TREND_WORDS: Final = (
    r"\b(trend|trends|over time|each year|by year|every year|year by year|yearly|annual)\b"
)
ESTIMATE_JOIN_ALL_STATES_WORDS: Final = (
    r"\b(all states|every state|each state|by state|all 50 states|other states|states)\b"
)
# Canonical sex -> question pattern ("women" names Female; "men" never matches inside "women").
ESTIMATE_JOIN_SEX_WORDS: Final[dict[str, str]] = {
    "Female": r"\b(women|woman|females?|girls?)\b",
    "Male": r"\b(men|man|males?|boys?)\b",
    "Both": r"\b(both sexes|all sexes|men and women combined|overall)\b",
}
# Published sex levels -> canonical (lowercase keys).
ESTIMATE_JOIN_SEX_LEVELS: Final[dict[str, str]] = {
    "female": "Female",
    "females": "Female",
    "f": "Female",
    "women": "Female",
    "male": "Male",
    "males": "Male",
    "m": "Male",
    "men": "Male",
    "both": "Both",
    "both sexes": "Both",
    "all sexes": "Both",
    "total": "Both",
    "overall": "Both",
    "a": "Both",
}
# Metric levels (normalised) -> the question words that name them.
ESTIMATE_JOIN_METRIC_WORDS: Final[dict[str, str]] = {
    "rate": r"\brates?\b",
    "number": r"\b(how many|number of|counts?)\b",
    "percent": r"\b(percent|percentage|share|proportion|attributable)\b|%",
}
# Non-role dimension stem -> other words a question uses for it ("by disease" groups cause_name).
ESTIMATE_JOIN_DIMENSION_WORDS: Final[dict[str, tuple[str, ...]]] = {
    "cause": ("disease", "infection", "cause"),
    "rei": ("risk", "risk factor"),
    "location": ("state", "place"),
}
# Rate columns whose source publishes them per 100,000 (CDC WONDER's names; lowercase).
ESTIMATE_JOIN_PER_100K_COLUMNS: Final = frozenset({"crude rate", "age adjusted rate", "age-adjusted rate"})

ESTIMATE_JOIN_NO_SOURCES: Final = "Choose at least one uploaded table to ask about."
ESTIMATE_JOIN_UNDECLARED: Final = (
    "{source} wasn't marked '" + POPULATION_ESTIMATES_LABEL + "' for its current version, so it can't be "
    "joined with other tables. If it holds only published estimates, upload it again with that box ticked."
)
ESTIMATE_JOIN_NOT_ESTIMATES: Final = (
    "{source} doesn't look like a table of estimates: no set of its columns identifies each row as a place, "
    "period or group (one column comes close to one value per row, like a record ID). It can't be joined here."
)
ESTIMATE_JOIN_OUT_WAYNE: Final = (
    "{source}'s Wayne County rows are out-Wayne: Wayne County excluding Detroit, which has its own health "
    "department. They are not Wayne County. {detroit}. A whole-county count is Detroit plus out-Wayne, and only "
    "for counts (rates can't be added). Ask about out-Wayne or Detroit by those names."
)
ESTIMATE_JOIN_COUNTY_GRAIN: Final = (
    "Tables are joined by state. Wayne County (and Detroit, which reports separately from out-Wayne) is part of "
    "a state, so it isn't joined here. Ask about Michigan, or ask the county table on its own."
)
ESTIMATE_JOIN_DISAGREE_TWO: Final = "To compare a model with surveillance, ask about exactly two tables."
ESTIMATE_JOIN_PRECISION_ONE: Final = "Ask about the precision of one table at a time."
ESTIMATE_JOIN_SEVERAL_LEVELS: Final = (
    "The question names several {column} values in {source} ({levels}). Name one, so values from different "
    "rows aren't put side by side as one."
)
ESTIMATE_JOIN_OPEN_HOLD: Final = "{source}: the question doesn't fix {column} ({levels}); {top} is the total."
ESTIMATE_JOIN_OPEN_REFUSAL: Final = (
    "{source}: the question doesn't fix {column} ({levels}), and there is no total among them. Name one value."
)
ESTIMATE_JOIN_AGE_OPEN: Final = (
    "{source} has both age-standardized and all-ages (crude) values, and they can't be swapped: age-standardized "
    "rates compare places with different age mixes; crude rates describe the actual population. Say "
    "'age-standardized' or 'crude'."
)
ESTIMATE_JOIN_CONFIRM: Final = "Confirm to use these choices (they are saved for each dataset), or name others."
ESTIMATE_JOIN_WHICH_VALUE: Final = "{source} has several value columns ({values}). Which one?"
ESTIMATE_JOIN_STANDARDIZATION_UNKNOWN: Final = (
    "{source}'s {value} doesn't say whether it is age-standardized or crude, and joined rates must match on that. "
    "Confirm which it is (most CDC surveillance rates, such as AtlasPlus, are crude)."
)
ESTIMATE_JOIN_SEASON_HOLD: Final = (
    "{source} reports by flu season ({column}), which spans two calendar years. Confirm season {default} for {year}, "
    "or choose another season."
)
ESTIMATE_JOIN_MONTH_HOLD: Final = (
    "{source} is a monthly table ({column}). Confirm 'all months' to see each month of the year asked, "
    "or choose one month."
)
ESTIMATE_JOIN_MONTH_BESIDE_ANNUAL: Final = (
    "{source} is a monthly table ({column}) and another table has one value per year. Choose the month to "
    "line up with the year; an annual total or average of the months isn't published, so none is computed."
)
ESTIMATE_JOIN_MONTH_VS_YEAR: Final = (
    "{source}'s value is for one month ({month}); the other table's is for the whole year {year}. They are "
    "shown side by side as different periods, not as the same estimate."
)
ESTIMATE_JOIN_MONTH_VS_YEAR_REFUSED: Final = (
    "One table's value is for a single month ({month}) and the other's is for a whole year, so they can't be "
    "compared as the same estimate. Ask for them side by side, or use tables with the same period."
)
ESTIMATE_JOIN_MONTHS_DIFFER: Final = (
    "The monthly tables are set to different months ({months}), so their values are not the same period and "
    "aren't joined. Choose the same month for each."
)
ESTIMATE_JOIN_SAME_MONTH_EACH_YEAR: Final = (
    "{source} is monthly: the change compares the same month (month {month}) of each year asked."
)
ESTIMATE_JOIN_NAME_MONTH_YEAR: Final = "{source} is a monthly table. Name a year (or a month, such as 2023-03)."
ESTIMATE_JOIN_SEASON_TREND: Final = (
    "{source} reports by flu season ({column}), which spans two calendar years, so it can't be lined up with "
    "calendar years over a range. Ask about one year, and confirm which season goes with it."
)
ESTIMATE_JOIN_YEAR_MISSING: Final = (
    "{source} has no {year} values (it has {first}–{last}). Confirm to show its {default} values beside {year}, or "
    "name another year."
)
ESTIMATE_JOIN_YEAR_ALIGNED: Final = " Its {year} row uses the year confirmed for it."
ESTIMATE_JOIN_NAME_YEAR: Final = "{source} has several years ({first}–{last}). Name a year."
ESTIMATE_JOIN_NAME_STATE: Final = (
    "{source} has many states. Name a state (or a group, such as the Great Lakes), or say 'all states'."
)
ESTIMATE_JOIN_NO_SEX: Final = (
    "{source} has no breakdown by sex, so it can't be split into women and men or put beside one sex's values. "
    "Ask without the split, or for both sexes together."
)
ESTIMATE_JOIN_COUNTY_TABLE: Final = (
    "{source} is by county or city ({column}), not by state. Tables are joined by state, and county values "
    "can't be turned into a state value here: rates and percentages can't be added up, and a state needs every "
    "county with its population. Use the source's state-level table instead."
)
ESTIMATE_JOIN_NO_STATE: Final = "{source} has no state column, so it can't be joined or ranked by state."
ESTIMATE_JOIN_NOT_STATES: Final = (
    "{source}: {count} location values aren't states and were left out ({levels}). Counties, cities and local "
    "health jurisdictions (Detroit, out-Wayne) are parts of a state, and territories, regions and countries "
    "aren't states, so none is joined as one."
)
ESTIMATE_JOIN_AGE_SUMMARIES_DROPPED: Final = (
    "{source}: summary ages (all ages, age-standardized, bands that contain other bands) are left out of a "
    "breakdown by age."
)
ESTIMATE_JOIN_AMBIGUOUS_PLACE: Final = (
    "{source}: {level} was left out. This table also lists countries, so it can't say whether {level} is the US "
    "state or the country."
)
ESTIMATE_JOIN_NATION_LEFT_OUT: Final = (
    "{source}: the national row is a total of the states, so it is left out of the comparison."
)
ESTIMATE_JOIN_EMPTY_KEYS: Final = "{source}: {count} rows with an empty state, year or sex (total rows) were left out."
ESTIMATE_JOIN_GRAIN: Final = (
    "{source} has more than one row for a {keys} combination after the question's choices, so its values can't "
    "be lined up one to one. Name a value for every column that identifies its rows."
)
ESTIMATE_JOIN_NO_SHARED_KEY: Final = "{a} and {b} share no state, year or sex column, so they can't be joined."
ESTIMATE_JOIN_FAN_OUT: Final = (
    "{source} has no {keys}, so its one value would be repeated beside each {keys}. Ask about one {keys}, or "
    "leave {source} out."
)
ESTIMATE_JOIN_UNMATCHED: Final = "Joined rows: {a_only} only in {a} and {b_only} only in {b} were left out."
ESTIMATE_JOIN_NOTHING_MATCHED: Final = (
    "No rows matched across the tables for the states, years and groups asked. Check that the tables cover the "
    "same years and states."
)
ESTIMATE_JOIN_METHOD_SOURCE: Final = (
    "{source}: {value}{bounds}{pinned} ({measure}); {rows} rows after the question's choices.{aligned}"
)
ESTIMATE_JOIN_METHOD_JOIN: Final = (
    "Joined on {keys} (states matched by name, postal code or FIPS to the two-digit FIPS code): {rows} rows."
)
ESTIMATE_JOIN_POOL_WORDS: Final = r"\b(average|averaged|mean|pooled|combined rate|overall rate|sum of|total across)\b"
ESTIMATE_JOIN_NO_POOL: Final = (
    "No average or total across rows is computed: each row is its own estimate for its own population, and "
    "intervals from different years or places can't be merged into one. The rows are shown instead."
)
ESTIMATE_JOIN_TABLE_CLAIM: Final = (
    "{rows} rows from {sources}, side by side. Nothing is computed across the tables; each value keeps its own "
    "interval."
)
ESTIMATE_JOIN_KIND_MISMATCH: Final = (
    "{a}'s value is a {ak} and {b}'s is a {bk}, so they can't be compared directly. Ask for the same kind of "
    "number from both (for example a rate per 100,000)."
)
ESTIMATE_JOIN_SCALE_MISMATCH: Final = "{a}'s rate is per {ap:,} and {b}'s per {bp:,}; ask for rates on the same scale."
ESTIMATE_JOIN_STANDARDIZATION_MISMATCH: Final = (
    "{a}'s rate is {ast} and {b}'s is {bst}. An age-standardized rate and a crude rate measure different things "
    "and are never compared. Ask for the same kind from both (for example GBD 'All ages' beside a crude rate)."
)
ESTIMATE_JOIN_STANDARD_MISMATCH: Final = (
    "Both rates are age-standardized, but {a} and {b} may use different standard populations (GBD uses its own "
    "world population; CDC tables mostly the 2000 US standard), so their gap could be the standard, not the "
    "disease. Compare crude rates instead."
)
ESTIMATE_JOIN_CHANGE_NO_YEAR: Final = "The tables have no year column, so a change over time can't be shown."
ESTIMATE_JOIN_CHANGE_ONE_YEAR: Final = (
    "A change needs two years with values; the rows asked for have {years}. Name a first and a last year."
)
ESTIMATE_JOIN_CHANGE_HIDDEN: Final = (
    "{group} ({source}): no change is computed, because the source suppressed a value it would be built on."
)
ESTIMATE_JOIN_CHANGE_OVERLAP: Final = (
    "The two years' intervals overlap, so they don't show a change on their own (a published percent change "
    "with its own interval is the test)."
)
ESTIMATE_JOIN_CHANGE_APART: Final = "The two years' intervals don't overlap."
ESTIMATE_JOIN_CHANGE_BAD_INTERVAL: Final = (
    "The file's interval for one of the years doesn't contain its estimate (the bounds look wrong), so no "
    "interval is used."
)
ESTIMATE_JOIN_PRECISION_BAD: Final = "{count} rows whose interval doesn't contain the estimate are not ranked."
ESTIMATE_JOIN_CHANGE_NO_BOUNDS: Final = (
    "The file gives no intervals, so whether this is more than noise can't be judged."
)
ESTIMATE_JOIN_CHANGE_METHOD: Final = (
    "Change = last year's value minus the first year's; percent change divides that by the first year's value. "
    "Each year keeps its own interval. No interval is computed for the change: two years' intervals can't be "
    "combined into one (the years' estimates are not independent)."
)
ESTIMATE_JOIN_RANK_WHICH: Final = "Rank by which table? Name one of {sources} in the question."
ESTIMATE_JOIN_RANK_OPEN: Final = (
    "The rows asked for vary in {columns}; a ranking needs one value each. Name one {columns}."
)
ESTIMATE_JOIN_RANK_FOCUS: Final = (
    "{state} ranks {rank} of {total} states ({order}) on {source} {value}{scope}: {shown}."
)
ESTIMATE_JOIN_RANK_RANGE: Final = (
    "Its interval overlaps other states', so its rank could be anywhere from {best} to {worst}."
)
ESTIMATE_JOIN_RANK_CERTAIN: Final = "Its interval overlaps no other state's, so this rank is firm."
ESTIMATE_JOIN_RANK_NO_BOUNDS: Final = "The file gives no intervals, so how firm the rank is can't be judged."
ESTIMATE_JOIN_RANK_TOP: Final = "{total} states ranked ({order}) on {source} {value}{scope}: {listed}."
ESTIMATE_JOIN_RANK_METHOD: Final = (
    "Rank range: states whose interval lies entirely beyond this state's are certainly ahead of it; states whose "
    "interval overlaps it may be. The national row is not a state and is not ranked."
)
ESTIMATE_JOIN_RANK_HIDDEN: Final = "{count} states whose value the source suppressed are not ranked."
ESTIMATE_JOIN_DISAGREE_NO_BOUNDS: Final = (
    "Neither {a} nor {b} gives an interval, so whether they disagree can't be judged."
)
ESTIMATE_JOIN_DISAGREE_CLAIM: Final = (
    "Of {total} rows, {observed}'s value is above {model}'s interval in {above} and below it in {below}; it "
    "{test} the interval in {inside}. Largest gaps ({observed} ÷ {model}): {listed}."
)
ESTIMATE_JOIN_DISAGREE_METHOD: Final = (
    "A row disagrees when {observed}'s value lies outside {model}'s uncertainty interval (or, when both have "
    "intervals, when the intervals don't overlap). This is a screen for where to look, not a significance test."
)
ESTIMATE_JOIN_PRECISION_NO_BOUNDS: Final = "{source} gives no intervals, so its precision can't be judged."
ESTIMATE_JOIN_PRECISION_CLAIM: Final = (
    "{total} estimates from {source}, widest interval first (relative width): {listed}."
)
ESTIMATE_JOIN_PRECISION_METHOD: Final = (
    "Relative width = (upper − lower) ÷ estimate. A width above 1 means the interval is wider than the estimate itself."
)
ESTIMATE_JOIN_PRECISION_SKIPPED: Final = (
    "{count} rows without an interval or with an estimate of 0 or below are not ranked."
)
