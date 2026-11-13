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


# Body temperature per unit (sources on the temperature range below); also the per-unit ranges
# of the °C/°F unit pair.
TEMPERATURE_C: Final = UnitBounds(unit="°C", minimum=10, maximum=47, median_below=50)
TEMPERATURE_F: Final = UnitBounds(unit="°F", minimum=50, maximum=117, median_below=None)
GLUCOSE_MAX_MG_DL: Final = 2656.0  # Guinness World Records 2008 (source on the glucose range)


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
        maximum=GLUCOSE_MAX_MG_DL,
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
        by_median=(TEMPERATURE_C, TEMPERATURE_F),
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


# --- Mixed units (DQ M3) ------------------------------------------------------------------------
# A lab recorded in two units in one column ("glucose" holding mg/dL and mmol/L values) has no
# meaningful mean. It is found as two value clusters whose ratio is the unit pair's conversion
# factor, each cluster plausible in its own unit. Only the pairs below are checked: a column
# without a configured pair is never guessed at.


@dataclass(frozen=True)
class UnitPair:
    """Two units of one variable: ``high = low * scale + offset`` (``low`` is the unit whose
    numbers are smaller: mmol/L beside mg/dL, °C beside °F).

    ``variable`` names the ``PlausibilityRange`` whose aliases find the column; ``aliases`` are
    used instead for a variable with no plausibility range (weight, height). ``low`` and
    ``high`` carry each unit's plausible bounds and the header suffixes that name it.
    """

    variable: str
    low: UnitBounds
    high: UnitBounds
    scale: float
    source: str
    offset: float = 0.0
    low_hints: tuple[str, ...] = ()
    high_hints: tuple[str, ...] = ()
    aliases: frozenset[str] = frozenset()


# Header suffixes (normalised like UNIT_HINTS) that name each unit.
_MG_DL: Final = ("mgdl",)
_MMOL_L: Final = ("mmoll", "mmol")
_UMOL_L: Final = ("umoll", "µmoll")


def _lab(
    variable: str,
    low: tuple[str, tuple[str, ...]],
    high: tuple[str, tuple[str, ...]],
    scale: float,
    high_maximum: float,
    source: str,
) -> UnitPair:
    """A concentration in two units: physical lower bound 0 in both, the upper bound given in
    ``high`` and converted for ``low`` (so both ranges describe the same patients)."""
    return UnitPair(
        variable=variable,
        low=UnitBounds(unit=low[0], minimum=0, maximum=high_maximum / scale, median_below=None),
        high=UnitBounds(unit=high[0], minimum=0, maximum=high_maximum, median_below=None),
        scale=scale,
        source=source,
        low_hints=low[1],
        high_hints=high[1],
    )


# Lab factors: mg/dL -> mmol/L is 10 / molar mass (g/mol); µmol/L adds a factor 1000. Source
# for every lab factor: Young DS. Implementation of SI units for clinical laboratory data.
# Ann Intern Med 1987;106:114-129 (conversion table). Upper bounds are margins above the
# highest values clinicians report, not reference ranges: a cluster is "plausible in its unit"
# when nearly all of it lies inside.
_YOUNG_1987: Final = "Young, Ann Intern Med 1987;106:114 (SI conversion table)"
# Homozygous familial hypercholesterolaemia: untreated LDL-C above 500 mg/dL (Cuchel M et al.,
# Eur Heart J 2014;35:2146). Bound: 3x that.
_FH_UPPER: Final = "Upper: 3x the untreated LDL-C of homozygous FH (Cuchel, Eur Heart J 2014;35:2146)"
_MMOL: Final = ("mmol/L", _MMOL_L)
_MGDL: Final = ("mg/dL", _MG_DL)
UNIT_PAIRS: Final[tuple[UnitPair, ...]] = (
    _lab(
        "glucose",
        _MMOL,
        _MGDL,
        18.016,
        GLUCOSE_MAX_MG_DL,
        f"{_YOUNG_1987}: glucose 180.16 g/mol, 1 mmol/L = 18.016 mg/dL. "
        "Upper: highest recorded blood glucose (Guinness World Records 2008)",
    ),
    _lab("cholesterol", _MMOL, _MGDL, 38.67, 1500.0, f"{_YOUNG_1987}: 1 mmol/L = 38.67 mg/dL. {_FH_UPPER}"),
    _lab("ldl", _MMOL, _MGDL, 38.67, 1500.0, f"{_YOUNG_1987}: 1 mmol/L = 38.67 mg/dL. {_FH_UPPER}"),
    # CETP deficiency: HDL-C above 100 mg/dL (Inazu A et al., N Engl J Med 1990;323:1234).
    _lab(
        "hdl",
        _MMOL,
        _MGDL,
        38.67,
        300.0,
        f"{_YOUNG_1987}: 1 mmol/L = 38.67 mg/dL. Upper: 3x the HDL-C of CETP deficiency "
        "(Inazu, N Engl J Med 1990;323:1234)",
    ),
    _lab(
        "triglycerides",
        _MMOL,
        _MGDL,
        88.57,
        30000.0,
        f"{_YOUNG_1987}: triglycerides as triolein, 1 mmol/L = 88.57 mg/dL. "
        "Upper: a margin above the >10,000 mg/dL of familial chylomicronaemia",
    ),
    _lab(
        "creatinine",
        _MGDL,
        ("µmol/L", _UMOL_L),
        88.4,
        4420.0,
        f"{_YOUNG_1987}: creatinine 113.12 g/mol, 1 mg/dL = 88.4 µmol/L. "
        "Upper: 50 mg/dL, a margin above severe acute kidney injury",
    ),
    UnitPair(
        variable="temperature",
        low=TEMPERATURE_C,
        high=TEMPERATURE_F,
        scale=1.8,
        offset=32.0,
        # Definition of the degree Fahrenheit: NIST SP 811 (2008), B.9: t/°F = 1.8 t/°C + 32.
        source="NIST SP 811 (2008) B.9: °F = 1.8 °C + 32. Bounds: the temperature plausibility range",
        low_hints=("c", "degc", "celsius"),
        high_hints=("f", "degf", "fahrenheit"),
    ),
    UnitPair(
        variable="weight",
        aliases=frozenset({"weight", "bodyweight", "weightkg", "weightlb", "weightlbs"}),
        # The international pound is exactly 0.45359237 kg (NIST SP 811 (2008), B.9). Bounds:
        # below the lightest surviving newborn (212 g, 2021) to above the heaviest recorded
        # human (635 kg), both Guinness World Records.
        low=UnitBounds(unit="kg", minimum=0.2, maximum=650, median_below=None),
        high=UnitBounds(unit="lb", minimum=0.2 / 0.45359237, maximum=650 / 0.45359237, median_below=None),
        scale=1 / 0.45359237,
        source="NIST SP 811 (2008) B.9: 1 lb = 0.45359237 kg. Bounds: Guinness World Records "
        "(lightest surviving newborn 212 g, heaviest human 635 kg)",
        low_hints=("kg", "kgs"),
        high_hints=("lb", "lbs", "pounds"),
    ),
    UnitPair(
        variable="height",
        aliases=frozenset({"height", "heightcm", "heightin", "stature"}),
        # The inch is exactly 2.54 cm (NIST SP 811 (2008), B.9). Bounds: below newborn length
        # to above the tallest recorded human (272 cm, Guinness World Records).
        low=UnitBounds(unit="in", minimum=20 / 2.54, maximum=275 / 2.54, median_below=None),
        high=UnitBounds(unit="cm", minimum=20, maximum=275, median_below=None),
        scale=2.54,
        source="NIST SP 811 (2008) B.9: 1 in = 2.54 cm. Bounds: below newborn length to above the "
        "tallest recorded human, 272 cm (Guinness World Records)",
        low_hints=("in", "inch", "inches"),
        high_hints=("cm",),
    ),
)

# Detection thresholds (contracts.unit_mix). The clusters are a two-component normal mixture
# fitted to log values. Ashman's D above 2 is "clean separation" of two components (Ashman KM,
# Bird CM, Zepf SE. Detecting bimodality in astronomical datasets. Astron J 1994;108:2348): a
# single skewed distribution, however wide, fits two overlapping components (D below 2).
UNIT_MIX_MIN_SEPARATION: Final = 2.0
# The clusters' centres must differ by the unit factor within this ratio either way: patients
# measured in one unit may be sicker than those in the other (centres 12.5 apart for glucose).
UNIT_MIX_RATIO_TOLERANCE: Final = 1.5
# ...and within this share of the factor on the log scale, which binds for the small factors
# (kg/lb 2.2, cm/in 2.54, °C/°F): there a 1.5x slack would admit children beside adults.
UNIT_MIX_RATIO_TOLERANCE_SHARE: Final = 0.25
# Two components must explain the values better than one by a BIC difference of at least 10,
# "very strong" evidence (Kass RE, Raftery AE. Bayes factors. J Am Stat Assoc 1995;90:773).
UNIT_MIX_MIN_BIC_GAIN: Final = 10.0
# Each cluster needs this many values; a column needs twice this many.
UNIT_MIX_MIN_CLUSTER_VALUES: Final = 3
# Share of each cluster that must lie inside its unit's bounds.
UNIT_MIX_IN_RANGE_SHARE: Final = 0.95
# Floor on a fitted component's log spread, so a cluster of identical values cannot collapse
# the fit (log 1.02: values 2% apart).
UNIT_MIX_MIN_LOG_SD: Final = 0.02
UNIT_MIX_EM_ITERATIONS: Final = 200
# A conversion rule is offered only when every value belongs to its cluster with at least this
# posterior probability: no value sits in the valley where its unit cannot be told.
UNIT_MIX_CLEAN_POSTERIOR: Final = 0.99
# ...and the clusters leave an empty stretch of at least this share of the units' log distance
# (glucose: the lowest mg/dL value at least 18 ** 0.25 = 2.1x the highest mmol/L one). Values in
# a narrower valley could be either unit (30: 30 mmol/L or 30 mg/dL), which no cut point settles.
UNIT_MIX_CLEAN_GAP_SHARE: Final = 0.25
UNIT_MIX_RULE_SUFFIX: Final = "_units"
