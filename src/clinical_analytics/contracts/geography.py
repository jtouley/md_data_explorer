# SPDX-License-Identifier: MIT
"""US geography for joining public aggregate tables on state (ADR014 §10).

Public tables name a state in different ways: GBD writes "Michigan", CDC WONDER writes "Michigan"
with a "State Code" of 26, AtlasPlus and PLACES write "MI" or a FIPS code, some files write
"26000" or "Michigan, US". Everything joins on the two-digit state FIPS code, as text ("06", not
6), so "Michigan" and 26 meet on one key and a table that writes Georgia as "13" never joins
California's "06" by accident.

Reference data only (public USPS and ANSI/FIPS codes, 50 states plus DC). No dataset values.

Levels that are not states are classified so the join can say why they were left out:

* the nation ("United States", "US"): a total of the states, never ranked among them;
* a county or a local health jurisdiction ("Wayne County", "Detroit", "Out-Wayne"): part of a
  state, never joined at state grain. Out-Wayne means Wayne County excluding Detroit (Detroit
  has its own health department), so it is never treated as Wayne County.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

GeoKind: TypeAlias = Literal["state", "nation", "substate", "territory", "region", "country", "other", "unknown"]


@dataclass(frozen=True)
class State:
    fips: str  # two digits, zero-padded
    usps: str
    name: str


STATES: Final[tuple[State, ...]] = (
    State("01", "AL", "Alabama"),
    State("02", "AK", "Alaska"),
    State("04", "AZ", "Arizona"),
    State("05", "AR", "Arkansas"),
    State("06", "CA", "California"),
    State("08", "CO", "Colorado"),
    State("09", "CT", "Connecticut"),
    State("10", "DE", "Delaware"),
    State("11", "DC", "District of Columbia"),
    State("12", "FL", "Florida"),
    State("13", "GA", "Georgia"),
    State("15", "HI", "Hawaii"),
    State("16", "ID", "Idaho"),
    State("17", "IL", "Illinois"),
    State("18", "IN", "Indiana"),
    State("19", "IA", "Iowa"),
    State("20", "KS", "Kansas"),
    State("21", "KY", "Kentucky"),
    State("22", "LA", "Louisiana"),
    State("23", "ME", "Maine"),
    State("24", "MD", "Maryland"),
    State("25", "MA", "Massachusetts"),
    State("26", "MI", "Michigan"),
    State("27", "MN", "Minnesota"),
    State("28", "MS", "Mississippi"),
    State("29", "MO", "Missouri"),
    State("30", "MT", "Montana"),
    State("31", "NE", "Nebraska"),
    State("32", "NV", "Nevada"),
    State("33", "NH", "New Hampshire"),
    State("34", "NJ", "New Jersey"),
    State("35", "NM", "New Mexico"),
    State("36", "NY", "New York"),
    State("37", "NC", "North Carolina"),
    State("38", "ND", "North Dakota"),
    State("39", "OH", "Ohio"),
    State("40", "OK", "Oklahoma"),
    State("41", "OR", "Oregon"),
    State("42", "PA", "Pennsylvania"),
    State("44", "RI", "Rhode Island"),
    State("45", "SC", "South Carolina"),
    State("46", "SD", "South Dakota"),
    State("47", "TN", "Tennessee"),
    State("48", "TX", "Texas"),
    State("49", "UT", "Utah"),
    State("50", "VT", "Vermont"),
    State("51", "VA", "Virginia"),
    State("53", "WA", "Washington"),
    State("54", "WV", "West Virginia"),
    State("55", "WI", "Wisconsin"),
    State("56", "WY", "Wyoming"),
)

# US territories: never states (left out of a state join, named), never foreign countries.
TERRITORIES: Final[tuple[State, ...]] = (
    State("60", "AS", "American Samoa"),
    State("66", "GU", "Guam"),
    State("69", "MP", "Northern Mariana Islands"),
    State("72", "PR", "Puerto Rico"),
    State("78", "VI", "U.S. Virgin Islands"),
)
_TERRITORY_KEYS: Final = frozenset(
    {t.fips for t in TERRITORIES}
    | {t.usps.casefold() for t in TERRITORIES}
    | {t.name.casefold() for t in TERRITORIES}
    | {
        "virgin islands",
        "us virgin islands",
        "u.s. virgin islands",
        "virgin islands, u.s.",
        "virgin islands of the u.s.",
        "virgin islands of the united states",
        "united states virgin islands",
        "commonwealth of the northern mariana islands",
        "northern mariana islands",
    }
)
# Census regions and divisions, HHS regions: groups of states, never countries.
_REGIONS: Final = frozenset(
    {
        "northeast", "midwest", "south", "west", "north central",
        "new england", "middle atlantic", "mid-atlantic", "east north central", "west north central",
        "south atlantic", "east south central", "west south central", "mountain", "pacific",
    }
)  # fmt: skip
_HHS_REGION: Final = re.compile(r"^(hhs\s+)?region\s*(10|[1-9])(\s*[-–:(].*)?$", re.IGNORECASE)
# Labels that sit beside places in public tables and are neither a place nor a country.
OTHER_PLACE_LABELS: Final = frozenset(
    {"unknown", "other", "total", "all", "overall", "not stated", "missing", "entire network", "all states", "none"}
)
# Countries and areas as GBD and WHO name them (lowercase). A table listing any of these is
# international, which makes a state name that is also a country's ("Georgia") ambiguous.
COUNTRIES: Final = frozenset(
    {
        "afghanistan", "albania", "algeria", "andorra", "angola", "antigua and barbuda", "argentina", "armenia",
        "australia", "austria", "azerbaijan", "bahamas", "bahrain", "bangladesh", "barbados", "belarus", "belgium",
        "belize", "benin", "bhutan", "bolivia", "bolivia (plurinational state of)", "bosnia and herzegovina",
        "botswana", "brazil", "brunei darussalam", "bulgaria", "burkina faso", "burundi", "cabo verde", "cambodia",
        "cameroon", "canada", "central african republic", "chad", "chile", "china", "colombia", "comoros", "congo",
        "costa rica", "cote d'ivoire", "côte d'ivoire", "croatia", "cuba", "cyprus", "czechia", "czech republic",
        "democratic people's republic of korea", "democratic republic of the congo", "denmark", "djibouti",
        "dominica", "dominican republic", "ecuador", "egypt", "el salvador", "equatorial guinea", "eritrea",
        "estonia", "eswatini", "ethiopia", "fiji", "finland", "france", "gabon", "gambia", "germany", "ghana",
        "greece", "grenada", "guatemala", "guinea", "guinea-bissau", "guyana", "haiti", "honduras", "hungary",
        "iceland", "india", "indonesia", "iran", "iran (islamic republic of)", "iraq", "ireland", "israel", "italy",
        "jamaica", "japan", "jordan", "kazakhstan", "kenya", "kiribati", "kuwait", "kyrgyzstan",
        "lao people's democratic republic", "latvia", "lebanon", "lesotho", "liberia", "libya", "lithuania",
        "luxembourg", "madagascar", "malawi", "malaysia", "maldives", "mali", "malta", "marshall islands",
        "mauritania", "mauritius", "mexico", "micronesia (federated states of)", "monaco", "mongolia", "montenegro",
        "morocco", "mozambique", "myanmar", "namibia", "nauru", "nepal", "netherlands", "new zealand", "nicaragua",
        "niger", "nigeria", "north macedonia", "norway", "oman", "pakistan", "palau", "palestine", "panama",
        "papua new guinea", "paraguay", "peru", "philippines", "poland", "portugal", "qatar", "republic of korea",
        "republic of moldova", "romania", "russian federation", "russia", "rwanda", "saint kitts and nevis",
        "saint lucia", "saint vincent and the grenadines", "samoa", "san marino", "sao tome and principe",
        "saudi arabia", "senegal", "serbia", "seychelles", "sierra leone", "singapore", "slovakia", "slovenia",
        "solomon islands", "somalia", "south africa", "south sudan", "spain", "sri lanka", "sudan", "suriname",
        "sweden", "switzerland", "syrian arab republic", "syria", "taiwan", "tajikistan", "thailand", "timor-leste",
        "togo", "tonga", "trinidad and tobago", "tunisia", "turkey", "türkiye", "turkmenistan", "tuvalu", "uganda",
        "ukraine", "united arab emirates", "united kingdom", "united republic of tanzania", "tanzania",
        "united states of america", "uruguay", "uzbekistan", "vanuatu", "venezuela",
        "venezuela (bolivarian republic of)", "viet nam", "vietnam", "yemen", "zambia", "zimbabwe", "greenland",
        "bermuda", "cook islands", "niue", "tokelau",
    }
)  # fmt: skip
_BY_FIPS: Final = {s.fips: s for s in STATES}
_BY_USPS: Final = {s.usps.casefold(): s for s in STATES}
_BY_NAME: Final = {s.name.casefold(): s for s in STATES} | {
    "washington dc": _BY_FIPS["11"],
    "washington, dc": _BY_FIPS["11"],
}

NATION_LEVELS: Final = frozenset({"united states", "us", "u.s.", "usa", "united states of america", "national"})
# A level naming part of a state: a county, a city or a local health jurisdiction. "Out-Wayne" is
# Wayne County excluding Detroit; "Detroit" is the city's own jurisdiction.
_SUBSTATE: Final = re.compile(
    r"\b(county|parish|borough|census area|municipality|city|out[- ]?wayne|detroit|excluding|excl\.?)\b",
    re.IGNORECASE,
)
# "Michigan, US" / "Michigan (state)" / "26000" (state FIPS written as a five-digit place code).
_STATE_SUFFIX: Final = re.compile(r"\s*(,\s*(us|usa|united states)|\(state\))\s*$", re.IGNORECASE)
_FIVE_DIGIT_STATE: Final = re.compile(r"^(\d{2})000$")

# Named groups of states a question may use ("Michigan and its Great Lakes neighbours"). Census
# Bureau regions; the Great Lakes states are the eight that border the lakes.
STATE_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "great lakes": ("17", "18", "26", "27", "36", "39", "42", "55"),
    "midwest": ("17", "18", "19", "20", "26", "27", "29", "31", "38", "39", "46", "55"),
    "northeast": ("09", "23", "25", "33", "34", "36", "42", "44", "50"),
    "south": (
        "01",
        "05",
        "10",
        "11",
        "12",
        "13",
        "21",
        "22",
        "24",
        "28",
        "37",
        "40",
        "45",
        "47",
        "48",
        "51",
        "54",
    ),  # fmt: skip
    "west": ("02", "04", "06", "08", "15", "16", "30", "32", "35", "41", "49", "53", "56"),
}


@dataclass(frozen=True)
class Place:
    """What a location level is: a state (with its FIPS), the nation, part of a state, or unknown."""

    kind: GeoKind
    state: State | None = None


# A column whose bare numbers are place codes (FIPS), by its name. Elsewhere 26 is just 26 (a
# patient number, a week): numbers are read as FIPS only in a column named for a place.
GEO_CODE_COLUMN: Final = re.compile(r"fips|state|county|location|geo|area|place|jurisdiction", re.IGNORECASE)


def is_geo_code_column(column: str) -> bool:
    return bool(GEO_CODE_COLUMN.search(column))


def classify(level: object, codes: bool = False) -> Place:
    """Classify one location level (name, USPS code, or, with ``codes``, FIPS as text or number).

    With ``codes`` (the column is named for a place), numbers are FIPS codes: 26 and "26" are
    Michigan; 26000 is Michigan written as a place code; any other five-digit code is a county,
    part of a state, never a state. Without it a bare number is unknown, never a state.
    """
    if level is None or isinstance(level, bool):
        return Place("unknown")
    if isinstance(level, int):
        if not codes:
            return Place("unknown")
        text = f"{level:02d}" if 0 < level < 100 else str(level)
    elif isinstance(level, float):
        if level != level or not level.is_integer():
            return Place("unknown")
        return classify(int(level), codes)
    else:
        text = str(level).strip()
    folded = _STATE_SUFFIX.sub("", text).strip().casefold()
    if folded in _TERRITORY_KEYS and (codes or not folded.isdigit()):
        return Place("territory")
    if folded in _REGIONS or _HHS_REGION.match(folded):
        return Place("region")
    if folded in OTHER_PLACE_LABELS:
        return Place("other")
    if folded in COUNTRIES and folded not in _BY_NAME and folded not in NATION_LEVELS:
        return Place("country")
    if folded in NATION_LEVELS:
        return Place("nation")
    if folded in _BY_NAME:
        return Place("state", _BY_NAME[folded])
    if len(folded) == 2 and folded in _BY_USPS:
        return Place("state", _BY_USPS[folded])
    if folded.isdigit():
        if not codes:
            return Place("unknown")
        if len(folded) <= 2:
            state = _BY_FIPS.get(folded.zfill(2))
            return Place("state", state) if state else Place("unknown")
        five = _FIVE_DIGIT_STATE.match(folded.zfill(5))
        if five and five.group(1) in _BY_FIPS:
            return Place("state", _BY_FIPS[five.group(1)])
        if len(folded) in (4, 5):
            return Place("substate")
        return Place("unknown")
    if _SUBSTATE.search(text):
        return Place("substate")
    return Place("unknown")


def state_named(text: str) -> list[State]:
    """States a question names by full name (longest first, so "West Virginia" is not "Virginia")."""
    folded = " " + re.sub(r"[^a-z]+", " ", text.casefold()) + " "
    found: list[State] = []
    for name in sorted(_BY_NAME, key=len, reverse=True):
        phrase = " " + re.sub(r"[^a-z]+", " ", name).strip() + " "
        if phrase in folded:
            folded = folded.replace(phrase, " ")
            found.append(_BY_NAME[name])
    return list(dict.fromkeys(found))


def groups_named(text: str) -> list[str]:
    """Names of STATE_GROUPS the question uses ("Great Lakes", "Midwest"). Whole state names are
    read first, so "South Carolina" or "West Virginia" never names the South or the West."""
    folded = " " + re.sub(r"[^a-z]+", " ", text.casefold()) + " "
    for name in sorted(_BY_NAME, key=len, reverse=True):
        folded = folded.replace(" " + re.sub(r"[^a-z]+", " ", name).strip() + " ", " ")
    return [name for name in STATE_GROUPS if re.search(rf"\b{re.escape(name)}\b", folded)]


# State names that are also country names: in a table that lists countries too, the level alone
# can't say which is meant.
COUNTRY_NAMESAKES: Final = frozenset({"georgia"})


def mentions_wayne(text: str) -> bool:
    """The question asks about Wayne County (the Detroit / out-Wayne trap)."""
    return bool(re.search(r"\bwayne\b", text, re.IGNORECASE))


def is_out_wayne(level: object) -> bool:
    """Wayne County excluding Detroit ("Out-Wayne", "Wayne County (excluding Detroit)")."""
    if not isinstance(level, str):
        return False
    folded = level.casefold()
    return bool(re.search(r"out[- ]?wayne", folded)) or ("wayne" in folded and "exclud" in folded)


def is_detroit(level: object) -> bool:
    return isinstance(level, str) and bool(re.fullmatch(r"\s*(city of\s+)?detroit(\s+city)?\s*", level, re.IGNORECASE))
