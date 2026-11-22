# SPDX-License-Identifier: MIT
"""Coverage (question -> plan) and links (plan -> question), checked in code before any number.

The model returns, with its plan, a mapping of the question's phrases to what each became
(``filter``, ``group``, ``measure``, ``unit`` or ``ignored``). Nothing here takes the model's word
for a link. Two checks:

- **Coverage.** Every content word of the question (not a question word, a counting or statistics
  word, politeness, a unit noun, a number, or a word that names what the whole table is about)
  must reach the plan: it names a column the plan reads (by the column's name, its dictionary
  description, its listed values or their labels, or a generic clinical synonym), or a value the
  plan compares with, or it sits in a mapped phrase whose column the plan reads and that phrase
  is verifiably close to that column (sentence-embedding similarity, the column among the closest
  few). A word that reaches nothing means the plan dropped part of the question: a count with no
  filter for "how many women" never passes.
- **Links.** Every column the plan reads for a filter, a group or a measure must be reached the
  same way from some phrase of the question. A column nothing in the question reaches is a
  substitution: a concept the table has no column for is never answered with another column.

Either failure goes back to the model once; if it persists the question is asked back, naming the
words and offering the closest columns, or held saying what is missing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import structlog

from clinical_analytics.core import embeddings
from clinical_analytics.model_query.context import ColumnInfo, DatasetContext, column_search_texts
from clinical_analytics.model_query.units import EVENT_NOUNS

logger = structlog.get_logger()

# A mapped phrase is close enough to a column when their embeddings are at least this similar and
# the column is among the closest few of the table for that phrase.
LINK_SIMILARITY: Final = 0.45
LINK_TOP: Final = 3
# A yes/no column linked by similarity alone needs this much (a topic word must never become a
# yes/no filter on a weak link).
STRONG_SIMILARITY: Final = 0.6
# A column offered as what an unmatched word might mean must be at least this close to it.
CANDIDATE_SIMILARITY: Final = 0.35
# Re-asked choice wording (a choice the user made names the column, or leaves a word out).
MEANS_PHRASE: Final = "By '{word}' I mean {column}."
LEAVE_OUT_PHRASE: Final = "Leave out '{word}'."
_MEANS = re.compile(r"by '([^']+)' i mean ([A-Za-z_][\w]*)\.", re.IGNORECASE)
_LEAVE_OUT = re.compile(r"leave out '([^']+)'\.", re.IGNORECASE)
_CHOICE_WORDS = re.compile(
    r"do not limit it by [\w ,]+\.|give one overall number\.|break it down by \w+\.|only where [^.]*\.|"
    r"count a missing value as (?:not )?meeting it\.|leave out rows with a missing value\.|"
    r"count one (?:when|only when) it was (?:ever|always) true[^.]*\.|use one value per patient[^.]*\.|"
    r"count every visit\.|use \w+\.|give counts, not a share\.|count [^.]* as the share's numerator\.",
    re.IGNORECASE,
)

# Words that carry no condition: question and auxiliary words, counting and statistics words,
# breakdown words, politeness, pronouns, articles, prepositions, negation and time words.
STRUCTURAL: Final = frozenset(
    """
    a an the of in on at to for from with without by per each every all any some both either neither
    and or but not no none nor than then as into onto over under among across between within during
    since until before after about around via versus vs
    what which who whom whose how many much when where why whether is are was were be been being am
    do does did done have has had having can could would should will shall may might must
    there here this that these those it its they them their our ours we us you your i me my he she his her
    please thanks thank tell show give list find get let see look know want need like just only also
    now again still ever always never once twice yet already even
    number count counts counted total totals share shares proportion proportions percent percentage
    percentages fraction fractions rate rates ratio ratios odds average averages mean means median medians
    minimum min maximum max sum sums distribution frequency frequencies prevalence how
    compare compared comparing comparison difference differences differ different significant
    significance test tests tested testing chi square squared fisher exact ttest anova correlation
    correlate correlated associated association relationship related relation likely likelihood
    more less most least higher lower greater fewer common commonly frequent often
    confidence interval ci p value values pvalue estimate estimated regression logistic unadjusted adjusted
    break broken breakdown split splitting group grouped groups grouping category categories each type types
    kind kinds level levels overall whole entire everyone everybody anyone one ones other others
    counting result results measurement measurements reading readings
    data dataset datasets table tables file files sheet registry cohort database spreadsheet record
    recorded records documented documentation available
    year years time times day days month months week weeks
    same different new old
    did done make made use used using
    yes true false
    got get gets getting came come comes coming went go goes going back seen saw took take takes taking
    received receive receives receiving underwent undergo undergoes underwent having gave given giving
    folks folk ones end ended ending turned turn up down out off
    run ran runs perform performed performing conduct conducted done saw
    scored score scores enrolled enrol enroll graded measured assessed evaluated examined collected
    reported observed followed sampled seen visited attended registered entered logged noted listed
    baseline first last latest final initial recent previous prior earlier later most least
    non nonm without among amongst whose who whom someone anybody
    kids kid patient patients people person persons subject subjects participant participants
    level levels value values reading readings result results status measure measures
    above below beyond exceeding exceed exceeds least through thru equal equals exactly approximately about
    consecutive successive between next since until per same days day weeks months
    two three four five six seven eight nine ten dozen multiple several repeat repeated more
    lack lacks lacked lacking absent absence free yet only either neither nor
    unique distinct separate individual
    keep kept keeping own include includes including included exclude excluding excluded report separately
    check checked checking meaning means mean
    """.split()
)  # fmt: skip
# Words too empty to name a column even when its description uses them.
_STOP: Final = frozenset("a an the of in on at to for from with by and or is are was were be this that it its".split())
# Generic clinical synonyms: a column whose own words include a key word is also named by these.
CONCEPTS: Final = {
    frozenset({"sex", "gender"}): frozenset(
        {"sex", "gender", "women", "woman", "men", "man", "female", "females", "male", "males", "girls", "boys"}
    ),
    frozenset({"age", "aged", "dob", "birth"}): frozenset(
        {"age", "aged", "old", "older", "oldest", "elderly", "young", "younger", "youngest", "adult", "adults",
         "child", "children", "pediatric", "paediatric", "infant", "infants", "senior", "seniors", "teen",
         "teens", "adolescent", "adolescents"}
    ),
    frozenset({"died", "death", "deaths", "dead", "deceased", "mortality", "vital", "expired"}): frozenset(
        {"died", "die", "dying", "death", "deaths", "dead", "deceased", "mortality", "survived", "survive",
         "survival", "alive", "fatal", "fatality"}
    ),
    frozenset(
        {"admission", "admitted", "admit", "hospitalized", "hospitalised", "hospitalization", "inpatient"}
    ): frozenset(
        {"admitted", "admission", "admissions", "admit", "hospitalized", "hospitalised", "hospitalization",
         "hospitalisation", "hospital", "inpatient", "inpatients"}
    ),
    frozenset({"year", "yr", "date", "dt", "time"}): frozenset({"year", "yearly", "annual", "annually", "calendar"}),
    frozenset({"smoker", "smoking", "smoke", "tobacco"}): frozenset(
        {"smoker", "smokers", "smoking", "smoke", "tobacco"}
    ),
}  # fmt: skip
_ABBREVIATIONS: Final = {
    "dx": ("diagnosis", "diagnosed", "diagnoses"),
    "tx": ("treatment", "treated", "therapy"),
    "rx": ("prescription", "prescribed", "medication"),
    "hx": ("history",),
    "abx": ("antibiotic", "antibiotics"),
    "yr": ("year",),
    "dob": ("birth", "born"),
    "los": ("length", "stay"),
    "bmi": ("body", "mass"),
    "bp": ("blood", "pressure"),
    "icu": ("intensive",),
    "ed": ("emergency",),
    "pt": ("patient",),
}


def _stem(word: str) -> str:
    """A crude stem shared by question words and column words ("scored"/"score", "adherent"/
    "adherence", "enrolled"/"enrollment" meet)."""
    for suffix in ("ments", "ment", "ences", "ence", "ances", "ance", "ent", "ant", "ies", "ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            word = word[: -len(suffix)] + ("y" if suffix == "ies" else "")
            break
    if word.endswith("e") and len(word) >= 5:
        word = word[:-1]
    if len(word) >= 5 and word[-1] == word[-2] and word[-1] not in "aeiou":
        word = word[:-1]  # "enroll" / "enrol"
    return word


# "C. diff", "E. coli", "S. aureus": an initial and a word are one name ("cdiff").
_INITIAL = re.compile(r"(?<![.\w])([a-z])\.\s*([a-z]{2,})\b")


def _join_initials(text: str) -> str:
    return _INITIAL.sub(r"\1\2", text.casefold())


def _tokens(text: str) -> list[str]:
    folded = text.casefold()
    words = re.findall(r"[a-z]+|\d+(?:\.\d+)?", folded)
    joined = [w for w in re.findall(r"[a-z]+", _join_initials(folded)) if w not in words]
    return words + joined


def stem_word(word: str) -> str:
    """The stem the coverage check compares words by (``_stem``)."""
    return _stem(word)


def word_tokens(text: str) -> list[str]:
    """The words the coverage check reads from a text (``_tokens``)."""
    return _tokens(text)


@dataclass(frozen=True)
class Link:
    """One mapped phrase: the question's words, their role, and the column they became."""

    phrase: str
    role: str
    column: str | None


def parse_mapping(raw: Any) -> list[Link]:
    """The model's mapping, keeping only well-formed entries."""
    if not isinstance(raw, list):
        return []
    links = []
    for item in raw:
        if isinstance(item, dict) and isinstance(item.get("phrase"), str):
            column = item.get("column")
            links.append(Link(item["phrase"], str(item.get("role") or ""), column if isinstance(column, str) else None))
    return links


def column_words(column: ColumnInfo) -> set[str]:
    """Stemmed words that name a column: its name, description, listed values, value labels,
    abbreviation expansions and the generic synonyms of its concept."""
    raw = set(_tokens(column.name.replace("_", " ")))
    for part in list(raw):
        raw.update(_ABBREVIATIONS.get(part, ()))
    # A description's general words ("at the visit", "of the patient", "number") say nothing
    # about which column a question means; only its other words name the column.
    raw.update(
        w
        for w in _tokens(column.description or "")
        if w not in STRUCTURAL and w not in EVENT_NOUNS and w not in _PERSON_NOUNS
    )
    if column.values is not None and column.text:
        for value in column.values:
            raw.update(_tokens(str(value)))
    for _code, label in column.labels:
        raw.update(_tokens(label))
    for keys, synonyms in CONCEPTS.items():
        if keys & raw:
            raw.update(synonyms)
    return {_stem(w) for w in raw if not w.isdigit() and w not in _STOP}


def topic_words(context: DatasetContext) -> set[str]:
    """Stemmed words that name what the whole table is about: in a fifth of its columns or more."""
    counts: dict[str, int] = {}
    for column in context.columns:
        for word in {_stem(w) for w in _tokens(column.name.replace("_", " "))}:
            counts[word] = counts.get(word, 0) + 1
    limit = max(3.0, 0.2 * len(context.columns))
    found = {word for word, n in counts.items() if n >= limit}
    found.update(_stem(w) for w in _tokens(context.title or ""))
    # The dictionary's opening sentence says what the table is ("A clinic registry: one row per
    # visit"); its words name no column unless a column's own words include them.
    opening = re.split(r"(?<=[.:;])\s|\n", (context.notes or "").strip(), maxsplit=1)[0][:160]
    named = set().union(*(column_words(c) for c in context.columns)) if context.columns else set()
    found.update(s for w in _tokens(opening) if (s := _stem(w)) not in named and not w.isdigit())
    return found


def content_chunks(question: str, context: DatasetContext) -> list[tuple[str, ...]]:
    """The question's runs of content words ("relapsing disease", "women", "clinic"), each a tuple
    of words in order. A run must reach the plan through at least one of its words. Structural
    words, unit nouns, numbers, words naming the whole table, and words the user chose to leave
    out break runs and are never part of one."""
    text = _join_initials(_CHOICE_WORDS.sub(" , ", question))
    text = _MEANS.sub(" , ", text)
    text = _LEAVE_OUT.sub(" , ", text)
    left_out = {_stem(w) for found in _LEAVE_OUT.findall(question) for w in _tokens(found)}
    topic = topic_words(context)
    limiting = limiting_nouns(context)
    # "How many <X> visits are in this file": the file is about X. A word that names no column of
    # the table there restates the file's topic and sets no limit.
    whole_file = bool(_WHOLE_FILE.search(question))
    named = set().union(*(column_words(c) for c in context.columns)) if whole_file and context.columns else set()
    chunks: list[tuple[str, ...]] = []
    current: list[str] = []
    for word in re.findall(r"[a-z]+|\d+(?:\.\d+)?|[^\sa-z0-9]", text.casefold()):
        stem = _stem(word)
        breaks = (
            not re.fullmatch(r"[a-z]+", word)
            or len(word) < 2
            or (word in STRUCTURAL and not (word in _LIMITING_NOUNS and word in limiting))
            or (word in _PERSON_NOUNS and word not in limiting)
            or word in EVENT_NOUNS
            or stem in topic
            or stem in left_out
            or (whole_file and stem not in named)
        )
        if breaks:
            if current:
                chunks.append(tuple(current))
            current = []
            continue
        current.append(word)
    if current:
        chunks.append(tuple(current))
    unique: list[tuple[str, ...]] = []
    for chunk in chunks:
        if tuple(_stem(w) for w in chunk) not in [tuple(_stem(w) for w in u) for u in unique]:
            unique.append(chunk)
    return unique


# Patient nouns that also limit the patients (they must reach a filter), by the concept that
# tells them apart: they limit only when the table has a column for it that is not already the
# same for everyone (a table of children has nothing to filter for "kids").
_LIMITING_NOUNS: Final = frozenset(
    {"women", "woman", "men", "man", "children", "child", "kids", "kid", "adults", "adult", "infants", "infant",
     "girls", "boys", "smokers"}
)  # fmt: skip
_SEX_NOUNS: Final = frozenset({"women", "woman", "men", "man", "girls", "boys"})
# Nouns for people in general, or for a sex or age group (those limit only where the table can
# tell them apart). Any other person noun ("smokers", "survivors") names a concept to find.
_PERSON_NOUNS: Final = frozenset(
    {"patient", "patients", "pt", "pts", "people", "person", "persons", "individual", "individuals", "participant",
     "participants", "subject", "subjects", "member", "members", "resident", "residents", "folks", "enrollees",
     "women", "woman", "men", "man", "children", "child", "kids", "kid", "adults", "adult", "infants", "infant",
     "girls", "boys"}
)  # fmt: skip
_CHILD_NOUNS: Final = frozenset({"children", "child", "kids", "kid", "infants", "infant", "girls", "boys"})
_ADULT_NOUNS: Final = frozenset({"adults", "adult"})
_WHOLE_FILE: Final = re.compile(
    r"\b(?:in|of) (?:this|the|our|my) (?:file|registry|dataset|data set|data|database|cohort|spreadsheet|table|"
    r"extract|study)\b|\bin total\b|\baltogether\b|\bin all\b",
    re.IGNORECASE,
)


def limiting_nouns(context: DatasetContext) -> frozenset[str]:
    """The person nouns that limit on this table: a sex noun when a sex column holds both sexes,
    a child or adult noun when an age column spans both sides of 18."""
    found: set[str] = set()
    for column in context.columns:
        words = set(_tokens(column.name.replace("_", " ")))
        labels = {label.casefold() for _code, label in column.labels}
        values = {str(v).casefold() for v in column.values or ()} | labels
        if words & {"sex", "gender"} and len(values) >= 2:
            found |= _SEX_NOUNS
        if "age" in words and column.numeric and column.low is not None and column.high is not None:
            if float(column.low) < 18 <= float(column.high):
                found |= _CHILD_NOUNS | _ADULT_NOUNS
    if any("smok" in c.name.casefold() or "tobacco" in c.name.casefold() for c in context.columns):
        found.add("smokers")
    return frozenset(found & _LIMITING_NOUNS)


@dataclass(frozen=True)
class Findings:
    uncovered: tuple[str, ...]  # content words (stems) that reach nothing in the plan
    unlinked: tuple[str, ...]  # plan columns nothing in the question reaches
    ignored_missing: tuple[str, ...]  # phrases the model mapped as having no column


def _similar(phrases: Sequence[str], context: DatasetContext) -> dict[str, list[tuple[str, float]]]:
    """For each phrase, the table's columns from closest to farthest with their similarity."""
    if not phrases:
        return {}
    texts = column_search_texts(context)
    names = list(texts)
    try:
        encoder = embeddings.get_encoder()
        matrix = np.asarray(
            embeddings.encode_cached(
                encoder, embeddings.texts_key("model_query_columns", list(texts.values())), list(texts.values())
            )
        )
        queries = np.asarray(encoder.encode(list(phrases)))
    except embeddings.ENCODER_LOAD_ERRORS as e:
        logger.warning("model_query_links_unranked", error_type=type(e).__name__)
        return {}
    out: dict[str, list[tuple[str, float]]] = {}
    for phrase, vector in zip(phrases, queries, strict=True):
        scores = matrix @ vector
        order = np.argsort(-scores, kind="stable")
        out[phrase] = [(names[int(i)], float(scores[int(i)])) for i in order]
    return out


def _verified(phrase: str, column: str, ranked: dict[str, list[tuple[str, float]]], threshold: float = 0.0) -> bool:
    best = ranked.get(phrase, [])[:LINK_TOP]
    floor = threshold or LINK_SIMILARITY
    return any(name.casefold() == column.casefold() and score >= floor for name, score in best)


def _is_flag_column(context: DatasetContext, name: str) -> bool:
    """A column holding two values (yes/no, 0/1, Y/N)."""
    column = context.column(name)
    return column is not None and column.values is not None and len(column.values) <= 2


def closest_columns(phrase: str, context: DatasetContext, n: int = 2) -> list[str]:
    """The ``n`` columns closest to ``phrase`` (by name and description words, then embeddings)."""
    stems = {_stem(w) for w in _tokens(phrase)}
    by_words = [c.name for c in context.columns if stems & column_words(c) and not c.is_patient_id]
    ranked = [name for name, score in _similar([phrase], context).get(phrase, []) if score >= CANDIDATE_SIMILARITY]
    found: list[str] = []
    for name in [*by_words, *ranked]:
        column = context.column(name)
        if name not in found and column is not None and not column.is_patient_id and not column.is_encounter_id:
            found.append(name)
    return found[:n]


def check(
    question: str,
    context: DatasetContext,
    *,
    used: Iterable[str],
    values: Iterable[str],
    links: Sequence[Link],
    stated: Iterable[str] = (),
) -> Findings:
    """Coverage and links of one compiled plan.

    ``used``: the table columns the plan reads for filters, groups and measures (not the unit key).
    ``values``: the text values (and their labels) the plan compares with.
    ``stated``: columns tied to the question by a value it states (``grounding``).
    """
    stated_names = {name.casefold() for name in stated}
    used_columns = [c for name in dict.fromkeys(used) if (c := context.column(name)) is not None]
    used_columns = [c for c in used_columns if not c.is_patient_id and not c.is_encounter_id]
    vocab = {c.name: column_words(c) for c in used_columns}
    value_words = {_stem(w) for value in values for w in _tokens(value)}
    for column in used_columns:
        for value in values:
            label = column.label_of(value)
            if label:
                value_words.update(_stem(w) for w in _tokens(label))
    means = {(_stem(w), name.casefold()) for found, name in _MEANS.findall(question) for w in _tokens(found)}
    chunks = content_chunks(question, context)
    mapped = [link for link in links if link.column and link.role != "ignored"]
    ranked = _similar([link.phrase for link in mapped], context)
    verified_links = [
        link
        for link in mapped
        if any(c.name.casefold() == (link.column or "").casefold() for c in used_columns)
        and (
            {_stem(w) for w in _tokens(link.phrase)} & vocab.get(link.column or "", set())
            or (
                _verified(link.phrase, link.column or "", ranked)
                and (
                    not _is_flag_column(context, link.column or "")
                    or _verified(link.phrase, link.column or "", ranked, STRONG_SIMILARITY)
                )
            )
        )
    ]
    reached_by_link = {_stem(w) for link in verified_links for w in _tokens(link.phrase)}
    named_in_question = {name for _stem_word, name in means}
    used_names = {c.name.casefold() for c in used_columns}

    def reaches(word: str) -> bool:
        if word in value_words or word in reached_by_link or any(word in vocab[c.name] for c in used_columns):
            return True
        return any(stem == word and name in used_names for stem, name in means)

    open_chunks = [chunk for chunk in chunks if not any(reaches(_stem(word)) for word in chunk)]
    # A run no word of which names a used column can still be close to one ("sat" and an
    # oxygen-saturation column): accepted when the column is among the closest few for it. A yes/no
    # column is only accepted on a strong link (a weak one is how a topic word becomes a filter).
    near = _similar([" ".join(chunk) for chunk in open_chunks], context)
    uncovered = []
    for chunk in open_chunks:
        text = " ".join(chunk)
        best = near.get(text, [])[:LINK_TOP]
        close = any(
            name.casefold() in used_names
            and score >= (STRONG_SIMILARITY if _is_flag_column(context, name) else LINK_SIMILARITY)
            for name, score in best
        )
        if not close:
            uncovered.append(text)
    question_stems = {_stem(w) for w in _tokens(question)}
    unlinked = []
    for column in used_columns:
        if vocab[column.name] & question_stems or column.name.casefold() in question.casefold():
            continue
        if column.name.casefold() in named_in_question or column.name.casefold() in stated_names:
            continue
        if any((link.column or "").casefold() == column.name.casefold() for link in verified_links):
            continue
        unlinked.append(column.name)
    ignored_missing = tuple(
        link.phrase
        for link in links
        if link.role == "ignored"
        and any(_stem(w) in {_stem(x) for c in uncovered for x in c.split()} for w in _tokens(link.phrase))
    )
    return Findings(tuple(uncovered), tuple(unlinked), ignored_missing)


# Name parts that say when a measure was taken: the same measure at another time is a sibling.
_TIME_PART = re.compile(
    r"^(?:baseline|base|bl|pre|post|followup|follow|up|fu|final|initial|first|last|latest|end|start|discharge|"
    r"admission|admit|entry|exit|enrol|enrollment|month|months|mo|m|wk|week|weeks|yr|year|years|day|days|d|"
    r"v|visit|t|time|current|prior|previous|recent|\d+|[a-z]{1,2}\d+|\d+[a-z]{1,3})$"
)
_TIME_SYNONYMS: Final = {
    "baseline": {"baseline", "base", "bl", "initial", "first", "start", "entry", "enrollment", "pre"},
    "followup": {"followup", "follow", "fu", "final", "last", "latest", "end", "exit", "post", "recent", "current"},
}


def _name_parts(name: str) -> list[str]:
    return re.findall(r"[a-z]+|\d+", name.casefold())


def _measure_stem(name: str) -> tuple[str, ...]:
    return tuple(p for p in _name_parts(name) if not _TIME_PART.match(p))


def siblings(column: ColumnInfo, context: DatasetContext) -> list[str]:
    """Other columns holding the same measure at another time (``a1c_baseline`` and ``a1c_12m``)."""
    stem = _measure_stem(column.name)
    if not stem or stem == tuple(_name_parts(column.name)):
        return []
    return [
        c.name
        for c in context.columns
        if c.name != column.name and c.numeric == column.numeric and _measure_stem(c.name) == stem
    ]


def measure_choice_needed(question: str, column: ColumnInfo, context: DatasetContext) -> list[str]:
    """The candidate columns to offer when a measure has siblings the question does not choose
    between (by naming the column, or a word of its time); none when it is clear."""
    others = siblings(column, context)
    if not others:
        return []
    folded = question.casefold()
    if any(name.casefold() in folded for name in [column.name, *others]):
        return []
    asked = set(_tokens(question))
    for kind, words in _TIME_SYNONYMS.items():
        if asked & words:
            asked |= words | {kind}

    def time_parts(name: str) -> set[str]:
        parts = set(_name_parts(name)) - set(_measure_stem(column.name))
        return parts | {d for p in parts for d in re.findall(r"\d+", p)}

    own = time_parts(column.name)
    rivals = {p for other in others for p in time_parts(other)}
    own_hits, rival_hits = own & asked, (rivals - own) & asked
    if own_hits and not rival_hits:
        return []
    return [column.name, *others]
