# SPDX-License-Identifier: MIT
"""Words of a question that carry no limit, shared by every check that reads a question.

The cohort peel (cohort_constraints.peel_cohort), the gap check run before execution
(cohort_constraints.unapplied_constraints) and the word-accounting gate
(word_accounting.unused_words) must agree on which words say nothing about the patients.
If one of them skipped a word the others treat as a limit, a question could pass one check
and fail another, or worse, a limit could be dropped in one place and never noticed.

Adding a word here makes every check accept it in any question, so the lists stay small and
reviewed. A word that can narrow the patients (a negation, "or", a direction such as "over",
"only", "currently") never belongs here; tests/core/test_question_vocabulary.py enforces that
for the logic words the gate knows about.
"""

from __future__ import annotations

# Articles, pronouns, auxiliaries, prepositions and politeness: grammar, never a limit.
FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        "a", "all", "among", "amongst", "an", "and", "any", "are", "as", "at", "be", "been", "being",
        "can", "could", "did", "do", "does", "for", "from", "had", "has", "have", "having", "i", "in",
        "is", "it", "its", "me", "my", "of", "on", "our", "please", "s", "that", "the", "their", "them",
        "there", "these", "they", "this", "those", "to", "up", "us", "was", "we", "were", "what", "which",
        "who", "whom", "whose", "will", "with", "within", "would", "you", "your",
    }
)  # fmt: skip

# Nouns that name the rows being counted or the dataset itself, not a subset of it.
SUBJECT_WORDS: frozenset[str] = frozenset(
    {
        "case", "cases", "cohort", "data", "dataset", "group", "groups", "individuals", "participant",
        "participants", "patient", "patients", "people", "person", "persons", "study", "subject",
        "subjects", "user", "users",
    }
)  # fmt: skip

# Verbs that only link a patient to a stored value ("patients who received placebo").
LINK_WORDS: frozenset[str] = frozenset({"given", "received", "receiving", "taking", "treated", "use", "uses", "using"})

NO_LIMIT_WORDS: frozenset[str] = FUNCTION_WORDS | SUBJECT_WORDS | LINK_WORDS

# Words that turn "<value>" into "everyone except <value>" when they stand right before it.
NEGATION_WORDS: frozenset[str] = frozenset(
    {"not", "non", "no", "without", "excluding", "exclude", "except", "outside", "never"}
)
