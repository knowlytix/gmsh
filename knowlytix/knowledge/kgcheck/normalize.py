# SPDX-License-Identifier: Apache-2.0
"""kgcheck's comparison keys for labels, versioned and kept beside the raw labels.

These keys decide when two spellings are treated as the same term in
collision and reconciliation checks. They follow the lexical part of
OntoDisc's ``normalize_entity`` (markdown emphasis and backticks removed,
underscores, hyphens, slashes and dashes read as spaces, whitespace collapsed,
lower case) but are kgcheck's own definition: OntoDisc's relation slugs add a
``has_`` prefix and its entity normalization rewrites dates, neither of which
is applied here. Label normalization applies to labels only: a numeric token is
compared by its exact decimal value, so ``-5000`` and ``5000`` stay different
while ``5000.0`` and ``5000`` agree, and an ISO date is compared verbatim.
Resolving a stored token to a declared term is a separate question, answered
the way the producer answers it: ``entity_resolution_key`` and
``relation_resolution_key`` reproduce OntoDisc's surface normalization. A change
to any of these functions changes ``NORMALIZATION_VERSION``.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from decimal import Context, Decimal

NORMALIZATION_VERSION = "3"

_EMPHASIS = re.compile(r"\*\*(.+?)\*\*")
_SEPARATORS = re.compile(r"[_\-/–—]+")
_SPACE = re.compile(r"\s+")
# A stored numeric token. OntoDisc stores numbers in canonical form ("$5,000" ->
# "5000") before persisting, so stored numeric tails match this pattern; a test
# pins it to the producer's canonical_number on every fixture token.
NUMERIC_TOKEN = re.compile(r"[+-]?[0-9]+(?:\.[0-9]+)?")  # ASCII digits only
ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


def label_key(label: str) -> str:
    """The comparison key of a label: ``"Account_Tier"`` and ``"account tier"`` agree."""
    text = unicodedata.normalize("NFKC", label)
    text = _EMPHASIS.sub(r"\1", text).replace("`", "").replace("*", "")
    text = _SEPARATORS.sub(" ", text)
    return _SPACE.sub(" ", text).strip().casefold()


def is_numeric_token(token: str) -> bool:
    return NUMERIC_TOKEN.fullmatch(token) is not None


def value_key(token: str) -> str:
    """The comparison key of an entity slot: exact decimal value, ISO date, or label key."""
    if is_numeric_token(token):
        number = Decimal(token)
        if not number:
            return "num:0"
        exact = Context(prec=len(token) + 2)  # never round: long identifiers stay distinct
        return "num:" + format(number.normalize(exact), "f")
    if ISO_DATE.fullmatch(token):
        return "date:" + token
    return label_key(token)


def triple_key(head: str, relation: str, obj: str) -> tuple[str, str, str]:
    return value_key(head), label_key(relation), value_key(obj)


# --- resolution keys: OntoDisc's own surface normalization ------------------------------
# ``label_key`` is kgcheck's comparison key for collisions and similarity. Whether a
# stored token resolves to a declared term is the producer's decision, so these keys
# reproduce ``normalize_entity`` / ``normalize_relation`` (knowlytix.core.ontodisc.kg
# .canonicalize) and ``canonical_date`` (.dates) exactly; a parity test pins them.
_MONTHS = {m: i for i, m in enumerate(
    ("january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"), 1)}
_MONTH = "(?:" + "|".join(_MONTHS) + ")"
_DATE = re.compile(
    rf"\b(?:\d{{4}}-\d{{2}}-\d{{2}}|{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?[,]?\s+\d{{4}}|"
    rf"\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}\s+\d{{4}})\b", re.I)
DATE_SURFACE = _DATE  # a full date as written in prose, for excluding its digits from numerals
_RELATION_VERBS = ("has_", "is_", "in_", "passes", "fails", "opposite_of", "is_functional",
                   "requires", "includes", "part_of", "type_of", "located_in", "related_to")


def _canonical_date(value: str) -> str | None:
    text = str(value).strip().lower()
    if not _DATE.fullmatch(text):
        return None
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            return date.fromisoformat(text).isoformat()
        parts = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text).replace(",", "").split()
        month, day = (parts[0], parts[1]) if parts[0] in _MONTHS else (parts[1], parts[0])
        return date(int(parts[2]), _MONTHS[month], int(day)).isoformat()
    except ValueError:
        return None


def entity_resolution_key(token: str) -> str:
    """The key under which OntoDisc resolves an entity surface to a declared term."""
    full_date = _canonical_date(token)
    if full_date is not None:
        return full_date
    text = _EMPHASIS.sub(r"\1", str(token)).replace("`", "").replace("*", "")
    text = re.sub(r"[_/–—]+|(?<=\w)-|-(?!\d)", " ", text)  # a minus before a digit stays
    return _SPACE.sub(" ", text).strip().lower()


def relation_resolution_key(token: str) -> str:
    """The key under which OntoDisc resolves a relation surface: its ``has_`` slug."""
    text = _EMPHASIS.sub(r"\1", str(token).strip()).replace("`", "")
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    if not text:
        return "related_to"
    return text if text.startswith(_RELATION_VERBS) or text in _RELATION_VERBS else f"has_{text}"
