# SPDX-License-Identifier: Apache-2.0
"""Quantities stated in evidence text: a numeral with what the text says around it.

A numeral is read as OntoDisc reads it (comma groups, a magnitude word) and
annotated with what the surrounding words state explicitly: a currency or unit
next to it, a period after it ("per day", "monthly") and a bound before or after
it ("at least", "up to", "or more"). Nothing is inferred from a bare number: a
numeral with no unit word has no unit. The ``$`` sign is read as some dollar
currency, compatible with any declared dollar unit and with ``dollar`` itself.

The lexicon is English and deliberately small; a word outside it is not read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

_CURRENCY_BEFORE = r"(?P<pre>US\$|USD|EUR|GBP|JPY|CAD|AUD|\$|€|£|¥)\s?"
_NUMBER = (r"(?<![\w.])" + rf"(?:{_CURRENCY_BEFORE})?" +
           r"(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?P<frac>\.\d+)?(?!\d)"
           r"(?:\s*(?P<mag>thousand|million|billion|bn|k)\b)?"
           r"(?:\s*(?P<post>%|percent\b|USD\b|EUR\b|GBP\b|JPY\b|dollars?\b|euros?\b|pounds?\b|yen\b"
           r"|days?\b|weeks?\b|months?\b|years?\b))?")
_PATTERN = re.compile(_NUMBER, re.I)
_MAGNITUDE = {"thousand": 1_000, "k": 1_000, "million": 1_000_000, "billion": 1_000_000_000,
              "bn": 1_000_000_000}
_UNITS = {"us$": "dollar", "$": "dollar", "usd": "USD", "eur": "EUR", "€": "EUR", "gbp": "GBP",
          "£": "GBP", "jpy": "JPY", "¥": "JPY", "cad": "CAD", "aud": "AUD", "%": "percent",
          "percent": "percent", "dollar": "dollar", "dollars": "dollar", "euro": "EUR",
          "euros": "EUR", "pound": "GBP", "pounds": "GBP", "yen": "JPY", "day": "day",
          "days": "day", "week": "week", "weeks": "week", "month": "month", "months": "month",
          "year": "year", "years": "year"}
DOLLARS = frozenset({"dollar", "USD", "CAD", "AUD", "NZD", "SGD", "HKD"})
_PERIOD = re.compile(r"\s*(?:(?:per|a|each|every)\s+(?P<unit>day|week|month|quarter|year|"
                     r"transaction)\b|(?P<adverb>daily|weekly|monthly|quarterly|annually|yearly|"
                     r"annual)\b)", re.I)
_ADVERB = {"daily": "day", "weekly": "week", "monthly": "month", "quarterly": "quarter",
           "annually": "year", "yearly": "year", "annual": "year"}
# Tried in order, so a negated bound ("no more than", "may not exceed") is read before the
# plain phrase it ends with ("more than", "exceed").
_BEFORE = (("maximum", re.compile(r"(?:no|not|never) (?:more|greater|higher) than\s*$|"
                                  r"(?:not|never|cannot|can't|won't) (?:to )?exceed\s*$", re.I)),
           ("minimum", re.compile(r"(?:no|not|never) (?:less|fewer|lower) than\s*$|"
                                  r"(?:not|never) (?:be )?below\s*$", re.I)),
           ("minimum", re.compile(r"(?:at least|minimum(?: of)?|exceeds?|exceeding|more than|"
                                  r"greater than|over|above)\s*$", re.I)),
           ("maximum", re.compile(r"(?:at most|maximum(?: of)?|up to|less than|under|below|"
                                  r"limit(?:ed)? (?:of|to))\s*$", re.I)),
           ("exact", re.compile(r"(?:exactly|precisely)\s*$", re.I)))
_AFTER = (("minimum", re.compile(r"\s*(?:or more|or greater|or above|and above|and over)\b", re.I)),
          ("maximum", re.compile(r"\s*(?:or less|or fewer|or below|and below|at most)\b", re.I)))
# "daily limit of $5,000": the adjective must qualify a quantity noun ("annual report lists
# 12 branches" states no period)
_DAILY_ADJECTIVE = re.compile(r"(?P<adverb>daily|weekly|monthly|quarterly|annual|yearly)\s+"
                              r"(?:\w+\s+)?(?:limits?|caps?|thresholds?|maximum|minimum|amounts?|"
                              r"fees?|rates?|totals?|withdrawals?|deposits?|payments?|spend(?:ing)?|"
                              r"allowances?|transfers?)\b(?:\s+(?:of|is|are|at|up to|to))*\s*$",
                              re.I)


@dataclass(frozen=True)
class Quantity:
    """A numeral in text: its value as written, the magnitude-scaled value and annotations."""

    text: str
    value: Decimal  # the numeral without its magnitude word
    scaled: Decimal  # with it
    unit: str | None
    period: str | None
    qualifier: str | None
    start: int
    end: int


def quantities(text: str) -> list[Quantity]:
    """Every numeral of ``text`` with its explicit annotations, in order."""
    out = []
    for m in _PATTERN.finditer(text):
        try:
            value = Decimal(m["int"].replace(",", "") + (m["frac"] or ""))
        except InvalidOperation:
            continue
        magnitude = _MAGNITUDE.get((m["mag"] or "").lower(), 1)
        unit = _unit(m["pre"]) or _unit(m["post"])
        out.append(Quantity(m[0].strip(), value, value * magnitude, unit,
                            _period(text, m.start(), m.end()),
                            _qualifier(text, m.start(), m.end()), m.start(), m.end()))
    return out


def _unit(token: str | None) -> str | None:
    return _UNITS.get(token.lower()) if token else None


def _period(text: str, start: int, end: int) -> str | None:
    bound = next((m for _, p in _AFTER if (m := p.match(text, end))), None)
    after = _PERIOD.match(text, end) or (_PERIOD.match(text, bound.end()) if bound else None)
    if after:
        return after["unit"].lower() if after["unit"] else _ADVERB[after["adverb"].lower()]
    before = _DAILY_ADJECTIVE.search(text[max(0, start - 40):start])
    return _ADVERB[before["adverb"].lower()] if before else None


def _qualifier(text: str, start: int, end: int) -> str | None:
    window = text[max(0, start - 40):start]
    for name, pattern in _BEFORE:
        if pattern.search(window):
            return name
    for name, pattern in _AFTER:
        if pattern.match(text, end):
            return name
    return None


def compatible_unit(declared: str, observed: str) -> bool:
    """True when an observed unit can be the declared one; ``$`` is any dollar currency."""
    if declared.lower() == observed.lower():
        return True
    if observed == "dollar":
        return declared.upper() in DOLLARS or declared.lower() == "dollar"
    return declared.lower() == "dollar" and observed in DOLLARS
