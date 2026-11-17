# SPDX-License-Identifier: Apache-2.0
"""Numbers in rule text that are not quantities: calendar dates and document references.

OntoDisc annotates these in ``rules.json`` (``nonquantity_numbers``) so numeric
memory never reads them as thresholds. ``nonquantity_numbers`` reproduces the
producer's recognizer (``knowlytix.discovery.ontodisc.rules.nonquantity_numbers``);
it is reimplemented, not imported, so the audit does not execute the code whose
output it checks, and a test pins the two to each other.
"""

from __future__ import annotations

import re
from datetime import datetime

_MONTH = ("(?:January|February|March|April|May|June|July|August|September|October|"
          "November|December)")
_DATES = ((re.compile(rf"\b{_MONTH} \d{{1,2}}, \d{{4}}\b", re.I), "%B %d, %Y"),
          (re.compile(rf"\b{_MONTH} \d{{4}}\b", re.I), "%B %Y"),
          (re.compile(r"\b\d{4}-\d{2}-\d{2}\b", re.I), "%Y-%m-%d"))
_REFERENCE = re.compile(r"\b(?:pages?|chapter|section|issue|version|revision)\s+\d+(?:\.\d+)*"
                        r"(?:\([a-z0-9]+\))*(?!\w)", re.I)
_UNIT_AFTER = re.compile(r"\s+(?:mg|kg|g|ml|liters?|metres?|meters?|seconds?|minutes?|hours?|"
                         r"days?|weeks?|years?|percent)\b", re.I)


def nonquantity_numbers(text: str) -> list[dict[str, object]]:
    """Calendar dates and labelled document references in ``text``, by position."""
    found: list[dict[str, object]] = []
    for pattern, fmt in _DATES:
        for m in pattern.finditer(text):
            try:
                datetime.strptime(m[0], fmt)
            except ValueError:
                continue
            found.append(_annotation(text, m.start(), m.end(), "calendar_date"))
    for m in _REFERENCE.finditer(text):
        if not _UNIT_AFTER.match(text, m.end()):
            found.append(_annotation(text, m.start(), m.end(), "document_reference"))
    return sorted(found, key=lambda a: (a["start"], a["end"]))


def _annotation(text: str, start: int, end: int, kind: str) -> dict[str, object]:
    return {"text": text[start:end], "start": start, "end": end, "kind": kind}
