# SPDX-License-Identifier: Apache-2.0
"""Sections and sentence units of retained source text, as OntoDisc divides them.

Evidence records cite a section key and, for ``sentence`` records, an index
into that section's sentence units. ``split_sections`` and ``sentence_units``
reproduce OntoDisc's ``parse_sections`` and its two sentence splitters (grounding,
and monetary attribution) so
these citations can be checked exactly; tests pin both to the producer on the
fixture sources and on edge cases (preamble, ``###`` subsections, repeated
headings). They are reimplemented here, not imported, so an artifact audit does
not depend on producer internals.
"""

from __future__ import annotations

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.;:!?])\s+|\n+")
# OntoDisc's monetary attribution (ontology_score._monetary_evidence) keeps colons
# intact for labeled values and counts every unit, blank ones included.
_MONETARY_SPLIT = re.compile(r"(?<=[.;!?])\s+|\n+")
MONEY_RELATIONS = frozenset({"has_filing_threshold", "has_minimum_aggregate_amount",
                             "has_monetary_amount"})
_COMMENT = re.compile(r"<!--.*?-->", re.S)


def sentence_units(text: str) -> list[str]:
    """Sentence-like units in the order ``sentence_index`` counts them (unstripped)."""
    return [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def monetary_units(text: str) -> list[str]:
    """The units a ``sentence`` record of a monetary relation (``MONEY_RELATIONS``) counts."""
    return _MONETARY_SPLIT.split(text)


def nonblank_lines(text: str) -> list[str]:
    """Non-blank lines, the units a ``line`` record's index counts."""
    return [line for line in text.split("\n") if line.strip()]


class _Splitter:
    def __init__(self, text: str) -> None:
        self.sections: dict[str, str] = {}
        self.h2: str | None = None
        self.h3: str | None = None
        self.title: str | None = None
        self.preamble, self.in_fence = True, False
        self.buf: list[str] = []
        self.h2_seen: set[str] = set()
        self.literal = {ln[3:].strip() for ln in text.split("\n") if ln.startswith("## ")}

    def add(self, key: str, content: str, keep_empty: bool = False) -> None:
        if not content and not keep_empty:
            return
        unique, n = key, 1
        while unique in self.sections:
            n += 1
            unique = f"{key} ({n})"
        self.sections[unique] = content
        if "/" not in key:
            self.h2_seen.add(unique)

    def save(self) -> None:
        content = "\n".join(self.buf).strip()
        if self.preamble:
            self.add(self.title or "Preamble", _COMMENT.sub("", content).strip())
        elif self.h2 is None:
            self.add(self.h3 or "Preamble", content)
        else:
            self.add(f"{self.h2}/{self.h3}" if self.h3 else self.h2, content)

    def heading(self, name: str) -> str:
        unique, n = name, 1
        while unique in self.h2_seen:
            n += 1
            unique = f"{name} ({n})"
            while unique in self.literal:
                n += 1
                unique = f"{name} ({n})"
        self.h2_seen.add(unique)
        return unique

    def feed(self, line: str) -> None:
        if self.preamble and line.lstrip().startswith("```"):
            self.in_fence = not self.in_fence
        if line.startswith("## "):
            self.save()
            self.h2 = self.heading(line[3:].strip())
            self.preamble, self.h3, self.buf = False, None, []
        elif line.startswith("### "):
            self.subsection(line[4:].strip())
        elif self.preamble and self.title is None and not self.in_fence and line.startswith("# "):
            self.title = line[2:].strip() or None
        else:
            self.buf.append(line)

    def subsection(self, name: str) -> None:
        if self.preamble or self.h2 is None or self.h3 is not None:
            self.save()
        else:
            self.add(self.h2, "\n".join(self.buf).strip(), keep_empty=True)
        self.preamble, self.h3, self.buf = False, name, []


def split_sections(text: str) -> dict[str, str]:
    """Section key (``H2`` or ``H2/H3``) to section text, in document order."""
    splitter = _Splitter(text)
    for line in text.split("\n"):
        splitter.feed(line)
    splitter.save()
    return splitter.sections
