# SPDX-License-Identifier: Apache-2.0
"""Span<->triple provenance for GEODE.

The extractor stores triples as canonicalized ``(h, r, t)`` tuples and discards
where each came from. GEODE's loop needs the reverse: when the geometric critic
flags a triple, point the actor back at the exact source span to re-read.

:class:`ProvenanceLedger` reconstructs that map by aligning each triple to the
markdown source — table cells (precise char span), schema bullets, and section
headers.

The alignment keys on the **full ``(head, relation, tail)`` triple**, not just
``(head, relation)``. When a row label recurs across tables (financial scenario
tables reusing "Cumulative"/"Total"), the bare ``(entity, relation)`` key
collides and a single-cell index silently overwrites; carrying the tail lets
:meth:`resolve` disambiguate by the cell's value, so each triple lands on its
own cell. :meth:`coverage` verifies that every triple in a graph maps to a span
(or is reported ``unaligned``), and :meth:`build`/:meth:`to_records` produce a
persistable map so the alignment is captured once at build time and read back at
query time via :meth:`from_store` rather than re-derived per query.

Scope: this aligns *structured* triples (tables/bullets/headers). Prose triples
from LLM extraction have no structural anchor and resolve ``unaligned``; they
need spans captured at extraction time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Provenance", "ProvenanceLedger", "is_consistent", "canon",
           "value_canon"]


def canon(s: str) -> str:
    """Match ``DocumentGraph._canon_entity``: collapse whitespace, lowercase."""
    return " ".join(str(s).split()).lower()


def value_canon(s: str) -> str:
    """Separator-insensitive canonical form for *comparing values* (not keys).

    Like :func:`canon` but also unifies space / underscore / hyphen, so a value
    an LLM emits as prose (``"dispute unit"``) compares equal to the graph's
    stored slug (``"dispute_unit"``). Use this ONLY to compare a generated value
    against a stored one (answer verification, eval scoring). For entity-key
    canonicalization use :func:`canon`, which must stay in lockstep with the
    graph's stored keys (``DocumentGraph._canon_entity``)."""
    return canon(re.sub(r"[_\-]+", " ", str(s)))


def _col_slug(header_cell: str) -> str:
    return header_cell.strip().replace(" ", "_").lower()


def _section_slug(header_text: str) -> str:
    tag = re.sub(r"^\d+\.\s*", "", header_text).strip()
    tag = tag.replace(" ", "_").lower()
    return re.sub(r"[^a-z0-9_]", "", tag)


_SCHEMA_RE = re.compile(
    r"^\s*[-*]\s*`?(?P<lhs>[^`\n]+?)`?\s+"
    r"(?P<rel>opposite_of|is_functional)\s+`?(?P<rhs>[^`\n]+?)`?\s*$"
)
_NUM_RE = re.compile(r"-?[\d,]*\.?\d+")


@dataclass
class Provenance:
    """Where an extracted triple came from in the source document.

    Attributes:
        head: canonicalized triple head.
        relation: triple relation.
        tail: canonicalized triple tail value.
        source_path: path of the source document.
        line_no: 1-based line number of the span, -1 when unaligned.
        char_start: absolute file offset where the span starts, -1 when unaligned.
        char_end: absolute file offset where the span ends.
        raw: exact source substring (cell or line) the triple aligned to.
        method: alignment kind: table_cell, schema_bullet, section_header, prose_sentence or unaligned.
    """

    head: str
    relation: str
    tail: str
    source_path: str
    line_no: int           # 1-based; -1 when unaligned
    char_start: int        # absolute file offset; -1 when unaligned
    char_end: int
    raw: str               # exact source substring (cell / line)
    method: str            # table_cell | schema_bullet | section_header | unaligned

    def location(self) -> str:
        """Return the source location as ``path:line:start-end``."""
        return f"{self.source_path}:{self.line_no}:{self.char_start}-{self.char_end}"


def _line_index(text: str) -> list[tuple[int, int, str]]:
    out, off = [], 0
    for i, line in enumerate(text.split("\n"), start=1):
        out.append((i, off, line))
        off += len(line) + 1
    return out


# A sentence ends at .!? that is followed by whitespace or end-of-text -- NOT at a
# decimal point inside a number ("$35.00") or a mid-token dot ("12 U.S.C.").
_SENT_BOUNDARY = re.compile(r"[.!?]+(?=\s|$)")
_STOP = frozenset(("the", "and", "for", "are", "was", "were", "has", "have",
                   "with", "that", "this", "its", "per", "over", "than", "when",
                   "each", "into", "must", "may", "any", "all", "not"))


def _prose_sentences(text: str) -> list[tuple[int, int, str]]:
    """Split the document's PROSE into sentences with absolute char offsets.

    Table rows (lines starting with ``|``) and markdown headers (``#``) are
    excluded, so only running policy prose is indexed. Consecutive prose lines are
    merged into a block (wrapped paragraphs), then split on sentence-ending
    punctuation. Returns ``(char_start, char_end, sentence_text)`` with the text
    whitespace-normalized for readability; the offsets index the original text.
    """
    blocks: list[tuple[int, int]] = []
    off = 0
    for line in text.split("\n"):
        s = line.strip()
        is_prose = bool(s) and not s.startswith("|") and not s.startswith("#")
        end = off + len(line)
        if is_prose:
            if blocks and off == blocks[-1][1] + 1:      # contiguous prose line
                blocks[-1] = (blocks[-1][0], end)
            else:
                blocks.append((off, end))
        off = end + 1
    out: list[tuple[int, int, str]] = []
    for bs, be in blocks:
        seg = text[bs:be]
        start = 0
        bounds = [m.end() for m in _SENT_BOUNDARY.finditer(seg)] + [len(seg)]
        for end in bounds:
            if end <= start:
                continue
            raw = seg[start:end]
            lead = len(raw) - len(raw.lstrip())
            norm = " ".join(raw.split())
            if len(norm) >= 15:
                out.append((bs + start + lead, bs + end, norm))
            start = end
    return out


def _value_in_text(tail: str, text: str) -> bool:
    """Does ``text`` assert the triple's tail value?

    A NUMERIC tail matches only an exact numeric TOKEN in the text (``35.0`` ->
    ``$35.00``), never a substring -- so ``0.0`` does not spuriously match inside
    ``500.00``. A categorical tail matches a value-canonical substring (so
    ``forbidden`` matches the sentence stating it)."""
    try:
        tv = float(str(tail).replace(",", ""))
    except ValueError:
        vt = value_canon(tail)
        return bool(vt) and vt in value_canon(text)
    nums = _NUM_RE.findall(text.replace(",", ""))
    out = []
    for n in nums:
        try:
            out.append(float(n))
        except ValueError:
            pass
    return any(abs(v - tv) < 1e-6 for v in out)


def _cells_with_spans(line: str, line_start: int) -> list[tuple[str, int, int]]:
    segs, idx = [], 0
    for part in line.split("|"):
        seg_start, seg_end = idx, idx + len(part)
        idx = seg_end + 1
        stripped = part.strip()
        if stripped == "" and (seg_start == 0 or seg_end >= len(line)):
            continue
        lead = len(part) - len(part.lstrip())
        c_start = line_start + seg_start + lead
        segs.append((stripped, c_start, c_start + len(stripped)))
    return segs


def _is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-+:?", c) for c in cells if c != "")


def _cell_matches_tail(raw: str, tail: str) -> bool:
    """Does a source cell's text carry the triple's tail value?

    String-equal under value canonicalization, or numerically equal within
    ``1e-6`` (the same test :func:`is_consistent` applies to a span)."""
    if value_canon(raw) == value_canon(tail):
        return True
    nums = _NUM_RE.findall(raw.replace(",", ""))
    try:
        return bool(nums) and abs(float(nums[0]) - float(tail)) < 1e-6
    except ValueError:
        return False


def _disambiguate(candidates, tail):
    """Pick the candidate cell whose value matches ``tail``.

    With one candidate there is nothing to separate, so it is returned as-is
    (the common, no-collision case). With several — a recurring row label — the
    tail value selects the right cell; if none matches, the triple is reported
    ``unaligned`` rather than bound to an arbitrary wrong cell."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    matches = [c for c in candidates if _cell_matches_tail(c[3], tail)]
    return matches[0] if matches else None


def _prov_from_record(r: dict) -> "Provenance":
    return Provenance(
        canon(r["head"]), r["relation"], canon(r["tail"]),
        r.get("source_path", ""), int(r["line_no"]), int(r["char_start"]),
        int(r["char_end"]), r.get("raw", ""), r["method"])


class ProvenanceLedger:
    """Maps each ``(h, r, t)`` triple to a source-span :class:`Provenance`."""

    def __init__(self, md_path: str, *, prefer_prose: bool = False):
        with open(md_path, encoding="utf-8") as f:
            self._init_from_text(f.read(), md_path, prefer_prose=prefer_prose)

    @classmethod
    def from_text(cls, text: str, source_path: str = "", *,
                  prefer_prose: bool = False) -> "ProvenanceLedger":
        """Build a ledger from in-memory markdown (no file read).

        Lets the RAG pipeline align triples against ``store.markdown`` directly
        rather than depending on the on-disk ``documents/combined.md`` copy.

        ``prefer_prose`` makes :meth:`resolve` return the SOURCE SENTENCE that
        asserts a fact when one exists, in preference to the table cell it was
        extracted from — so a fact's provenance is the readable policy statement,
        not a bare value. Off by default, so a table-only corpus is unaffected.
        """
        obj = cls.__new__(cls)
        obj._init_from_text(text, source_path, prefer_prose=prefer_prose)
        return obj

    def _init_from_text(self, text: str, source_path: str, *,
                        prefer_prose: bool = False) -> None:
        self.path = source_path
        self.text = text
        self.prefer_prose = prefer_prose
        self._lines = _line_index(self.text)
        # Tables and schema bullets index to a *list* of candidate spans per
        # (entity, relation) key — a recurring row label ("Total") keeps every
        # cell it names rather than overwriting, so resolve() can pick the one
        # whose value matches the queried tail.
        self._table: dict[tuple[str, str], list[tuple[int, int, int, str]]] = {}
        self._schema: dict[tuple[str, str], list[tuple[int, int, int, str]]] = {}
        self._section: dict[str, tuple[int, int, int, str]] = {}
        # Prose sentences with char offsets, for prefer_prose resolution.
        self._sentences: list[tuple[int, int, str]] = (
            _prose_sentences(text) if prefer_prose else [])
        # Optional persisted (h, r, t) -> Provenance map (see from_store /
        # from_records); consulted before the text index when present.
        self._resolved: dict[tuple[str, str, str], Provenance] | None = None
        self._index()

    def _line_of(self, char_pos: int) -> int:
        """1-based line number containing an absolute char offset."""
        ln = 1
        for i, off, _s in self._lines:
            if off <= char_pos:
                ln = i
            else:
                break
        return ln

    def _alias_terms(self, head_canon: str) -> list[str]:
        """Alias texts declared for a head via ``has_alias`` rows in the doc."""
        recs = self._table.get((head_canon, "has_alias"), [])
        return [r[3] for r in recs]

    def _resolve_prose(self, head: str, relation: str, tail: str):
        """The source SENTENCE that asserts ``(head, relation, tail)``, or None.

        A sentence qualifies if it carries the tail value; among those, it is
        scored by how much of the fact's identity it names — the head's own words,
        any declared alias of the head (weighted), and the relation's words (the
        attribute, e.g. ``unencrypted_channel_pii`` -> "unencrypted", "channel").
        The best-scoring sentence with a non-zero identity match wins, so a fact
        binds to the policy statement about it, not to an unrelated sentence that
        merely repeats the value."""
        h = canon(head)
        head_terms = [w for w in h.replace("_", " ").split()
                      if len(w) > 2 and w not in _STOP]
        rel_terms = [w for w in relation.replace("has_", "", 1).replace("_", " ").split()
                     if len(w) > 2 and w not in _STOP]
        aliases = [canon(a) for a in self._alias_terms(h)]
        best, best_score = None, 0
        for a, b, sent in self._sentences:
            if not _value_in_text(tail, sent):
                continue
            sc = canon(sent)
            # A sentence naming the head entity directly outranks one that only
            # mentions a customer alias -- an alias (e.g. "nsf" for overdraft) can
            # appear in a DIFFERENT policy's sentence and must not outweigh the
            # head's own statement. Head name x3, relation words x1, alias x1.
            score = (3 * sum(t in sc for t in head_terms)
                     + sum(t in sc for t in rel_terms)
                     + sum(al in sc for al in aliases))
            if score > best_score:
                best, best_score = (a, b, sent), score
        if best is None:
            return None
        a, b, sent = best
        return Provenance(h, relation, canon(tail), self.path, self._line_of(a),
                          a, b, sent, "prose_sentence")

    def _index(self) -> None:
        block: list[tuple[int, int, str]] = []

        def flush(blk):
            rows = [(ln, st, _cells_with_spans(s, st)) for ln, st, s in blk]
            rows = [r for r in rows if not _is_separator([c[0] for c in r[2]])]
            if len(rows) < 2:
                return
            header = [c[0] for c in rows[0][2]]
            for ln, _st, cells in rows[1:]:
                if not cells:
                    continue
                row_entity = canon(cells[0][0])
                for ci in range(1, min(len(cells), len(header))):
                    rel = "has_" + _col_slug(header[ci])
                    txt, cs, ce = cells[ci]
                    self._table.setdefault((row_entity, rel), []).append(
                        (ln, cs, ce, txt))

        for ln, st, s in self._lines:
            if s.strip().startswith("|"):
                block.append((ln, st, s))
            elif block:
                flush(block)
                block = []
        if block:
            flush(block)

        for ln, st, s in self._lines:
            m = _SCHEMA_RE.match(s)
            if m:
                lhs, rel = canon(m.group("lhs")), m.group("rel")
                lead = len(s) - len(s.lstrip())
                self._schema.setdefault((lhs, rel), []).append(
                    (ln, st + lead, st + len(s.rstrip()), s.strip()))
            h = s.lstrip()
            if h.startswith("#"):
                name = h.lstrip("#").strip()
                self._section[_section_slug(name)] = (ln, st, st + len(s.rstrip()),
                                                       s.strip())

    def resolve(self, head: str, relation: str, tail: str) -> Provenance:
        """Locate the source span for a ``(head, relation, tail)`` triple.

        Consults the persisted map first when present, then optional prose
        sentences, then the table/schema/section indices, disambiguating
        recurring keys by the tail value.

        Args:
            head: triple head entity.
            relation: triple relation.
            tail: triple tail value, used to disambiguate recurring keys.

        Returns:
            A :class:`Provenance`; its ``method`` is ``"unaligned"`` when no span
            matches.
        """
        h, t = canon(head), canon(tail)
        # Persisted map (captured at build time) takes precedence over the
        # text index, so query-time resolution is a direct lookup.
        if self._resolved is not None:
            hit = self._resolved.get((h, relation, t))
            if hit is not None:
                return hit
        # Prefer the source SENTENCE that states the fact (readable provenance)
        # over the table cell it was extracted from, when prefer_prose is set.
        if self.prefer_prose:
            p = self._resolve_prose(head, relation, tail)
            if p is not None:
                return p
        key = (h, relation)
        if relation in ("opposite_of", "is_functional"):
            rec = _disambiguate(self._schema.get(key, []), t)
            if rec is not None:
                ln, cs, ce, raw = rec
                return Provenance(h, relation, t, self.path, ln, cs, ce, raw,
                                  "schema_bullet")
        elif relation == "in_section" and t in self._section:
            ln, cs, ce, raw = self._section[t]
            return Provenance(h, relation, t, self.path, ln, cs, ce, raw,
                              "section_header")
        else:
            rec = _disambiguate(self._table.get(key, []), t)
            if rec is not None:
                ln, cs, ce, raw = rec
                return Provenance(h, relation, t, self.path, ln, cs, ce, raw,
                                  "table_cell")
        return Provenance(h, relation, t, self.path, -1, -1, -1, "", "unaligned")

    def build(self, triples) -> dict[tuple[str, str, str], Provenance]:
        """Resolve every triple and return a ``(h, r, t) -> Provenance`` map."""
        return {(h, r, t): self.resolve(h, r, t) for h, r, t in triples}

    def recurring_keys(self) -> dict[tuple[str, str], list[str]]:
        """``(entity, relation)`` keys that name more than one source cell.

        These are exactly the recurring row labels a ``(entity, relation)``-only
        index would collide on; :meth:`resolve` separates them by the queried
        tail value. Returns the key -> list of candidate cell texts.
        """
        return {k: [r[3] for r in recs]
                for k, recs in self._table.items() if len(recs) > 1}

    def coverage(self, triples) -> dict:
        """Resolve every triple and report how many map to a source span.

        The build-time guarantee that *every* structured triple is anchored:
        ``unaligned`` triples are listed explicitly (prose / derived facts with
        no structural cell), never silently dropped.
        """
        provs, unaligned = {}, []
        for h, r, t in triples:
            p = self.resolve(h, r, t)
            provs[(canon(h), r, canon(t))] = p
            if p.method == "unaligned":
                unaligned.append((canon(h), r, canon(t)))
        aligned = len(provs) - len(unaligned)
        return {
            "provenances": provs,
            "aligned": aligned,
            "unaligned": len(unaligned),
            "unaligned_triples": unaligned,
            "ratio": aligned / max(1, len(provs)),
        }

    # --- Persistence: capture the (h, r, t) -> span map once, read it back ---

    def to_records(self, triples) -> list[dict]:
        """Serializable provenance records for ``triples`` (one per triple)."""
        return [
            {"head": p.head, "relation": p.relation, "tail": p.tail,
             "source_path": p.source_path, "line_no": p.line_no,
             "char_start": p.char_start, "char_end": p.char_end,
             "raw": p.raw, "method": p.method}
            for p in self.build(triples).values()
        ]

    @classmethod
    def from_records(cls, records, source_path: str = "<report>"
                     ) -> "ProvenanceLedger":
        """Build a ledger backed by a persisted ``(h, r, t)`` map.

        ``resolve`` becomes a direct dict lookup with no text re-parse — the
        read-back side of :meth:`to_records`.
        """
        obj = cls.__new__(cls)
        obj.path = source_path
        obj.text = ""
        obj.prefer_prose = False
        obj._lines = []
        obj._table = {}
        obj._schema = {}
        obj._section = {}
        obj._sentences = []
        obj._resolved = {(canon(r["head"]), r["relation"], canon(r["tail"])):
                         _prov_from_record(r) for r in records}
        return obj

    @classmethod
    def from_store(cls, store, *, prefer_prose: bool = False) -> "ProvenanceLedger":
        """Ledger for a built store: read the persisted ``provenance.json`` if
        present (direct lookup), with the store's markdown as the fallback
        index for triples not in the persisted map (e.g. hand-built queries).
        """
        import json
        import os

        text = getattr(store, "markdown", "") or ""
        led = cls.from_text(text, "<report>", prefer_prose=prefer_prose)
        path = os.path.join(store.store_path, "provenance.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                records = json.load(f)
            led._resolved = {(canon(r["head"]), r["relation"], canon(r["tail"])):
                             _prov_from_record(r) for r in records}
        return led

    def line(self, line_no: int) -> str:
        """Return the source text of a 1-based line number, or empty when out of range."""
        return self._lines[line_no - 1][2] if 1 <= line_no <= len(self._lines) else ""

    def table_block(self, line_no: int) -> str:
        """Full contiguous markdown table containing ``line_no`` (a bounded
        span to hand the actor for localized re-reasoning)."""
        n = len(self._lines)
        if not (1 <= line_no <= n):
            return ""
        lo = hi = line_no
        while lo > 1 and self._lines[lo - 2][2].strip().startswith("|"):
            lo -= 1
        while hi < n and self._lines[hi][2].strip().startswith("|"):
            hi += 1
        return "\n".join(self._lines[i - 1][2] for i in range(lo, hi + 1))


def is_consistent(p: Provenance) -> bool:
    """Does the resolved source span actually support the triple?"""
    if p.method == "unaligned":
        return False
    if p.method == "section_header":
        return _section_slug(p.raw.lstrip("#").strip()) == p.tail
    if p.method in ("schema_bullet", "prose_sentence"):
        return _value_in_text(p.tail, p.raw)
    if canon(p.raw) == p.tail:
        return True
    nums = _NUM_RE.findall(p.raw.replace(",", ""))
    try:
        return bool(nums) and abs(float(nums[0]) - float(p.tail)) < 1e-6
    except ValueError:
        return False
