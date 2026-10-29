# SPDX-License-Identifier: Apache-2.0
"""GMSH Provenance Chain — document-to-truth lineage tracking.

Every truth object, test, run, and verdict must trace back to source
documents and exact spans.  Provenance is stored as GMS triples in the
DOETracingGMS store, making the lineage queryable via geometric search.

Provenance flow:
    source document → normalized sections → triples → truths → claims → verdicts
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tracing import DOETracingGMS


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SectionSpan:
    """A located section within a markdown document."""

    tag: str                  # normalized section name
    heading: str = ""         # original heading text
    level: int = 1            # heading level (1-3)
    char_start: int = 0
    char_end: int = 0

    @property
    def length(self) -> int:
        return self.char_end - self.char_start


@dataclass
class DocumentRecord:
    """Immutable record of an ingested source document.

    The doc_id is the SHA-256 hash of the raw content, providing
    content-addressable versioning.
    """

    doc_id: str               # SHA-256 of content
    source_path: str
    content_hash: str         # same as doc_id
    ingested_at: str = ""
    version: int = 1
    parent_doc_id: str = ""   # for version chains
    title: str = ""
    sections: list[SectionSpan] = field(default_factory=list)

    def __post_init__(self):
        if not self.ingested_at:
            self.ingested_at = time.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class NormalizedDocument:
    """Markdown with section spans tracked for provenance."""

    doc_id: str
    markdown: str
    sections: list[SectionSpan] = field(default_factory=list)

    def get_section_text(self, tag: str) -> str | None:
        """Retrieve the text of a section by tag."""
        for s in self.sections:
            if s.tag == tag:
                return self.markdown[s.char_start:s.char_end]
        return None

    def section_for_offset(self, offset: int) -> SectionSpan | None:
        """Find which section contains a character offset."""
        for s in self.sections:
            if s.char_start <= offset < s.char_end:
                return s
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)", re.MULTILINE)


def _normalize_tag(heading_text: str) -> str:
    """Normalize a heading into a section tag."""
    tag = re.sub(r"^\d+\.\s*", "", heading_text).strip()
    tag = re.sub(r"[^a-zA-Z0-9_\s]", "", tag).strip()
    return tag.replace(" ", "_").lower()


def _compute_hash(content: str) -> str:
    """SHA-256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _extract_sections(markdown: str) -> list[SectionSpan]:
    """Extract section spans from markdown headings."""
    sections: list[SectionSpan] = []
    matches = list(_HEADING_RE.finditer(markdown))

    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        tag = _normalize_tag(heading)
        char_start = m.start()
        char_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)

        sections.append(SectionSpan(
            tag=tag,
            heading=heading,
            level=level,
            char_start=char_start,
            char_end=char_end,
        ))

    return sections


def _extract_title(markdown: str) -> str:
    """Extract document title from first H1 heading."""
    m = re.search(r"^#\s+(.+)", markdown, re.MULTILINE)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# ProvenanceRegistry
# ---------------------------------------------------------------------------

class ProvenanceRegistry:
    """Tracks document → section → triple → truth lineage.

    Stores provenance as GMS triples in the DOETracingGMS store,
    making the full lineage queryable via geometric search.

    Parameters
    ----------
    tracing : DOETracingGMS or None
        The tracing store for recording provenance triples.
        If None, provenance is tracked in-memory only.
    """

    def __init__(self, tracing: "DOETracingGMS | None" = None):
        self.tracing = tracing
        self._documents: dict[str, DocumentRecord] = {}
        self._normalized: dict[str, NormalizedDocument] = {}
        self._triple_provenance: dict[str, tuple[str, str]] = {}  # triple_key -> (doc_id, section_tag)

    def register_document(
        self,
        doc_path: str,
        markdown: str,
    ) -> DocumentRecord:
        """Register a source document and extract section structure.

        Parameters
        ----------
        doc_path : str
            Path to the source document.
        markdown : str
            Normalized markdown content.

        Returns
        -------
        DocumentRecord
            The registered document with section spans.
        """
        content_hash = _compute_hash(markdown)
        sections = _extract_sections(markdown)
        title = _extract_title(markdown)

        # Check if this exact content was already registered
        if content_hash in self._documents:
            return self._documents[content_hash]

        doc = DocumentRecord(
            doc_id=content_hash,
            source_path=os.path.abspath(doc_path),
            content_hash=content_hash,
            title=title,
            sections=sections,
        )
        self._documents[content_hash] = doc

        norm = NormalizedDocument(
            doc_id=content_hash,
            markdown=markdown,
            sections=sections,
        )
        self._normalized[content_hash] = norm

        # Record provenance as GMS triples
        if self.tracing:
            from knowlytix.knowledge.extract import ExtractedTriple
            triples = [
                ExtractedTriple(
                    head=content_hash[:16],
                    relation="is_type",
                    tail="source_document",
                ),
                ExtractedTriple(
                    head=content_hash[:16],
                    relation="source_path",
                    tail=os.path.basename(doc_path),
                ),
                ExtractedTriple(
                    head=content_hash[:16],
                    relation="title",
                    tail=title[:80] if title else os.path.basename(doc_path),
                ),
            ]
            for sec in sections:
                triples.append(ExtractedTriple(
                    head=content_hash[:16],
                    relation="has_section",
                    tail=sec.tag,
                ))
            self.tracing._learn(triples)

        return doc

    def link_triple_to_source(
        self,
        head: str,
        relation: str,
        tail: str,
        doc_id: str,
        section_tag: str,
    ):
        """Record that a triple was sourced from a specific document section.

        Parameters
        ----------
        head, relation, tail : str
            The knowledge triple.
        doc_id : str
            Source document ID (content hash).
        section_tag : str
            Normalized section tag.
        """
        key = f"{head}|{relation}|{tail}"
        self._triple_provenance[key] = (doc_id, section_tag)

        if self.tracing:
            from knowlytix.knowledge.extract import ExtractedTriple
            triple_id = _compute_hash(key)[:12]
            self.tracing._learn([
                ExtractedTriple(
                    head=triple_id,
                    relation="sourced_from",
                    tail=doc_id[:16],
                ),
                ExtractedTriple(
                    head=triple_id,
                    relation="in_section",
                    tail=section_tag,
                ),
            ])

    def get_provenance(
        self, head: str, relation: str, tail: str,
    ) -> tuple[str, str] | None:
        """Look up the source document and section for a triple.

        Returns
        -------
        (doc_id, section_tag) or None
        """
        key = f"{head}|{relation}|{tail}"
        return self._triple_provenance.get(key)

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        """Retrieve a registered document by ID."""
        return self._documents.get(doc_id)

    def get_normalized(self, doc_id: str) -> NormalizedDocument | None:
        """Retrieve the normalized document by ID."""
        return self._normalized.get(doc_id)

    @property
    def documents(self) -> list[DocumentRecord]:
        """All registered documents."""
        return list(self._documents.values())
