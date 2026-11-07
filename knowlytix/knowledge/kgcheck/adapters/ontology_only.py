# SPDX-License-Identifier: Apache-2.0
"""Read an ontology on its own, without a store, into an ``AuditSnapshot``.

The ontology is an ``ontology.json`` file or the same structure in memory. Its
terms become nodes, and its ``schema`` rows ``(head, relation, tail)`` become
the ``proposal`` view, the class-level statements of an ontology proposal. The
rows stay opaque unless the audit schema sidecar declares how to read them: a
row whose relation is a declared subclass predicate is then a subclass edge.
The other store views are absent, so rules that need extracted facts,
provenance or numeric memory report unavailable. Parsing reuses the store
reader, so malformed rows and repeated keys are diagnosed the same way.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from knowlytix.knowledge.kgcheck.adapters._json import (
    RawArtifact,
    json_artifact_from_text,
    read_json_artifact,
)
from knowlytix.knowledge.kgcheck.adapters.ontodisc_store import (
    SOURCE_TEXT,
    Triple,
    _Builder,
    _is_triple,
)
from knowlytix.knowledge.kgcheck.contract import artifact_specs, contract_version
from knowlytix.knowledge.kgcheck.models import (
    ArtifactRecord,
    AuditSnapshot,
    Capabilities,
    GraphView,
    RawRef,
    SemanticsDecl,
    SourceText,
)

ADAPTER = "ontology"
ADAPTER_VERSION = "2"
ONTOLOGY = "ontology.json"

OntologyInput = str | Path | Mapping[str, object]


def read_ontology(ontology: OntologyInput, audit_schema: OntologyInput | None = None) -> AuditSnapshot:
    """Snapshot of an ontology (a file path or its parsed JSON) and an optional sidecar.

    Raises FileNotFoundError when a path names no file.
    """
    # imported here: audit_schema imports adapters._json, which loads this package
    from knowlytix.knowledge.kgcheck.audit_schema import (
        SIDECAR,
        LoadedSidecar,
        sidecar_from_artifact,
    )
    raw = {s.filename: RawArtifact(ArtifactRecord(s.filename, False, False)) for s in artifact_specs()}
    raw[ONTOLOGY] = _artifact(ontology, ONTOLOGY)
    sidecar = (sidecar_from_artifact(_artifact(audit_schema, SIDECAR)) if audit_schema is not None
               else LoadedSidecar(None, SemanticsDecl()))
    return _OntologyBuilder(raw, RawArtifact(ArtifactRecord(SOURCE_TEXT, False, False)),
                            sidecar).build()


def _artifact(source: OntologyInput, name: str) -> RawArtifact:
    if isinstance(source, Mapping):
        return json_artifact_from_text(name, json.dumps(source, ensure_ascii=False))
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is not a file")
    path = path.resolve()  # a named file is not a store: no containment check applies to it
    art = read_json_artifact(path.parent, path.name, True)
    record = replace(art.record, name=name)
    diags = tuple(replace(d, location=RawRef(name, d.location.pointer)) for d in art.diagnostics)
    return replace(art, record=record, diagnostics=diags)


class _OntologyBuilder(_Builder):
    """The store reader with the schema rows as the only view."""

    def _read(self) -> list[RawArtifact]:
        extra = [self.sidecar.artifact] if self.sidecar.artifact is not None else []
        return [self.raw[ONTOLOGY], *extra]

    def view_rows(self) -> dict[GraphView, list[tuple[Triple, RawRef]]]:
        spec = self.raw[ONTOLOGY].value
        rows = spec.get("schema") if isinstance(spec, dict) else None
        if not isinstance(rows, list):
            return {GraphView.PROPOSAL: []} if isinstance(spec, dict) else {}
        return {GraphView.PROPOSAL: [((r[0], r[1], r[2]), RawRef(ONTOLOGY, f"/schema/{i}"))
                                     for i, r in enumerate(rows) if _is_triple(r)]}

    def snapshot_id(self) -> str:
        digests = sorted((a.record.name, a.record.sha256 or "") for a in self._read())
        payload = json.dumps([ADAPTER, ADAPTER_VERSION, contract_version(), digests])
        return "s:" + hashlib.sha256(payload.encode()).hexdigest()[:20]

    def capabilities(self, sources: list[SourceText], evidence_ok: bool) -> Capabilities:
        return replace(super().capabilities(sources, evidence_ok), adapter=ADAPTER,
                       adapter_version=ADAPTER_VERSION)
