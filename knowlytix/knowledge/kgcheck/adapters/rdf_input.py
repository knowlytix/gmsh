# SPDX-License-Identifier: Apache-2.0
"""Read an RDF graph into an ``AuditSnapshot`` (optional: needs ``rdflib``).

Every triple of the file is a row of the ``stored`` view. IRIs keep their full
text as the token, blank nodes get rdflib's canonical labels so the same graph
always yields the same snapshot, and a literal's token is its lexical form,
with its datatype or language kept in the node's literal basis. The canonical
N-Triples text is kept, so CON007 validates the graph as read rather than a
round trip through the snapshot.

Vocabulary statements that are constraints in OWL are imported as
declarations with source ``imported``: ``rdfs:subClassOf`` and ``rdf:type`` as
the subclass and type predicates, ``owl:inverseOf``, the symmetric, asymmetric,
transitive and irreflexive property types, and ``owl:disjointWith``.
``rdfs:domain`` and ``rdfs:range`` are not imported, because in RDF they infer
types instead of restricting them; declare an application domain or range in
the audit schema sidecar instead. There is no ``ontology.json``, so rules that
need one report unavailable.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from knowlytix.knowledge.kgcheck.adapters.ontodisc_store import (
    _annotate,
    _declared_capabilities,
)
from knowlytix.knowledge.kgcheck.identity import assertion_id, node_id
from knowlytix.knowledge.kgcheck.models import (
    ArtifactRecord,
    Assertion,
    AssertionRow,
    AuditSnapshot,
    Capabilities,
    Declaration,
    DeclKey,
    DeclSource,
    Diagnostic,
    GraphView,
    LiteralValue,
    Node,
    NodeKind,
    OntologyDecl,
    RawRef,
    SemanticsDecl,
)

ADAPTER = "rdf"
ADAPTER_VERSION = "1"
GRAPH = "graph.nt"  # the canonical N-Triples text, kept in ``parsed``
IMPORTED_VERSION = "rdf-import"
_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_RDFS = "http://www.w3.org/2000/01/rdf-schema#"
_OWL = "http://www.w3.org/2002/07/owl#"
_PROPERTY_TYPES = {f"{_OWL}SymmetricProperty": DeclKey.SYMMETRIC,
                   f"{_OWL}AsymmetricProperty": DeclKey.ASYMMETRIC,
                   f"{_OWL}TransitiveProperty": DeclKey.TRANSITIVE,
                   f"{_OWL}IrreflexiveProperty": DeclKey.IRREFLEXIVE}


def read_rdf(path: str | Path, *, format: str | None = None,
             audit_schema: str | Path | None = None) -> AuditSnapshot:
    """Snapshot of the RDF file at ``path``; ``format`` as rdflib names it (guessed if None).

    Raises FileNotFoundError when a path names no file and ImportError without rdflib.
    """
    from rdflib import Graph
    from rdflib.compare import to_canonical_graph
    from rdflib.util import guess_format

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{source} is not a file")
    data = source.read_bytes()
    graph = to_canonical_graph(Graph().parse(data=data, format=format or guess_format(str(source))
                                             or "turtle"))
    record = ArtifactRecord(source.name, True, True, len(data), hashlib.sha256(data).hexdigest())
    return _Reader(graph, record, _sidecar(audit_schema)).build()


def _sidecar(audit_schema: str | Path | None):  # type: ignore[no-untyped-def]
    from knowlytix.knowledge.kgcheck.audit_schema import LoadedSidecar, load_sidecar
    if audit_schema is None:
        return LoadedSidecar(None, SemanticsDecl())
    explicit = Path(audit_schema)
    return load_sidecar(explicit.parent, explicit)


@dataclass
class _Reader:
    graph: object
    record: ArtifactRecord
    sidecar: object
    diags: list[Diagnostic] = field(default_factory=list)
    _nodes: dict[str, Node] = field(default_factory=dict)

    def build(self) -> AuditSnapshot:
        # rdflib's own N-Triples row writer; the kgcheck-rdf extra pins rdflib below 8
        from rdflib.plugins.serializers.nt import _nt_row
        rows, assertions, lines = [], {}, []
        for i, (s, p, o) in enumerate(sorted(self.graph)):  # type: ignore[attr-defined]
            h, r, t = self._term(s), self._relation(p), self._term(o)
            aid = assertion_id(h, r, t)
            assertions[aid] = Assertion(aid, node_id("entity", h), node_id("relation", r),
                                        node_id("entity", t))
            rows.append(AssertionRow(aid, RawRef(self.record.name, f"/{i}")))
            lines.append(_nt_row((s, p, o)))
        semantics = self._semantics()
        nodes = _annotate(sorted(self._nodes.values(), key=lambda n: n.node_id),
                          semantics.accepted())
        artifacts = [self.record]
        if self.sidecar.artifact is not None:  # type: ignore[attr-defined]
            artifacts.append(self.sidecar.artifact.record)  # type: ignore[attr-defined]
        return AuditSnapshot(
            snapshot_id=self._snapshot_id(), nodes=tuple(nodes), assertions=tuple(assertions.values()),
            views={GraphView.STORED: tuple(rows)}, ontology=OntologyDecl(), semantics=semantics,
            capabilities=Capabilities(**_declared_capabilities(semantics.accepted()),
                                      adapter=ADAPTER, adapter_version=ADAPTER_VERSION),
            artifacts=tuple(artifacts),
            diagnostics=(*self.sidecar.diagnostics, *dict.fromkeys(self.diags)),  # type: ignore[attr-defined]
            parsed={GRAPH: "".join(lines)})

    def _term(self, term: object) -> str:
        from rdflib import BNode, Literal
        if isinstance(term, Literal):
            token = str(term)
            basis = f"rdf:{term.datatype}" if term.datatype else (
                f"rdf:@{term.language}" if term.language else "rdf:plain")
            self._add(Node(node_id("entity", token), token, "entity", NodeKind.LITERAL,
                           declared_by=(self.record.name,), literal=LiteralValue(token, basis)))
            return token
        token = f"_:{term}" if isinstance(term, BNode) else str(term)
        self._add(Node(node_id("entity", token), token, "entity", declared_by=(self.record.name,)))
        return token

    def _relation(self, term: object) -> str:
        token = str(term)
        self._add(Node(node_id("relation", token), token, "relation", NodeKind.PROPERTY,
                       declared_by=(self.record.name,)))
        return token

    def _add(self, node: Node) -> None:
        known = self._nodes.get(node.node_id)
        if known is None:
            self._nodes[node.node_id] = node
        elif (known.kind is NodeKind.LITERAL) != (node.kind is NodeKind.LITERAL) or (
                known.literal and node.literal and known.literal.basis != node.literal.basis):
            self.diags.append(Diagnostic(
                "rdf_token_collision", f"{node.label!r} names terms of different kinds or "
                "datatypes; they share one node", RawRef(self.record.name)))
            if node.kind is not NodeKind.LITERAL:
                self._nodes[node.node_id] = node  # an IRI outranks a literal of the same text

    def _semantics(self) -> SemanticsDecl:
        from knowlytix.knowledge.kgcheck.audit_schema import contradictions
        imported = tuple(self._imported())
        declared = self.sidecar.semantics  # type: ignore[attr-defined]
        combined = (*declared.declarations, *imported)
        problems = contradictions(combined)
        if problems:
            self.diags.extend(problems)
            return SemanticsDecl()
        version = declared.version or IMPORTED_VERSION
        return SemanticsDecl(version, combined, declared.competency)

    def _imported(self) -> Iterator[Declaration]:
        ref = RawRef(self.record.name)

        def decl(subject: str, key: DeclKey, value: str = "true") -> Declaration:
            return Declaration(subject, key, value, DeclSource.IMPORTED, ref)

        relations = {str(p) for _, p, _ in self.graph}  # type: ignore[attr-defined]
        if f"{_RDFS}subClassOf" in relations:
            yield decl(node_id("relation", f"{_RDFS}subClassOf"), DeclKey.SUBCLASS_PREDICATE)
        if f"{_RDF}type" in relations:
            yield decl(node_id("relation", f"{_RDF}type"), DeclKey.TYPE_PREDICATE)
        for s, p, o in sorted(self.graph):  # type: ignore[attr-defined]
            p, s_, o_ = str(p), str(s), str(o)
            if p == f"{_OWL}inverseOf":
                yield decl(node_id("relation", s_), DeclKey.INVERSE_OF, node_id("relation", o_))
            elif p == f"{_RDF}type" and o_ in _PROPERTY_TYPES:
                yield decl(node_id("relation", s_), _PROPERTY_TYPES[o_])
            elif p == f"{_OWL}disjointWith":
                pair = " ".join(sorted((s_, o_)))  # A disjointWith B states B disjointWith A
                group = f"disjoint:rdf:{hashlib.sha256(pair.encode()).hexdigest()[:12]}"
                yield decl(group, DeclKey.DISJOINT, node_id("entity", s_))
                yield decl(group, DeclKey.DISJOINT, node_id("entity", o_))

    def _snapshot_id(self) -> str:
        digests = [(self.record.name, self.record.sha256)]
        if self.sidecar.artifact is not None:  # type: ignore[attr-defined]
            digests.append(("audit_schema.json", self.sidecar.artifact.record.sha256))  # type: ignore[attr-defined]
        payload = json.dumps([ADAPTER, ADAPTER_VERSION, sorted(digests)])
        return "s:" + hashlib.sha256(payload.encode()).hexdigest()[:20]
