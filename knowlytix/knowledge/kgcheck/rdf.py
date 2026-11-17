# SPDX-License-Identifier: Apache-2.0
"""RDF for kgcheck: a snapshot view as an RDF graph, and the audit schema as SHACL shapes.

Optional: everything here needs ``rdflib`` (install the ``kgcheck-rdf`` extra);
importing this module without it raises ImportError, which callers report as
an unavailable backend.

A view maps to RDF term by term (``Terms``). A snapshot read from RDF keeps its
own terms: a token is the original IRI or blank node, so translated shapes
target the graph as read. Any other snapshot's nodes become the IRIs
``urn:kgcheck:<namespace>:<token>`` (the token percent-encoded). The object of a
relation with a value contract is a literal of the contract's datatype (as
CON002 reads it: declared, else decimal when bounded, else string), unless it is
a declared entity, which CON002 also rejects; other literal nodes are plain
literals. The lexical form is kept even when it is not a valid value of the
type. Declared and asserted types become ``rdf:type`` and declared subclass
edges ``rdfs:subClassOf``.

``shapes_from_audit_schema`` translates the constraints CON001-CON006 check
natively into SHACL Core shapes, plus SPARQL constraints for irreflexivity,
materialization and numeric allowed values, which compare by value as CON002
does. SHACL is closed-world: a ``sh:class`` constraint fails on an untyped node,
as CON001 does only under ``closed_world.typing``. ``sh:maxCount`` counts
distinct RDF terms, CON003's lexical equality; a maximum under ``equality:
value`` is not translated and ``untranslated`` lists it. One known difference
remains: rdflib accepts an exponent in an ``xsd:decimal`` (``"1e3"``), which the
XSD lexical space and CON002 do not.
"""

from __future__ import annotations

from collections.abc import Iterator
from urllib.parse import quote, unquote

from rdflib import RDF, RDFS, XSD, BNode, Graph, Literal, Namespace, URIRef
from rdflib.collection import Collection

from knowlytix.knowledge.kgcheck.datatypes import effective_datatype, value_relations
from knowlytix.knowledge.kgcheck.models import (
    AuditSnapshot,
    DeclKey,
    GraphView,
    SemanticsDecl,
)
from knowlytix.knowledge.kgcheck.rules._common import is_literal, label
from knowlytix.knowledge.kgcheck.typed import typing

BASE = "urn:kgcheck:"
SH = Namespace("http://www.w3.org/ns/shacl#")
SHAPE = Namespace(BASE + "shape:")
XSD_TYPES = {"decimal": XSD.decimal, "integer": XSD.integer, "date": XSD.date,
             "boolean": XSD.boolean, "string": XSD.string}
NUMERIC = frozenset({"decimal", "integer"})


def iri(node_id: str) -> URIRef:
    """The kgcheck IRI of a node id (``entity:<token>`` or ``relation:<token>``)."""
    namespace, _, token = node_id.partition(":")
    return URIRef(f"{BASE}{namespace}:{quote(token, safe='')}")


def node_of(term: object) -> str | None:
    """The node id a kgcheck IRI names, or None for any other term."""
    text = str(term)
    if not isinstance(term, URIRef) or not text.startswith(BASE):
        return None
    namespace, _, token = text[len(BASE):].partition(":")
    return f"{namespace}:{unquote(token)}" if namespace in ("entity", "relation") else None


class Terms:
    """Node ids to RDF terms and back, for one snapshot."""

    def __init__(self, snapshot: AuditSnapshot) -> None:
        self.snapshot = snapshot
        self.native = snapshot.capabilities.adapter == "rdf"  # tokens are the graph's own terms

    def term(self, node_id: str):  # type: ignore[no-untyped-def]
        if not self.native:
            return iri(node_id)
        token = node_id.partition(":")[2]
        return BNode(token[2:]) if token.startswith("_:") else URIRef(token)

    def node(self, term: object) -> str | None:
        if not self.native:
            return node_of(term)
        if isinstance(term, BNode):
            return f"entity:_:{term}"
        if not isinstance(term, URIRef):
            return None
        entity = f"entity:{term}"
        return entity if entity in self.snapshot.node_index else f"relation:{term}"


def view_graph(snapshot: AuditSnapshot, view: GraphView) -> Graph:
    """The view as RDF: its assertions, the nodes' types and the subclass edges."""
    from knowlytix.knowledge.kgcheck.rules.constraints import is_declared_entity
    sem, terms = snapshot.semantics.accepted(), Terms(snapshot)
    contracted = {r: effective_datatype(sem, r) for r in value_relations(sem)}
    graph = Graph()
    for row in snapshot.view(view):
        a = snapshot.assertion_index[row.assertion_id]
        datatype = contracted.get(a.predicate)
        if datatype is not None and not is_declared_entity(snapshot, a.object):
            obj = _literal(label(snapshot, a.object), datatype)
        elif is_literal(snapshot, a.object):
            obj = Literal(label(snapshot, a.object))
        else:
            obj = terms.term(a.object)
        graph.add((terms.term(a.head), terms.term(a.predicate), obj))
    types = typing(snapshot, view)
    for node, classes in types.direct.items():
        for cls in classes:
            graph.add((terms.term(node), RDF.type, terms.term(cls)))
    for child, parents in types.parents.items():
        for parent in parents:
            graph.add((terms.term(child), RDFS.subClassOf, terms.term(parent)))
    return graph


def _literal(lexical: str, datatype: str) -> Literal:
    return Literal(lexical) if datatype == "string" else Literal(lexical,
                                                                 datatype=XSD_TYPES[datatype])


def untranslated(snapshot: AuditSnapshot) -> list[str]:
    """Declared constraints ``shapes_from_audit_schema`` does not express, as relation ids."""
    sem = snapshot.semantics.accepted()
    return [f"cardinality.max with equality value: {r}" for r in sem.subjects(DeclKey.MAX_COUNT)
            if (e := sem.find(DeclKey.EQUALITY, r)) is not None and e.value == "value"]


def shapes_from_audit_schema(snapshot: AuditSnapshot) -> Graph:
    """SHACL shapes for the accepted constraint declarations of the audit schema."""
    sem, terms = snapshot.semantics.accepted(), Terms(snapshot)
    graph = Graph()
    graph.bind("sh", SH)
    for triple in (*_types(sem, terms), *_values(sem, terms), *_cardinality(sem, terms),
                   *_disjoint(sem, terms), *_sparql(sem, terms)):
        graph.add(triple)
    return graph


def _shape(kind: str, relation: str) -> URIRef:
    return SHAPE[f"{kind}:{quote(relation, safe='')}"]


def _target(shape: URIRef, target: URIRef, relation: str, terms: Terms) -> list[tuple]:
    return [(shape, RDF.type, SH.NodeShape), (shape, target, terms.term(relation))]


def _class_constraint(shape: URIRef, classes: tuple[str, ...], terms: Terms) -> list[tuple]:
    if len(classes) == 1:
        return [(shape, SH["class"], terms.term(classes[0]))]
    scratch, head, members, out = Graph(), BNode(), [], []
    for cls in classes:
        member = BNode()
        members.append(member)
        out.append((member, SH["class"], terms.term(cls)))
    Collection(scratch, head, members)
    return [*out, *scratch, (shape, SH["or"], head)]


def _types(sem: SemanticsDecl, terms: Terms) -> Iterator[tuple]:
    for key, target, kind in ((DeclKey.DOMAIN, SH.targetSubjectsOf, "domain"),
                              (DeclKey.RANGE, SH.targetObjectsOf, "range")):
        for relation in sem.subjects(key):
            shape = _shape(kind, relation)
            yield from _target(shape, target, relation, terms)
            yield from _class_constraint(shape, sem.values(key, relation), terms)


def _values(sem: SemanticsDecl, terms: Terms) -> Iterator[tuple]:
    for relation in value_relations(sem):
        datatype = effective_datatype(sem, relation)
        shape, prop = _shape("value", relation), BNode()
        yield from _target(shape, SH.targetSubjectsOf, relation, terms)
        yield from ((shape, SH.property, prop), (prop, SH.path, terms.term(relation)),
                    (prop, SH.datatype, XSD_TYPES[datatype]))
        for key, term in ((DeclKey.MINIMUM, SH.minInclusive), (DeclKey.MAXIMUM, SH.maxInclusive)):
            bound = sem.find(key, relation)
            if bound is not None:
                yield prop, term, Literal(bound.value, datatype=XSD.decimal)
        allowed = sem.values(DeclKey.ALLOWED_VALUE, relation)
        if allowed and datatype in NUMERIC:
            yield from _sparql_shape("valuein", relation, _allowed_query(
                terms.term(relation), allowed, datatype), terms)
        elif allowed:
            scratch, head = Graph(), BNode()
            Collection(scratch, head, [_literal(v, datatype) for v in allowed])
            yield from scratch
            yield prop, SH["in"], head


def _allowed_query(relation: URIRef, allowed: tuple[str, ...], datatype: str) -> str:
    """Numeric allowed values compared by value (5.0 = 5), as CON002 compares them.

    ``=`` compares numbers by value; rdflib's ``IN`` compares RDF terms.
    """
    equal = " || ".join(f"?value = {_literal(v, datatype).n3()}" for v in allowed)
    return (f"SELECT $this ?value WHERE {{ $this {relation.n3()} ?value . "
            f"FILTER(isLiteral(?value) && !({equal})) }}")


def _cardinality(sem: SemanticsDecl, terms: Terms) -> Iterator[tuple]:
    for relation in sem.subjects(DeclKey.MAX_COUNT):
        equality = sem.find(DeclKey.EQUALITY, relation)
        if equality is not None and equality.value == "value":
            continue  # sh:maxCount counts RDF terms; see ``untranslated``
        shape, prop = _shape("max", relation), BNode()
        yield from _target(shape, SH.targetSubjectsOf, relation, terms)
        yield from ((shape, SH.property, prop), (prop, SH.path, terms.term(relation)),
                    (prop, SH.maxCount, Literal(int(sem.find(DeclKey.MAX_COUNT, relation).value))))  # type: ignore[union-attr]
    for relation in sem.subjects(DeclKey.MIN_COUNT):
        if not sem.flag(DeclKey.COMPLETE, relation):
            continue  # an incomplete relation cannot violate a minimum (CON003)
        low = Literal(int(sem.find(DeclKey.MIN_COUNT, relation).value))  # type: ignore[union-attr]
        for i, cls in enumerate(sem.values(DeclKey.DOMAIN, relation) or ()):
            shape, prop = _shape(f"min{i}", relation), BNode()
            yield from ((shape, RDF.type, SH.NodeShape), (shape, SH.targetClass, terms.term(cls)),
                        (shape, SH.property, prop), (prop, SH.path, terms.term(relation)),
                        (prop, SH.minCount, low))
        shape, prop = _shape("min", relation), BNode()
        yield from _target(shape, SH.targetSubjectsOf, relation, terms)
        yield from ((shape, SH.property, prop), (prop, SH.path, terms.term(relation)),
                    (prop, SH.minCount, low))


def _disjoint(sem: SemanticsDecl, terms: Terms) -> Iterator[tuple]:
    groups: dict[str, list[str]] = {}
    for d in sem.of(DeclKey.DISJOINT):
        groups.setdefault(d.subject, []).append(d.value)
    for group, members in sorted(groups.items()):
        for a in members:
            for b in members:
                if a >= b:
                    continue
                shape, negated = SHAPE[f"disjoint:{quote(group, safe='')}:{quote(a, safe='')}:"
                                       f"{quote(b, safe='')}"], BNode()
                yield from ((shape, RDF.type, SH.NodeShape), (shape, SH.targetClass, terms.term(a)),
                            (shape, SH["not"], negated), (negated, SH["class"], terms.term(b)))


_IRREFLEXIVE = "SELECT $this WHERE {{ $this {r} $this . }}"
_SYMMETRIC = ("SELECT $this ?value WHERE {{ $this {r} ?value . FILTER(!isLiteral(?value) && "
              "?value != $this) FILTER NOT EXISTS {{ ?value {r} $this }} }}")
_INVERSE = ("SELECT $this ?value WHERE {{ $this {r} ?value . FILTER(!isLiteral(?value)) "
            "FILTER NOT EXISTS {{ ?value {s} $this }} }}")
_TRANSITIVE = ("SELECT DISTINCT $this ?value WHERE {{ $this {r}/{r}+ ?value . "
               "FILTER NOT EXISTS {{ $this {r} ?value }} }}")


def _sparql(sem: SemanticsDecl, terms: Terms) -> Iterator[tuple]:
    term = lambda node: terms.term(node).n3()  # noqa: E731
    queries = [("irreflexive", d.subject, _IRREFLEXIVE.format(r=term(d.subject)))
               for d in sem.of(DeclKey.IRREFLEXIVE) if d.value == "true"]
    for d in sem.of(DeclKey.MATERIALIZE):
        if d.value == "symmetric":
            queries.append(("symmetric", d.subject, _SYMMETRIC.format(r=term(d.subject))))
        elif d.value == "transitive":
            queries.append(("transitive", d.subject, _TRANSITIVE.format(r=term(d.subject))))
        else:
            partner = next((b if a == d.subject else a for a, b in sem.inverse_pairs()
                            if d.subject in (a, b)), None)
            if partner is not None:
                queries.append(("inverse", d.subject,
                                _INVERSE.format(r=term(d.subject), s=term(partner))))
    for kind, relation, query in queries:
        yield from _sparql_shape(kind, relation, query, terms)


def _sparql_shape(kind: str, relation: str, query: str, terms: Terms) -> Iterator[tuple]:
    shape, constraint = _shape(kind, relation), BNode()
    yield from _target(shape, SH.targetSubjectsOf, relation, terms)
    yield shape, SH.sparql, constraint
    yield constraint, SH.select, Literal(query)
