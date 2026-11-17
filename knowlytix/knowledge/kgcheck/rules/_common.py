# SPDX-License-Identifier: Apache-2.0
"""Small read helpers shared by the shipped rules."""

from __future__ import annotations

from collections.abc import Mapping

from knowlytix.knowledge.kgcheck.adapters._json import pointer_token
from knowlytix.knowledge.kgcheck.frozen import FrozenJson
from knowlytix.knowledge.kgcheck.models import (
    AuditSnapshot,
    GraphView,
    NodeKind,
    RawRef,
)
from knowlytix.knowledge.kgcheck.normalize import (
    entity_resolution_key,
    relation_resolution_key,
)

Triple = tuple[str, str, str]
_TRIPLE_VIEWS = (GraphView.STORED, GraphView.CONFORMANT, GraphView.QUARANTINED)


def label(snapshot: AuditSnapshot, node_id: str) -> str:
    """A node's raw token; node ids are ``<namespace>:<token>``."""
    node = snapshot.node_index.get(node_id)
    return node.label if node else node_id.partition(":")[2]


def triple(snapshot: AuditSnapshot, assertion_id: str) -> Triple:
    a = snapshot.assertion_index[assertion_id]
    return label(snapshot, a.head), label(snapshot, a.predicate), label(snapshot, a.object)


def rows_by_assertion(snapshot: AuditSnapshot, view: GraphView) -> dict[str, list[RawRef]]:
    """Raw row locations of each assertion in ``view``, in artifact order."""
    out: dict[str, list[RawRef]] = {}
    for row in snapshot.view(view):
        out.setdefault(row.assertion_id, []).append(row.raw)
    return out


def triple_views(snapshot: AuditSnapshot) -> tuple[GraphView, ...]:
    """The present views that hold triples, in a fixed order."""
    return tuple(v for v in _TRIPLE_VIEWS if snapshot.has_view(v))


_GROUP_KEY = {"entity": "entities", "relation": "relations"}


def term_ref(namespace: str, term: str) -> RawRef:
    """Location of a declared term in ontology.json."""
    return RawRef("ontology.json", f"/{_GROUP_KEY[namespace]}/{pointer_token(term)}")


def mapping(value: FrozenJson) -> Mapping[str, FrozenJson]:
    return value if isinstance(value, Mapping) else {}


def sequence(value: FrozenJson) -> tuple[FrozenJson, ...]:
    return value if isinstance(value, tuple) else ()


def artifact(snapshot: AuditSnapshot, name: str) -> Mapping[str, FrozenJson]:
    """A parsed object artifact, or an empty mapping when absent or not an object."""
    return mapping(snapshot.parsed.get(name))


def quarantine_reasons(snapshot: AuditSnapshot) -> dict[Triple, list[str]]:
    """Reasons recorded per quarantined triple, across all its records."""
    out: dict[Triple, list[str]] = {}
    for record in sequence(snapshot.parsed.get("quarantine.json")):
        rec = mapping(record)
        t, reasons = sequence(rec.get("triple")), sequence(rec.get("reasons"))
        if len(t) == 3 and all(isinstance(x, str) for x in t):
            bucket = out.setdefault((t[0], t[1], t[2]), [])
            bucket += [r for r in reasons if isinstance(r, str) and r not in bucket]
    return out


class Resolver:
    """Declared terms by OntoDisc resolution key, in the producer's order.

    ``targets(token)`` lists every declared term the token's key reaches, as a
    canonical name or as an alias, in the order OntoDisc registers surfaces: by
    term in declaration order, each term's name before its aliases. The first
    target is the one OntoDisc resolves to; more than one is an ambiguity.
    """

    def __init__(self, groups: Mapping[str, tuple[str, ...]], *, relation: bool = False) -> None:
        self.relation = relation
        self.names: dict[str, list[str]] = {}
        self.aliases: dict[str, list[str]] = {}
        self._order: dict[str, list[str]] = {}
        for term, aliases in groups.items():
            for key in self.keys(term):
                _add(self.names, key, term)
                _add(self._order, key, term)
            for alias in aliases:
                for key in self.keys(alias):
                    _add(self.aliases, key, term)
                    _add(self._order, key, term)

    def keys(self, token: str) -> frozenset[str]:
        key = relation_resolution_key(token) if self.relation else entity_resolution_key(token)
        return frozenset({key}) if key else frozenset()

    def targets(self, token: str) -> list[str]:
        return [t for key in sorted(self.keys(token)) for t in self._order.get(key, ())]

    def declared(self, token: str) -> bool:
        return bool(self.targets(token))


def _add(index: dict[str, list[str]], key: str, term: str) -> None:
    bucket = index.setdefault(key, [])
    if term not in bucket:
        bucket.append(term)


def entity_resolver(snapshot: AuditSnapshot) -> Resolver:
    return Resolver(snapshot.ontology.entities)


def relation_resolver(snapshot: AuditSnapshot) -> Resolver:
    return Resolver(snapshot.ontology.relations, relation=True)


def is_literal(snapshot: AuditSnapshot, node_id: str) -> bool:
    node = snapshot.node_index.get(node_id)
    return node is not None and node.kind is NodeKind.LITERAL
