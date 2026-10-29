# SPDX-License-Identifier: Apache-2.0
"""Read an OntoDisc store directory into an ``AuditSnapshot``, losslessly.

Reads only the JSON and Markdown artifacts the contract names; never opens model
weights, loads the store or imports training code. Every raw row is kept with its
location, duplicates included; a malformed row, field or file becomes a
diagnostic rather than disappearing or being coerced. A view is present only when
its container is valid: a missing, null or mistyped container leaves the view
absent, never a valid empty view. Node kinds are only what an artifact or the
producer's documented convention states: declared ontology entities stay
``UNKNOWN``; an undeclared token used only as an object is a literal when it is
numeric, an ISO date, or the object of a text relation. An audit schema sidecar
(``kgcheck.audit_schema``) may declare kinds, types and descriptions explicitly;
an explicit declaration overrides these readings.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from knowlytix.knowledge.kgcheck.adapters._json import (
    RawArtifact,
    finite_float,
    lexical_numbers,
    pointer_token,
    read_json_artifact,
    read_text_artifact,
)
from knowlytix.knowledge.kgcheck.contract import artifact_specs, contract_version
from knowlytix.knowledge.kgcheck.identity import assertion_id, node_id
from knowlytix.knowledge.kgcheck.models import (
    Assertion,
    AssertionRow,
    AuditSnapshot,
    Capabilities,
    DeclKey,
    Diagnostic,
    EvidenceLink,
    GraphView,
    JsonValue,
    LiteralValue,
    Node,
    NodeKind,
    NumericConflict,
    NumericSlot,
    OntologyDecl,
    PhaseRange,
    RawRef,
    SemanticsDecl,
    SourceText,
)
from knowlytix.knowledge.kgcheck.normalize import ISO_DATE, is_numeric_token

if TYPE_CHECKING:
    from knowlytix.knowledge.kgcheck.audit_schema import LoadedSidecar

ADAPTER = "ontodisc_store"
ADAPTER_VERSION = "7"
SOURCE_TEXT = "documents/combined.md"
# OntoDisc's built-in rule-text predicates (core ontology_score.RULE_TEXT_RELATIONS);
# a test pins this copy to the producer's constant.
TEXT_RELATIONS = frozenset({"has_condition", "has_amount_expression", "has_rule_text",
                           "has_description", "has_definition"})
_EVIDENCE_TEXT = ("kind", "section", "sentence", "span")

_SIDECAR = "audit_schema.json"  # kgcheck.audit_schema.SIDECAR, imported lazily there

Triple = tuple[str, str, str]
StatMap = dict[str, tuple[int, int] | str]


def read_store(store_dir: str | Path, audit_schema: str | Path | None = None, *,
               partial: bool = False) -> AuditSnapshot:
    """Snapshot of the store at ``store_dir``.

    ``audit_schema`` names a sidecar file to use instead of the store's own
    ``audit_schema.json``. ``partial`` reads a directory that holds only some
    artifacts by design (in-memory triples, a build before training): no
    artifact is then required. Raises FileNotFoundError if the store is not a
    directory or an explicit sidecar is not a file.
    """
    # imported here: audit_schema imports adapters._json, which loads this package
    from knowlytix.knowledge.kgcheck.audit_schema import load_sidecar
    directory = Path(store_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} is not a store directory")
    native = (directory / 'assertions').is_dir() and not (directory / 'triples.json').exists()
    if (directory / 'assertions/CURRENT').exists():
        from knowlytix.knowledge.assertions.store import load_snapshot
        try:
            native = load_snapshot(directory)['mode'] == 'enforce'
        except (OSError, ValueError, KeyError, TypeError):
            native = True  # A corrupt assertion store must never fall back to legacy.
    if native:
        if audit_schema is not None:
            raise ValueError('Typed sidecars for source-assertion graphs are not supported yet')
        from .source_assertions import read_assertion_store
        return read_assertion_store(directory)
    explicit = Path(audit_schema) if audit_schema is not None else None
    sidecar_path = explicit if explicit is not None else directory / _SIDECAR
    before = _stat_all(directory, sidecar_path)
    raw = {s.filename: read_json_artifact(directory, s.filename, s.required and not partial)
           for s in artifact_specs()}
    source = read_text_artifact(directory, SOURCE_TEXT)
    sidecar = load_sidecar(directory, explicit)
    builder = _Builder(raw, source, sidecar)
    builder.diags += _changed(before, _stat_all(directory, sidecar_path))
    return builder.build()


# --- stable read --------------------------------------------------------------
def _stat_all(directory: Path, sidecar: Path) -> StatMap:
    """(size, mtime) per present file; an error text when a file cannot be stat'ed.

    The sidecar is the file actually read (an explicit one replaces the store's).
    """
    out: StatMap = {}
    paths = {name: directory / name for name in [s.filename for s in artifact_specs()] + [SOURCE_TEXT]}
    for name, path in {**paths, _SIDECAR: sidecar}.items():
        try:
            st = path.stat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            out[name] = type(exc).__name__
            continue
        out[name] = (st.st_size, st.st_mtime_ns)
    return out


def _changed(before: StatMap, after: StatMap) -> list[Diagnostic]:
    names = sorted(set(before) | set(after))
    return [Diagnostic("changed_during_read", "file changed or became unreadable while the store "
                       "was read; the snapshot may mix states", RawRef(n))
            for n in names if before.get(n) != after.get(n) or isinstance(after.get(n), str)]


# --- typed access ------------------------------------------------------------------
_KINDS: dict[type, str] = {list: "a JSON list", dict: "a JSON object"}


def _is_triple(row: JsonValue) -> bool:
    return isinstance(row, list) and len(row) == 3 and all(isinstance(t, str) for t in row)


def _count(value: JsonValue) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _triple_rows(items: Iterable[tuple[JsonValue, RawRef]],
                 diags: list[Diagnostic]) -> list[tuple[Triple, RawRef]]:
    out: list[tuple[Triple, RawRef]] = []
    for row, ref in items:
        if _is_triple(row):
            out.append(((row[0], row[1], row[2]), ref))
        else:
            diags.append(Diagnostic("malformed_triple", f"not three strings: {row!r}"[:200], ref))
    return out


@dataclass
class _Builder:
    raw: dict[str, RawArtifact]
    source: RawArtifact
    sidecar: LoadedSidecar
    diags: list[Diagnostic] = field(default_factory=list)

    def __post_init__(self) -> None:
        for art in [*self.raw.values(), self.source]:
            self.diags += art.diagnostics
        self.diags += self.sidecar.diagnostics

    def _read(self) -> list[RawArtifact]:
        """Every artifact read, the sidecar last when there is one."""
        extra = [self.sidecar.artifact] if self.sidecar.artifact is not None else []
        return [*self.raw.values(), self.source, *extra]

    def note(self, code: str, message: str, ref: RawRef) -> None:
        self.diags.append(Diagnostic(code, message[:300], ref))

    def top(self, name: str, kind: type) -> list | dict | None:
        """The artifact's top-level container, or None (diagnosed) when unusable."""
        art = self.raw[name]
        if not art.parsed:
            return None
        if not isinstance(art.value, kind):
            self.note("invalid_container", f"top level must be {_KINDS[kind]}, "
                      f"got {type(art.value).__name__}", RawRef(name))
            return None
        return art.value

    def member(self, parent: dict, key: str, kind: type, ref: RawRef, *,
               required: bool) -> list | dict | None:
        """A field's container. Absent optional -> None silently; otherwise diagnosed."""
        if key not in parent:
            if required:
                self.note("missing_required_field", f"{key!r} is required", ref)
            return None
        value = parent[key]
        if not isinstance(value, kind):
            code = "null_container" if value is None else "unexpected_container"
            self.note(code, f"{key!r} must be {_KINDS[kind]}", RawRef(ref.artifact, f"{ref.pointer}/{key}"))
            return None
        return value

    # --- views -------------------------------------------------------------------
    def view_rows(self) -> dict[GraphView, list[tuple[Triple, RawRef]]]:
        views: dict[GraphView, list[tuple[Triple, RawRef]]] = {}
        stored = self.top("triples.json", list)
        if stored is not None:
            views[GraphView.STORED] = self._rows(stored, "triples.json", "")
        closed = self.top("closed_schema.json", dict)
        triples = None if closed is None else self.member(
            closed, "triples", list, RawRef("closed_schema.json"), required=True)
        if triples is not None:
            views[GraphView.CONFORMANT] = self._rows(triples, "closed_schema.json", "/triples")
        quarantined = self._quarantined()
        if quarantined is not None:
            views[GraphView.QUARANTINED] = quarantined
        return views

    def _rows(self, rows: list, name: str, prefix: str) -> list[tuple[Triple, RawRef]]:
        return _triple_rows(((r, RawRef(name, f"{prefix}/{i}")) for i, r in enumerate(rows)), self.diags)

    def _quarantined(self) -> list[tuple[Triple, RawRef]] | None:
        rows = self.top("quarantine.json", list)
        if rows is None:
            return None
        items: list[tuple[JsonValue, RawRef]] = []
        for i, r in enumerate(rows):
            if isinstance(r, dict) and "triple" in r:
                items.append((r["triple"], RawRef("quarantine.json", f"/{i}/triple")))
            else:
                self.note("malformed_quarantine_record", "record without a triple",
                          RawRef("quarantine.json", f"/{i}"))
        return _triple_rows(items, self.diags)

    # --- ontology ---------------------------------------------------------------------
    def ontology(self) -> OntologyDecl:
        spec = self.top("ontology.json", dict) or {}
        ref = RawRef("ontology.json")
        get = lambda key, kind: self.member(spec, key, kind, ref, required=bool(spec)) or kind()  # noqa: E731
        return OntologyDecl(entities=self._groups(get("entities", dict), "entities"),
                            relations=self._groups(get("relations", dict), "relations"),
                            distinct=tuple((a, b) for a, b in self._tuples(get("distinct", list), "distinct", 2)),
                            schema_rows=tuple((a, b, c) for a, b, c in self._tuples(get("schema", list), "schema", 3)))

    def _groups(self, groups: dict, key: str) -> dict[str, tuple[str, ...]]:
        out: dict[str, tuple[str, ...]] = {}
        for term, aliases in groups.items():
            if isinstance(aliases, list) and all(isinstance(a, str) for a in aliases):
                out[term] = tuple(aliases)
            else:
                self.note("malformed_alias_group", "aliases are not a list of strings",
                          RawRef("ontology.json", f"/{key}/{pointer_token(term)}"))
                out[term] = ()
        return out

    def _tuples(self, rows: list, key: str, size: int) -> list[list[str]]:
        out: list[list[str]] = []
        for i, row in enumerate(rows):
            if isinstance(row, list) and len(row) == size and all(isinstance(t, str) for t in row):
                out.append(row)
            else:
                self.note(f"malformed_{key}_row", f"expected {size} strings", RawRef("ontology.json", f"/{key}/{i}"))
        return out

    # --- evidence -----------------------------------------------------------------------
    def evidence(self) -> list[EvidenceLink]:
        art = self.raw["provenance.json"]
        if art.parsed and isinstance(art.value, list):
            self.note("unsupported_provenance_format", "a flat provenance list (GEODE) is not "
                      "the nested OntoDisc format", RawRef("provenance.json"))
            return []
        prov = self.top("provenance.json", dict)
        records = None if prov is None else self.member(
            prov, "triples", list, RawRef("provenance.json"), required=True)
        return [link for i, rec in enumerate(records or []) for link in self._record_links(i, rec)]

    def _record_links(self, i: int, rec: JsonValue) -> list[EvidenceLink]:
        ref = RawRef("provenance.json", f"/triples/{i}")
        if not isinstance(rec, dict) or not _is_triple(rec.get("triple")):
            self.note("malformed_provenance_record", "record without a valid triple", ref)
            return []
        evidence = self.member(rec, "evidence", list, ref, required=True) or []
        aid = assertion_id(*rec["triple"])
        return [link for j, ev in enumerate(evidence)
                if (link := self._link(aid, ev, RawRef(ref.artifact, f"{ref.pointer}/evidence/{j}")))]

    def _link(self, aid: str, ev: JsonValue, ref: RawRef) -> EvidenceLink | None:
        bad = self._evidence_problems(ev)
        if bad:
            self.note("malformed_evidence", f"invalid field(s): {', '.join(bad)}", ref)
            return None
        return EvidenceLink(aid, ev["kind"], ev["section"], ev["sentence_index"], ev["sentence"],
                            ev["span"], ref, ev.get("document"))

    @staticmethod
    def _evidence_problems(ev: JsonValue) -> list[str]:
        if not isinstance(ev, dict):
            return ["record"]
        bad = [k for k in _EVIDENCE_TEXT if not isinstance(ev.get(k), str)]
        if not _count(ev.get("sentence_index")):
            bad.append("sentence_index")
        if "document" in ev and not isinstance(ev["document"], str):
            bad.append("document")
        return bad

    def sources(self) -> list[SourceText]:
        record = self.source.record
        if self.source.text is None or record.sha256 is None:
            return []
        return [SourceText(record.name, record.sha256, self.source.text)]

    # --- numeric memory -------------------------------------------------------------------
    def numeric_slots(self) -> list[NumericSlot]:
        enm = self.top("enm.json", dict) or {}
        art = self.raw["enm.json"]
        lexical = lexical_numbers(art.text) if art.text is not None else None
        lexical = lexical if isinstance(lexical, dict) else {}
        return [slot for key, value in enm.items()
                if (slot := self._slot(key, value, lexical.get(key)))]

    def _slot(self, key: str, value: JsonValue, lexical: JsonValue) -> NumericSlot | None:
        ref = RawRef("enm.json", f"/{pointer_token(key)}")
        relation, sep, head = key.partition("::")  # relation first; a head may contain "::"
        number = finite_float(value)
        if not sep or not relation or number is None:
            self.note("malformed_numeric_slot", f"{key!r}: {str(lexical or value)[:60]} is not a "
                      "finite number under a '<relation>::<head>' key", ref)
            return None
        return NumericSlot(relation, head, number, str(lexical if lexical is not None else value), ref)

    def numeric_conflicts(self) -> list[NumericConflict]:
        meta = self.top("metadata.json", dict) or {}
        rows = self.member(meta, "numeric_conflicts", list, RawRef("metadata.json"), required=False)
        return [c for i, row in enumerate(rows or []) if (c := self._conflict(i, row))]

    def _conflict(self, i: int, row: JsonValue) -> NumericConflict | None:
        ref = RawRef("metadata.json", f"/numeric_conflicts/{i}")
        values = row.get("values") if isinstance(row, dict) else None
        numbers = [finite_float(v) for v in values] if isinstance(values, list) else []
        ok = (isinstance(row, dict) and isinstance(row.get("relation"), str)
              and isinstance(row.get("head"), str) and len(numbers) >= 2 and None not in numbers)
        if not ok:
            self.note("malformed_numeric_conflict", "needs string relation and head and at least "
                      "two finite numeric values", ref)
            return None
        return NumericConflict(row["relation"], row["head"], tuple(numbers), ref)

    def phase_ranges(self) -> list[PhaseRange]:
        out: list[PhaseRange] = []
        for name, cfg in (self.top("phase.json", dict) or {}).items():
            ref = RawRef("phase.json", f"/{pointer_token(name)}")
            lo, hi = ((finite_float(cfg.get("v_min")), finite_float(cfg.get("v_max")))
                      if isinstance(cfg, dict) else (None, None))
            if lo is None or hi is None:
                self.note("malformed_phase_range", "v_min/v_max are not finite numbers", ref)
                continue
            out.append(PhaseRange(name, lo, hi, ref))
        return out

    # --- nodes --------------------------------------------------------------------------------
    def nodes(self, ontology: OntologyDecl, views: dict[GraphView, list[tuple[Triple, RawRef]]],
              text_relations: frozenset[str]) -> list[Node]:
        usage = _Usage()
        usage.declare_ontology(ontology)
        usage.declare_adapter(self.top("adapter.json", dict) or {})
        for rows in views.values():
            for (h, r, t), ref in rows:
                usage.use(h, r, t, ref.artifact)
        return _annotate(usage.finish(ontology, text_relations), self.sidecar.semantics.accepted())

    def text_relations(self) -> frozenset[str]:
        meta = self.top("metadata.json", dict) or {}
        declared = self.member(meta, "text_relations", list, RawRef("metadata.json"), required=False)
        for i, r in enumerate(declared or []):
            if not isinstance(r, str):
                self.note("malformed_text_relation", "a text relation must be a string",
                          RawRef("metadata.json", f"/text_relations/{i}"))
        return TEXT_RELATIONS | frozenset(r for r in declared or [] if isinstance(r, str))

    def capabilities(self, sources: list[SourceText], evidence_ok: bool) -> Capabilities:
        meta = self.top("metadata.json", dict) or {}
        verification = self.raw["semantic_verification.json"]
        semantic = (meta.get("grounding_policy") == "lexical_semantic"
                    or (verification.parsed and isinstance(verification.value, list)))
        return Capabilities(provenance=evidence_ok, source_text=bool(sources),
                            numeric_conflicts_recorded=isinstance(meta.get("numeric_conflicts"), list),
                            semantic_verification=semantic,
                            **_declared_capabilities(self.sidecar.semantics.accepted()),
                            adapter=ADAPTER, adapter_version=ADAPTER_VERSION)

    def snapshot_id(self) -> str:
        digests = sorted((a.record.name, a.record.sha256 or "") for a in self._read()
                         if a.record.present)
        payload = json.dumps([ADAPTER, ADAPTER_VERSION, contract_version(), digests])
        return "s:" + hashlib.sha256(payload.encode()).hexdigest()[:20]

    def _provenance_usable(self) -> bool:
        prov = self.raw["provenance.json"]
        return prov.parsed and isinstance(prov.value, dict) and isinstance(prov.value.get("triples"), list)

    def build(self) -> AuditSnapshot:
        views = self.view_rows()
        ontology = self.ontology()
        nodes = self.nodes(ontology, views, self.text_relations())
        triples = dict.fromkeys(t for rows in views.values() for t, _ in rows)
        evidence, sources = self.evidence(), self.sources()
        return AuditSnapshot(
            snapshot_id=self.snapshot_id(), nodes=tuple(nodes),
            assertions=tuple(_assertion(t) for t in triples),
            views={v: tuple(AssertionRow(assertion_id(*t), ref) for t, ref in rows)
                   for v, rows in views.items()},
            ontology=ontology, semantics=self.sidecar.semantics,
            evidence=tuple(evidence), sources=tuple(sources),
            numeric_slots=tuple(self.numeric_slots()), numeric_conflicts=tuple(self.numeric_conflicts()),
            phase_ranges=tuple(self.phase_ranges()),
            capabilities=self.capabilities(sources, self._provenance_usable()),
            artifacts=tuple(a.record for a in self._read()),
            diagnostics=tuple(dict.fromkeys(self.diags)), metadata=self.top("metadata.json", dict) or {},
            parsed={n: a.value for n, a in self.raw.items() if a.parsed})


def _assertion(t: Triple) -> Assertion:
    h, r, o = t
    return Assertion(assertion_id(h, r, o), node_id("entity", h), node_id("relation", r),
                     node_id("entity", o))


# --- node kinds ---------------------------------------------------------------------------------
@dataclass
class _Usage:
    """Where each token is declared and how it is used, to settle node kinds."""

    declared: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    as_head: set[str] = field(default_factory=set)
    as_object: dict[str, set[str]] = field(default_factory=dict)

    def _declare(self, namespace: str, token: str, artifact: str) -> None:
        by = self.declared.setdefault((namespace, token), [])
        if artifact not in by:
            by.append(artifact)

    def declare_ontology(self, ontology: OntologyDecl) -> None:
        for term in ontology.entities:
            self._declare("entity", term, "ontology.json")
        for term in ontology.relations:
            self._declare("relation", term, "ontology.json")

    def declare_adapter(self, adapter: dict) -> None:
        for key, namespace in (("entity_to_idx", "entity"), ("relation_to_idx", "relation")):
            vocab = adapter.get(key)
            for token in vocab if isinstance(vocab, dict) else ():
                self._declare(namespace, token, "adapter.json")

    def use(self, head: str, relation: str, obj: str, artifact: str) -> None:
        for namespace, token in (("entity", head), ("relation", relation), ("entity", obj)):
            self._declare(namespace, token, artifact)
        self.as_head.add(head)
        self.as_object.setdefault(obj, set()).add(relation)

    def finish(self, ontology: OntologyDecl, text_relations: frozenset[str]) -> list[Node]:
        return [self._node(ns, tok, by, ontology, text_relations)
                for (ns, tok), by in sorted(self.declared.items())]

    def _node(self, ns: str, tok: str, by: list[str], ontology: OntologyDecl,
              text_relations: frozenset[str]) -> Node:
        groups = ontology.entities if ns == "entity" else ontology.relations
        if ns == "relation":
            return Node(node_id(ns, tok), tok, ns, NodeKind.PROPERTY, groups.get(tok, ()), tuple(by))
        literal = None if tok in groups else self._literal(tok, text_relations)
        kind = NodeKind.LITERAL if literal else NodeKind.UNKNOWN
        return Node(node_id(ns, tok), tok, ns, kind, groups.get(tok, ()), tuple(by), literal)

    def _literal(self, tok: str, text_relations: frozenset[str]) -> LiteralValue | None:
        relations = self.as_object.get(tok)
        if not relations or tok in self.as_head:
            return None
        if relations <= text_relations:
            return LiteralValue(tok, "text_relation_object")
        if is_numeric_token(tok):
            return LiteralValue(tok, "numeric_token")
        if ISO_DATE.fullmatch(tok):
            return LiteralValue(tok, "iso_date_token")
        return None


# --- audit schema sidecar ------------------------------------------------------------------------
def _declared_capabilities(sem: SemanticsDecl) -> dict[str, bool]:
    """What the sidecar's declarations let rules check."""
    has = lambda *keys: any(sem.of(k) for k in keys)  # noqa: E731
    return {"node_kinds": has(DeclKey.KIND, DeclKey.TYPE),
            "descriptions": has(DeclKey.DESCRIPTION),
            "hierarchy": has(DeclKey.SUBCLASS_PREDICATE, DeclKey.TYPE_PREDICATE, DeclKey.TYPE),
            "relation_properties": has(DeclKey.INVERSE_OF, DeclKey.SYMMETRIC, DeclKey.ASYMMETRIC,
                                       DeclKey.TRANSITIVE),
            "naming_conventions": has(DeclKey.CASE, DeclKey.SEPARATOR, DeclKey.NUMBER),
            "typed_constraints": has(*TYPED_KEYS),
            "competency_questions": bool(sem.competency)}


# declarations that constrain instance data (CON rules)
TYPED_KEYS = (DeclKey.DOMAIN, DeclKey.RANGE, DeclKey.DATATYPE, DeclKey.MINIMUM, DeclKey.MAXIMUM,
              DeclKey.ALLOWED_VALUE, DeclKey.MIN_COUNT, DeclKey.MAX_COUNT, DeclKey.DISJOINT,
              DeclKey.IRREFLEXIVE, DeclKey.MATERIALIZE)
# declarations whose values name classes
CLASS_VALUED = (DeclKey.TYPE, DeclKey.DOMAIN, DeclKey.RANGE, DeclKey.DISJOINT)


_NODE_KEYS = (DeclKey.KIND, DeclKey.TYPE, DeclKey.DESCRIPTION, DeclKey.LANGUAGE, DeclKey.STANDALONE,
              DeclKey.INVERSE_OF, DeclKey.SYMMETRIC, DeclKey.ASYMMETRIC, DeclKey.TRANSITIVE)


def _annotate(nodes: list[Node], sem: SemanticsDecl) -> list[Node]:
    """Nodes with declared kinds, types and descriptions; terms only the sidecar names added.

    The sidecar's ``terms`` and ``relations`` entries and every class named as a
    type, domain, range or disjoint class become nodes. Roots, module members and predicates that name no node
    are left for rules to report, not added.
    """
    if sem.version is None:
        return nodes
    by_id = {n.node_id: n for n in nodes}
    classes = {d.value for k in CLASS_VALUED for d in sem.of(k)}
    subjects = {d.subject for k in _NODE_KEYS for d in sem.of(k) if d.subject} | classes
    for subject in subjects:
        by_id[subject] = _declared(by_id.get(subject), subject, sem, subject in classes)
    return [by_id[k] for k in sorted(by_id)]


def _declared(node: Node | None, subject: str, sem: SemanticsDecl, is_class: bool) -> Node:
    namespace, _, token = subject.partition(":")
    base = node or Node(subject, token, namespace,
                        NodeKind.PROPERTY if namespace == "relation" else NodeKind.UNKNOWN)
    stated = sem.find(DeclKey.KIND, subject)
    kind = NodeKind(stated.value) if stated else (NodeKind.CLASS if is_class else base.kind)
    literal = base.literal or LiteralValue(token, "declared_literal") if kind is NodeKind.LITERAL \
        else None
    description = sem.find(DeclKey.DESCRIPTION, subject)
    by = base.declared_by if _SIDECAR in base.declared_by else (*base.declared_by, _SIDECAR)
    return replace(base, kind=kind, literal=literal, declared_by=by,
                   description=description.value if description else base.description,
                   types=sem.values(DeclKey.TYPE, subject) or base.types)
