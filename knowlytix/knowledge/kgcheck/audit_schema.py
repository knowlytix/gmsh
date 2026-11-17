# SPDX-License-Identifier: Apache-2.0
"""The optional audit schema sidecar: metadata an OntoDisc store does not record.

``audit_schema.json`` declares node kinds and types, descriptions, hierarchy and
type predicates, relation properties and constraints, disjoint classes, the
closed-world scope of typing, expected modules, naming conventions and
competency questions (contract: ``schemas/audit_schema.schema.json``). It is read from the store
directory, or from an explicit path that takes precedence. It is flattened into
``Declaration`` records that keep who supplied each statement and its JSON
pointer. A sidecar that does not parse cleanly (a repeated key, a non-finite
number), violates the contract or contradicts itself supplies no metadata at all: partial metadata could let a rule report a check as passed on
declarations the author never made consistent.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from decimal import Decimal
from functools import cache
from importlib.resources import files
from pathlib import Path

from knowlytix.knowledge.kgcheck.adapters._json import (
    RawArtifact,
    pointer_token,
    read_json_artifact,
)
from knowlytix.knowledge.kgcheck.identity import node_id
from knowlytix.knowledge.kgcheck.models import (
    CompetencyQuestion,
    Declaration,
    DeclKey,
    DeclSource,
    Diagnostic,
    JsonObject,
    RawRef,
    SemanticsDecl,
)
from knowlytix.knowledge.kgcheck.validate import Validator, Violation

SIDECAR = "audit_schema.json"
SCHEMA_FILE = "audit_schema.schema.json"

_TERM_FLAGS = {"standalone": DeclKey.STANDALONE}
_RELATION_FLAGS = {"symmetric": DeclKey.SYMMETRIC, "asymmetric": DeclKey.ASYMMETRIC,
                   "transitive": DeclKey.TRANSITIVE, "irreflexive": DeclKey.IRREFLEXIVE,
                   "inherited": DeclKey.INHERITED}
_VALUE = {"datatype": DeclKey.DATATYPE, "minimum": DeclKey.MINIMUM, "maximum": DeclKey.MAXIMUM,
          "unit": DeclKey.UNIT, "period": DeclKey.PERIOD, "qualifier": DeclKey.QUALIFIER}
_CARDINALITY = {"min": DeclKey.MIN_COUNT, "max": DeclKey.MAX_COUNT, "equality": DeclKey.EQUALITY,
                "complete": DeclKey.COMPLETE}
NUMERIC_DATATYPES = frozenset({"decimal", "integer"})
_CONVENTIONS = {"case": DeclKey.CASE, "separator": DeclKey.SEPARATOR, "number": DeclKey.NUMBER}


@cache
def _schema_text() -> str:
    return files("knowlytix.knowledge.kgcheck").joinpath("schemas", SCHEMA_FILE).read_text(
        encoding="utf-8")


def sidecar_schema() -> JsonObject:
    """The sidecar contract (a fresh copy on every call)."""
    return json.loads(_schema_text())


def sidecar_violations(value: object) -> list[Violation]:
    """Where ``value`` breaks the sidecar contract; empty when it conforms."""
    schema = {k: v for k, v in sidecar_schema().items() if not k.startswith("x-")}  # annotations
    return Validator(schema).errors(value, schema)


@dataclass(frozen=True)
class LoadedSidecar:
    """The sidecar as read: the raw artifact (None if there is none) and its declarations."""

    artifact: RawArtifact | None
    semantics: SemanticsDecl
    diagnostics: tuple[Diagnostic, ...] = ()


def load_sidecar(store_dir: Path, explicit: Path | None = None) -> LoadedSidecar:
    """The sidecar for ``store_dir``; ``explicit`` overrides the store's own copy.

    Raises FileNotFoundError when ``explicit`` names no file.
    """
    notes: list[Diagnostic] = []
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"audit schema {explicit} is not a file")
        own = store_dir / SIDECAR
        if own.is_file() and own.resolve() != explicit.resolve():
            notes.append(Diagnostic("audit_schema_overridden", f"the store's {SIDECAR} was ignored "
                                    f"in favour of {explicit.name}", RawRef(SIDECAR)))
        target = explicit.resolve()  # named explicitly, so a symbolic link is followed
        art = read_json_artifact(target.parent, target.name, True)
    elif (store_dir / SIDECAR).is_file():
        art = read_json_artifact(store_dir, SIDECAR, False)
    else:
        return LoadedSidecar(None, SemanticsDecl())
    return _interpret(_renamed(art), notes)


def sidecar_from_artifact(art: RawArtifact) -> LoadedSidecar:
    """Interpret an already-read sidecar artifact, recorded under the standard name."""
    return _interpret(_renamed(art), [])


def _renamed(art: RawArtifact) -> RawArtifact:
    """Record an explicit sidecar under the standard name, so reports read the same."""
    record = replace(art.record, name=SIDECAR)
    diags = tuple(replace(d, location=RawRef(SIDECAR, d.location.pointer)) for d in art.diagnostics)
    return replace(art, record=record, diagnostics=diags)


def _interpret(art: RawArtifact, notes: list[Diagnostic]) -> LoadedSidecar:
    notes += art.diagnostics
    if not art.parsed or art.diagnostics:  # e.g. a repeated key: the file is ambiguous
        return LoadedSidecar(art, SemanticsDecl(), tuple(notes))
    violations = sidecar_violations(art.value)
    if violations:
        notes += [Diagnostic("invalid_audit_schema", v.message[:300], RawRef(SIDECAR, v.pointer))
                  for v in violations]
        return LoadedSidecar(art, SemanticsDecl(), tuple(notes))
    assert isinstance(art.value, dict)
    declarations = tuple(_flatten(art.value))
    conflicts = contradictions(declarations)
    if conflicts:
        return LoadedSidecar(art, SemanticsDecl(), tuple(notes + conflicts))
    semantics = SemanticsDecl(str(art.value["version"]), declarations, competency(art.value))
    return LoadedSidecar(art, semantics, tuple(notes))


# --- flattening -------------------------------------------------------------------------
def declarations(doc: Mapping[str, object]) -> tuple[Declaration, ...]:
    """The declarations of a sidecar document that conforms to the contract."""
    return tuple(_flatten(doc))


@dataclass(frozen=True)
class _Emit:
    """Builds declarations under one JSON pointer prefix with an inherited source."""

    prefix: str
    source: DeclSource

    def at(self, *keys: str) -> _Emit:
        return _Emit(self.prefix + "".join(f"/{pointer_token(k)}" for k in keys), self.source)

    def within(self, block: Mapping[str, object]) -> _Emit:
        """The same location with the block's own ``source``, when it states one."""
        return _Emit(self.prefix, DeclSource(block.get("source", self.source)))

    def __call__(self, subject: str, key: DeclKey, value: str, *pointer: str) -> Declaration:
        return Declaration(subject, key, value, self.source, RawRef(SIDECAR, self.at(*pointer).prefix))


def _flatten(doc: Mapping[str, object]) -> Iterator[Declaration]:
    root = _Emit("", DeclSource(doc.get("default_source", DeclSource.USER)))
    if "language" in doc:
        yield root("", DeclKey.LANGUAGE, str(doc["language"]), "language")
    for token, block in _items(doc, "terms"):
        yield from _term(root.at("terms", token).within(block), token, block)
    for token, block in _items(doc, "relations"):
        yield from _relation(root.at("relations", token).within(block), token, block)
    yield from _hierarchy(root, doc.get("hierarchy"))
    for name, block in _items(doc, "modules"):
        emit = root.at("modules", name).within(block)
        for i, member in enumerate(block["members"]):  # type: ignore[union-attr]
            yield emit(node_id("entity", member), DeclKey.MODULE, name, "members", str(i))
    yield from _conventions(root, doc.get("conventions"))
    for i, group in enumerate(doc.get("disjoint", ())):  # type: ignore[arg-type]
        for j, cls in enumerate(group):
            yield root(f"disjoint:{i}", DeclKey.DISJOINT, node_id("entity", cls), "disjoint",
                       str(i), str(j))
    closed = doc.get("closed_world")
    if isinstance(closed, Mapping) and "typing" in closed:
        emit = root.at("closed_world").within(closed)
        yield emit("", DeclKey.CLOSED_TYPING, _flag(closed["typing"]), "typing")


def competency(doc: Mapping[str, object]) -> tuple[CompetencyQuestion, ...]:
    """The competency questions of a sidecar document that conforms to the contract."""
    default = DeclSource(doc.get("default_source", DeclSource.USER))
    out = []
    for qid, block in _items(doc, "competency"):
        out.append(CompetencyQuestion(
            qid, str(block["relation"]), DeclSource(block.get("source", default)),
            RawRef(SIDECAR, f"/competency/{pointer_token(qid)}"), str(block.get("question", "")),
            block.get("head"), block.get("object"),  # type: ignore[arg-type]
            tuple(block.get("expected", ())),  # type: ignore[arg-type]
            bool(block.get("absent", False)), bool(block.get("exhaustive", False))))
    return tuple(out)


def _items(doc: Mapping[str, object], key: str) -> list[tuple[str, Mapping[str, object]]]:
    block = doc.get(key) or {}
    assert isinstance(block, Mapping)
    return sorted(block.items())


def _flag(value: object) -> str:
    return "true" if value else "false"


def _term(emit: _Emit, token: str, block: Mapping[str, object]) -> Iterator[Declaration]:
    subject = node_id("entity", token)
    for key, decl in (("kind", DeclKey.KIND), ("description", DeclKey.DESCRIPTION),
                      ("language", DeclKey.LANGUAGE)):
        if key in block:
            yield emit(subject, decl, str(block[key]), key)
    for i, cls in enumerate(block.get("types", ())):  # type: ignore[arg-type]
        yield emit(subject, DeclKey.TYPE, node_id("entity", cls), "types", str(i))
    for key, decl in _TERM_FLAGS.items():
        if key in block:
            yield emit(subject, decl, _flag(block[key]), key)


def _relation(emit: _Emit, token: str, block: Mapping[str, object]) -> Iterator[Declaration]:
    subject = node_id("relation", token)
    for key, decl in (("description", DeclKey.DESCRIPTION), ("language", DeclKey.LANGUAGE)):
        if key in block:
            yield emit(subject, decl, str(block[key]), key)
    if "inverse_of" in block:
        yield emit(subject, DeclKey.INVERSE_OF, node_id("relation", str(block["inverse_of"])),
                   "inverse_of")
    for key, decl in _RELATION_FLAGS.items():
        if key in block:
            yield emit(subject, decl, _flag(block[key]), key)
    for key, decl in (("domain", DeclKey.DOMAIN), ("range", DeclKey.RANGE)):
        for i, cls in enumerate(block.get(key, ())):  # type: ignore[arg-type]
            yield emit(subject, decl, node_id("entity", cls), key, str(i))
    yield from _relation_blocks(emit, subject, block)


def _relation_blocks(emit: _Emit, subject: str, block: Mapping[str, object]) -> Iterator[Declaration]:
    value = block.get("value") or {}
    assert isinstance(value, Mapping)
    for key, decl in _VALUE.items():
        if key in value:
            yield emit(subject, decl, _text(value[key]), "value", key)
    for i, allowed in enumerate(value.get("allowed", ())):
        yield emit(subject, DeclKey.ALLOWED_VALUE, str(allowed), "value", "allowed", str(i))
    cardinality = block.get("cardinality") or {}
    assert isinstance(cardinality, Mapping)
    for key, decl in _CARDINALITY.items():
        if key in cardinality:
            value = cardinality[key]
            if isinstance(value, float):  # the contract accepts 2.0 as an integer count
                value = int(value)
            yield emit(subject, decl, _text(value), "cardinality", key)
    materialize = block.get("materialize") or {}
    assert isinstance(materialize, Mapping)
    for key in ("inverse", "symmetric", "transitive"):
        if materialize.get(key):
            yield emit(subject, DeclKey.MATERIALIZE, key, "materialize", key)


def _text(value: object) -> str:
    """A declared value as text: flags as true/false, numbers as written in the JSON."""
    if isinstance(value, bool):
        return _flag(value)
    return repr(value) if isinstance(value, float) else str(value)


def _hierarchy(root: _Emit, block: object) -> Iterator[Declaration]:
    if not isinstance(block, Mapping):
        return
    emit = root.at("hierarchy").within(block)
    for key, decl, ns in (("subclass_predicates", DeclKey.SUBCLASS_PREDICATE, "relation"),
                          ("type_predicates", DeclKey.TYPE_PREDICATE, "relation"),
                          ("roots", DeclKey.ROOT, "entity")):
        for i, token in enumerate(block.get(key, ())):
            yield emit(node_id(ns, token), decl, "true", key, str(i))
    if "acyclic" in block:
        yield emit("", DeclKey.ACYCLIC, _flag(block["acyclic"]), "acyclic")


def _conventions(root: _Emit, block: object) -> Iterator[Declaration]:
    if not isinstance(block, Mapping):
        return
    emit = root.at("conventions").within(block)
    for group, namespace in (("entities", "entity"), ("relations", "relation")):
        style = block.get(group, {})
        for key, decl in _CONVENTIONS.items():
            if key in style:
                yield emit(namespace, decl, str(style[key]), group, key)
        for i, token in enumerate(style.get("exceptions", ())):
            yield emit(node_id(namespace, token), DeclKey.CONVENTION_EXCEPTION, "true",
                       group, "exceptions", str(i))


# --- contradictions ------------------------------------------------------------------------
def contradictions(declarations: tuple[Declaration, ...]) -> list[Diagnostic]:
    """Declarations that cannot all hold, one diagnostic per contradiction."""
    sem = SemanticsDecl("check", declarations)
    found = [*_symmetry(sem), *_inverses(sem), *_kinds(sem), *_predicates(sem), *_bounds(sem),
             *_value_contract(sem), *_materialization(sem)]
    return [Diagnostic("contradictory_declaration", message, decl.raw) for message, decl in found]


def _true(sem: SemanticsDecl, key: DeclKey) -> dict[str, Declaration]:
    return {d.subject: d for d in sem.of(key) if d.value == "true"}


def _symmetry(sem: SemanticsDecl) -> Iterator[tuple[str, Declaration]]:
    asymmetric = _true(sem, DeclKey.ASYMMETRIC)
    for subject in sorted(set(_true(sem, DeclKey.SYMMETRIC)) & set(asymmetric)):
        yield f"{subject} is declared both symmetric and asymmetric", asymmetric[subject]
    for d in sem.of(DeclKey.INVERSE_OF):
        if d.subject == d.value and d.subject in asymmetric:
            yield f"{d.subject} is its own inverse, so it cannot be asymmetric", d


def _inverses(sem: SemanticsDecl) -> Iterator[tuple[str, Declaration]]:
    partners: dict[str, set[str]] = {}
    for a, b in sem.inverse_pairs():
        partners.setdefault(a, set()).add(b)
        partners.setdefault(b, set()).add(a)
    for d in sem.of(DeclKey.INVERSE_OF):
        for subject in (d.subject, d.value):
            if len(partners.get(subject, ())) > 1:
                yield (f"{subject} has more than one declared inverse: "
                       f"{', '.join(sorted(partners[subject]))}", d)
                partners[subject] = set()  # report each relation once


def _kinds(sem: SemanticsDecl) -> Iterator[tuple[str, Declaration]]:
    kinds = {d.subject: d for d in sem.of(DeclKey.KIND)}
    for key, use in ((DeclKey.TYPE, "a type"), (DeclKey.DOMAIN, "a domain"),
                     (DeclKey.RANGE, "a range"), (DeclKey.DISJOINT, "a disjoint class")):
        for d in sem.of(key):
            declared = kinds.get(d.value)
            if declared is not None and declared.value != "class":
                yield f"{d.value} is used as {use} but declared {declared.value}", declared


def _bounds(sem: SemanticsDecl) -> Iterator[tuple[str, Declaration]]:
    for low_key, high_key, what in ((DeclKey.MINIMUM, DeclKey.MAXIMUM, "value"),
                                    (DeclKey.MIN_COUNT, DeclKey.MAX_COUNT, "cardinality")):
        for low in sem.of(low_key):
            high = sem.find(high_key, low.subject)
            if high is not None and Decimal(low.value) > Decimal(high.value):
                yield (f"{low.subject} declares a {what} minimum {low.value} above its "
                       f"maximum {high.value}", high)


def _value_contract(sem: SemanticsDecl) -> Iterator[tuple[str, Declaration]]:
    """A value contract no object can satisfy, read under its effective datatype."""
    from knowlytix.knowledge.kgcheck.datatypes import (
        effective_datatype,
        parse_typed,
        value_relations,
    )
    for d in sem.of(DeclKey.DATATYPE):
        if d.value not in NUMERIC_DATATYPES:
            for key in (DeclKey.MINIMUM, DeclKey.MAXIMUM, DeclKey.UNIT, DeclKey.PERIOD,
                        DeclKey.QUALIFIER):
                if sem.find(key, d.subject) is not None:
                    yield f"{d.subject} declares {key.value} for {d.value} values", d
    for relation in value_relations(sem):
        datatype = effective_datatype(sem, relation)
        low, high = sem.find(DeclKey.MINIMUM, relation), sem.find(DeclKey.MAXIMUM, relation)
        for allowed in sem.of(DeclKey.ALLOWED_VALUE):
            if allowed.subject != relation:
                continue
            parsed = parse_typed(allowed.value, datatype)
            if parsed.problem:
                yield (f"allowed value {allowed.value!r} of {relation} is not a valid {datatype}",
                       allowed)
            elif isinstance(parsed.value, Decimal) and (
                    (low is not None and parsed.value < Decimal(low.value))
                    or (high is not None and parsed.value > Decimal(high.value))):
                yield f"allowed value {allowed.value!r} of {relation} lies outside its bounds", allowed
    literal = set(value_relations(sem))
    for d in sem.of(DeclKey.RANGE):
        if d.subject in literal:
            yield f"{d.subject} declares both a class range and a literal value contract", d


_REQUIRES = {"inverse": DeclKey.INVERSE_OF, "symmetric": DeclKey.SYMMETRIC,
             "transitive": DeclKey.TRANSITIVE}


def _materialization(sem: SemanticsDecl) -> Iterator[tuple[str, Declaration]]:
    for d in sem.of(DeclKey.MATERIALIZE):
        required = _REQUIRES[d.value]
        stated = sem.find(required, d.subject)
        partner = required is DeclKey.INVERSE_OF and any(
            x.value == d.subject for x in sem.of(DeclKey.INVERSE_OF))
        if not partner and (stated is None or (required is not DeclKey.INVERSE_OF
                                               and stated.value != "true")):
            yield (f"{d.subject} must materialize {d.value} edges but does not declare "
                   f"{required.value}"), d


def _predicates(sem: SemanticsDecl) -> Iterator[tuple[str, Declaration]]:
    subclass = _true(sem, DeclKey.SUBCLASS_PREDICATE)
    for subject, d in sorted(_true(sem, DeclKey.TYPE_PREDICATE).items()):
        if subject in subclass:
            yield f"{subject} is declared both a subclass and a type predicate", d
