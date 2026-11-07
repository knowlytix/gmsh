# SPDX-License-Identifier: Apache-2.0
"""CON008: consistency and unsatisfiable classes from an OWL 2 DL reasoner (HermiT).

The view is written as an OWL ontology in N-Triples: declared and asserted
classes, subclass edges under the declared subclass predicates, disjoint class
groups, the nodes' declared and asserted types, entity-to-entity assertions as
object property assertions, the declared symmetric, asymmetric, transitive and
irreflexive characteristics and inverses, and ``owl:differentFrom`` for the
ontology's distinct pairs. Domain and range are not sent: in OWL they infer
types, which is not what the audit schema declares them for. Facts with literal
objects are not sent either, so datatype reasoning is out of scope.

OWL is open-world and has no unique-name assumption, so an untyped node or a
missing value is never a contradiction here, and two names can denote one
individual unless declared distinct. OWL 2 DL forbids asymmetry and
irreflexivity on a transitive property; such a characteristic is not sent and
the omission is reported. HermiT runs in a Java subprocess under a time limit;
a timeout leaves consistency unknown and the result partial. The jar is the
one bundled with owlready2 (the ``kgcheck-owl`` extra); without it or without
Java the rule is unavailable.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from importlib.util import find_spec
from pathlib import Path
from urllib.parse import quote, unquote

from knowlytix.knowledge.kgcheck.models import DeclKey, Finding, SemanticsDecl, Severity
from knowlytix.knowledge.kgcheck.registry import Outcome, RuleContext, RuleSpec, builtin
from knowlytix.knowledge.kgcheck.rules._common import is_literal, label
from knowlytix.knowledge.kgcheck.taxonomy import HIERARCHY_VIEWS, accepted_predicates
from knowlytix.knowledge.kgcheck.typed import typing

BASE = "urn:kgcheck:"
_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_RDFS = "http://www.w3.org/2000/01/rdf-schema#"
_OWL = "http://www.w3.org/2002/07/owl#"
_CHARACTERISTICS = {DeclKey.SYMMETRIC: "SymmetricProperty", DeclKey.ASYMMETRIC: "AsymmetricProperty",
                    DeclKey.TRANSITIVE: "TransitiveProperty",
                    DeclKey.IRREFLEXIVE: "IrreflexiveProperty"}
_NOT_FOR_TRANSITIVE = (DeclKey.ASYMMETRIC, DeclKey.IRREFLEXIVE)
_NOTHING = re.compile(r"^EquivalentClasses\( (.*) \)$", re.M)
_WITNESS = 50


def _iri(node_id: str) -> str:
    namespace, _, token = node_id.partition(":")
    return f"<{BASE}{namespace}:{quote(token, safe='')}>"


def _node(iri: str, native: bool = False) -> str | None:
    text = iri.strip("<>")
    if native:
        return f"entity:{text}"
    return unquote(text[len(BASE):]) if text.startswith(BASE) else None


@builtin(RuleSpec(
    "CON008", "2", "constraint", "OWL consistency and unsatisfiable classes", Severity.ERROR,
    phase="E", views=HIERARCHY_VIEWS, cacheable=False,
    parameters={"timeout_s": 60.0, "java_memory_mb": 2000},
    parameter_bounds={"timeout_s": (1.0, 3600.0), "java_memory_mb": (256, 65536)},
    references=("https://www.w3.org/TR/owl2-primer/", "http://www.hermit-reasoner.com/"),
    basis="HermiT (OWL 2 DL, open world, no unique-name assumption) on the view written as an "
          "OWL ontology: classes, declared subclass edges, disjoint classes, types, object "
          "property assertions, declared property characteristics and inverses and distinct "
          "pairs; domain, range and literal facts are not sent. An RDF input is sent as read, "
          "with the declared characteristics added. A timeout is unknown, not consistent"))
def con008(ctx: RuleContext) -> Outcome:
    jar, java = _hermit(), shutil.which("java")
    if jar is None or java is None:
        return Outcome.unavailable("backend:hermit" if jar is None else "backend:java")
    ontology = _Ontology(ctx)
    if not ontology.has_axioms():
        return Outcome.unavailable("owl_axioms")
    lines = ontology.ntriples()
    findings = list(ontology.omissions())
    eligible = 1 + len(ontology.classes)
    outcome, output = _run(java, jar, lines, ctx)
    metrics = {"reasoner": "HermiT (owlready2)", "classes": len(ontology.classes),
               "individuals": len(ontology.individuals), "axioms": len(lines),
               "literal_facts_not_sent": ontology.literal_facts,
               "timeout_s": ctx.params["timeout_s"], "result": outcome}
    if outcome == "timeout":
        return Outcome(eligible=eligible, tested=0, findings=tuple(findings), metrics=metrics,
                       partial=f"HermiT did not finish within {ctx.params['timeout_s']} s; "
                               "consistency is unknown")
    if outcome == "inconsistent":
        findings.append(_inconsistent(ctx))
    else:
        findings += [_unsatisfiable(ctx, c) for c in _unsatisfiable_classes(output, ontology.native)]
    return Outcome(eligible=eligible, findings=tuple(findings), metrics=metrics)


def _hermit() -> Path | None:
    spec = find_spec("owlready2")
    if spec is None or not spec.submodule_search_locations:
        return None
    jar = Path(next(iter(spec.submodule_search_locations))) / "hermit" / "HermiT.jar"
    return jar if jar.is_file() else None


def _run(java: str, jar: Path, lines: list[str], ctx: RuleContext) -> tuple[str, str]:
    """("consistent" | "inconsistent" | "timeout", reasoner output). Raises on a Java error."""
    classpath = os.pathsep.join([str(jar.parent), str(jar)])
    with tempfile.TemporaryDirectory(prefix="kgcheck-owl-") as tmp:
        source = Path(tmp) / "view.nt"
        source.write_text("".join(lines), encoding="utf-8")
        command = [java, f"-Xmx{int(ctx.params['java_memory_mb'])}M", "-cp", classpath,
                   "org.semanticweb.HermiT.cli.CommandLine", "-c", "-O", "-D", "-I",
                   source.as_uri()]
        try:
            done = subprocess.run(command, capture_output=True, text=True,
                                  timeout=float(ctx.params["timeout_s"]), check=False)
        except subprocess.TimeoutExpired:
            return "timeout", ""
    output = done.stdout + done.stderr
    if done.returncode == 0:
        return "consistent", done.stdout
    if "InconsistentOntologyException" in output or "Inconsistent ontology" in output:
        return "inconsistent", output
    cause = next((line.strip() for line in output.splitlines() if "Exception" in line),
                 output.strip()[-400:])
    raise RuntimeError(f"HermiT failed (exit {done.returncode}): {cause[:400]}")


def _unsatisfiable_classes(output: str, native: bool = False) -> list[str]:
    out: set[str] = set()
    for m in _NOTHING.finditer(output):
        iris = m[1].split()
        if f"<{_OWL}Nothing>" in iris:
            out |= {n for i in iris if i != f"<{_OWL}Nothing>" and (n := _node(i, native)) is not None}
    return sorted(out)


def _inconsistent(ctx: RuleContext) -> Finding:
    return ctx.finding(
        "the reasoner finds the view inconsistent: no interpretation satisfies its classes, "
        "types and property characteristics together", discriminator="inconsistent",
        witness={"reasoner": "HermiT", "related_rules": ["CON004", "CON005", "STR006"]},
        suggestion="CON004 (disjoint types), CON005 (irreflexive) and STR006 (asymmetric) "
                   "locate the usual causes; the reasoner gives no justification")


def _unsatisfiable(ctx: RuleContext, cls: str) -> Finding:
    return ctx.finding(f"class {label(ctx.snapshot, cls)!r} is unsatisfiable: it can have no "
                       "instance", severity=Severity.WARNING, affected=(cls,),
                       discriminator="unsatisfiable",
                       suggestion="its superclasses or restrictions contradict each other; "
                                  "CON004 shows disjoint superclasses")


class _Ontology:
    """The OWL axioms of one view, by node id."""

    def __init__(self, ctx: RuleContext) -> None:
        snap = ctx.snapshot
        self.ctx, self.sem = ctx, snap.semantics.accepted()
        self.types = typing(snap, ctx.view)
        self.typing_predicates = set(accepted_predicates(snap, DeclKey.TYPE_PREDICATE)) | set(
            accepted_predicates(snap, DeclKey.SUBCLASS_PREDICATE))
        self.groups = _groups(self.sem)
        self.classes = sorted({*self.types.parents, *(p for ps in self.types.parents.values() for p in ps),
                               *(c for cs in self.types.direct.values() for c in cs),
                               *(c for g in self.groups.values() for c in g),
                               *(n.node_id for n in snap.nodes if n.kind.value == "class")})
        self.individuals = self.types.typed()
        self.literal_facts = 0
        self.native = snap.capabilities.adapter == "rdf"  # tokens are the graph's own IRIs
        self.transitive = _non_simple(self.sem)

    def iri(self, node_id: str) -> str:
        if not self.native:
            return _iri(node_id)
        token = node_id.partition(":")[2]
        return token if token.startswith("_:") else f"<{token}>"

    def has_axioms(self) -> bool:
        """Anything a reasoner could find contradictory: classes, or property semantics."""
        if self.native:
            return isinstance(self.ctx.snapshot.parsed.get("graph.nt"), str)
        return bool(self.classes) or any(
            d.value == "true" for k in _CHARACTERISTICS for d in self.sem.of(k)) \
            or bool(self.sem.inverse_pairs()) or bool(self.ctx.snapshot.ontology.distinct)

    def ntriples(self) -> list[str]:
        if self.native:  # an RDF input is sent as read, with the declared semantics added
            graph = str(self.ctx.snapshot.parsed["graph.nt"])
            return [*(line + "\n" for line in graph.splitlines() if line.strip()),
                    *self._properties()]
        lines = [f"{self.iri(c)} <{_RDF}type> <{_OWL}Class> .\n" for c in self.classes]
        lines += [f"{self.iri(c)} <{_RDFS}subClassOf> {self.iri(p)} .\n"
                  for c, ps in sorted(self.types.parents.items()) for p in ps]
        for members in self.groups.values():
            lines += [f"{self.iri(a)} <{_OWL}disjointWith> {self.iri(b)} .\n"
                      for a in members for b in members if a < b]
        for node in self.individuals:
            lines.append(f"{self.iri(node)} <{_RDF}type> <{_OWL}NamedIndividual> .\n")
            lines += [f"{self.iri(node)} <{_RDF}type> {self.iri(c)} .\n" for c in self.types.direct[node]]
        lines += list(self._assertions()) + list(self._properties()) + list(self._distinct())
        return lines

    def _assertions(self) -> Iterator[str]:
        snap, declared = self.ctx.snapshot, set()
        for row in snap.view(self.ctx.view):
            a = snap.assertion_index[row.assertion_id]
            if a.predicate in self.typing_predicates:
                continue
            if is_literal(snap, a.object):
                self.literal_facts += 1
                continue
            if a.predicate not in declared:
                declared.add(a.predicate)
                yield f"{self.iri(a.predicate)} <{_RDF}type> <{_OWL}ObjectProperty> .\n"
            yield f"{self.iri(a.head)} {self.iri(a.predicate)} {self.iri(a.object)} .\n"

    def _properties(self) -> Iterator[str]:
        for key, owl in _CHARACTERISTICS.items():
            for d in self.sem.of(key):
                if d.value == "true" and not (key in _NOT_FOR_TRANSITIVE and d.subject in self.transitive):
                    yield f"{self.iri(d.subject)} <{_RDF}type> <{_OWL}ObjectProperty> .\n"
                    yield f"{self.iri(d.subject)} <{_RDF}type> <{_OWL}{owl}> .\n"
        for a, b in self.sem.inverse_pairs():
            yield f"{self.iri(a)} <{_OWL}inverseOf> {self.iri(b)} .\n"

    def _distinct(self) -> Iterator[str]:
        for a, b in self.ctx.snapshot.ontology.distinct:
            yield f"{self.iri('entity:' + a)} <{_OWL}differentFrom> {self.iri('entity:' + b)} .\n"

    def omissions(self) -> Iterator[Finding]:
        for key in _NOT_FOR_TRANSITIVE:
            for d in self.sem.of(key):
                if d.value == "true" and d.subject in self.transitive:
                    yield self.ctx.finding(
                        f"{label(self.ctx.snapshot, d.subject)!r} is transitive, or the inverse of "
                        f"a transitive relation, and declared {key.value}; OWL 2 DL forbids that, "
                        f"so {key.value} was not sent to the reasoner", severity=Severity.INFO, affected=(d.subject,),
                        locations=(d.raw,), discriminator=f"not_simple:{key.value}",
                        suggestion="CON005 and STR006 still check the declaration natively")


def _non_simple(sem: SemanticsDecl) -> set[str]:
    """Properties OWL 2 DL treats as non-simple: the transitive ones and their inverses."""
    out = {d.subject for d in sem.of(DeclKey.TRANSITIVE) if d.value == "true"}
    pairs = sem.inverse_pairs()
    while True:
        more = {b for a, b in pairs if a in out} | {a for a, b in pairs if b in out}
        if more <= out:
            return out
        out |= more


def _groups(sem: SemanticsDecl) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for d in sem.of(DeclKey.DISJOINT):
        if d.value not in groups.setdefault(d.subject, []):
            groups[d.subject].append(d.value)
    return groups
