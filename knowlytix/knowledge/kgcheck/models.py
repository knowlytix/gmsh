# SPDX-License-Identifier: Apache-2.0
"""Audit data model: the snapshot an audit reads and the results it produces.

Everything here is immutable, including nested contents: construction copies
and freezes containers, so neither a rule nor the caller's original arguments
can change a snapshot after it exists. A snapshot keeps raw tokens, every raw
row (duplicates included) and where each came from, so a rule can cite exact
artifact locations. Unknown is a legitimate state: a node whose kind no artifact
declares is ``NodeKind.UNKNOWN``, never a guess.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from knowlytix.knowledge.kgcheck.frozen import FrozenJson, freeze_mapping, thaw
from knowlytix.knowledge.kgcheck.identity import finding_id

if TYPE_CHECKING:
    from knowlytix.knowledge.kgcheck.gate import GateDecision, GatePolicy
    from knowlytix.knowledge.kgcheck.reporting.json_report import RunInfo

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]


class NodeKind(StrEnum):
    CLASS = "class"
    INSTANCE = "instance"
    LITERAL = "literal"
    PROPERTY = "property"
    UNKNOWN = "unknown"


class GraphView(StrEnum):
    """Named views of one store: trained, conformant deliverable, quarantine."""

    STORED = "stored"
    CONFORMANT = "conformant"
    QUARANTINED = "quarantined"
    PROPOSAL = "proposal"


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUGGESTION = "suggestion"


class RuleStatus(StrEnum):
    """Outcome of one rule execution. Only PASSED certifies a clean result."""

    PASSED = "passed"
    FINDINGS = "findings"
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    PARTIAL = "partial"
    DISABLED = "disabled"


def _set(obj: object, name: str, value: object) -> None:
    object.__setattr__(obj, name, value)


def _tuple(obj: object, *names: str) -> None:
    for name in names:
        _set(obj, name, tuple(getattr(obj, name)))


@dataclass(frozen=True)
class RawRef:
    """A location in an input artifact: file name and JSON pointer (``/3/1``)."""

    artifact: str
    pointer: str = ""


@dataclass(frozen=True)
class Diagnostic:
    """A problem met while reading an artifact, kept instead of silently dropped."""

    code: str
    message: str
    location: RawRef


@dataclass(frozen=True)
class LiteralValue:
    """A token read as a literal, with the rule that made it one."""

    lexical: str
    basis: str  # e.g. "numeric_token", "iso_date_token", "text_relation_object"


@dataclass(frozen=True)
class Node:
    """A graph term. ``declared_by`` lists the artifacts that name it."""

    node_id: str
    label: str
    namespace: str  # "entity" or "relation"
    kind: NodeKind = NodeKind.UNKNOWN
    aliases: tuple[str, ...] = ()
    declared_by: tuple[str, ...] = ()
    literal: LiteralValue | None = None
    description: str | None = None
    types: tuple[str, ...] = ()  # node ids of the classes it is declared an instance of

    def __post_init__(self) -> None:
        _tuple(self, "aliases", "declared_by", "types")


@dataclass(frozen=True)
class Assertion:
    """One distinct (head, predicate, object) statement, by node ids."""

    assertion_id: str
    head: str
    predicate: str
    object: str


@dataclass(frozen=True)
class AssertionRow:
    """One raw row of a view: which assertion it states and where it was read."""

    assertion_id: str
    raw: RawRef


@dataclass(frozen=True)
class EvidenceLink:
    """One provenance citation for an assertion."""

    assertion_id: str
    kind: str
    section: str
    sentence_index: int
    sentence: str
    span: str
    raw: RawRef
    document: str | None = None


@dataclass(frozen=True)
class SourceText:
    """Retained source text, kept in the snapshot so evidence checks need no store.

    ``sections`` describes how the text is divided: OntoDisc writes one
    ``## <section name>`` heading per extracted section.
    """

    artifact: str
    sha256: str
    text: str
    sections: str = "markdown_h2_per_section"


@dataclass(frozen=True)
class NumericSlot:
    """One exact-numeric-memory entry, keyed by (relation, head) as the store keys it.

    ``lexical`` is the number exactly as written in the artifact.
    """

    relation: str
    head: str
    value: float
    lexical: str
    raw: RawRef


@dataclass(frozen=True)
class NumericConflict:
    """A slot the producer recorded as ambiguous and left out of numeric memory."""

    relation: str
    head: str
    values: tuple[float, ...]
    raw: RawRef

    def __post_init__(self) -> None:
        _tuple(self, "values")


@dataclass(frozen=True)
class PhaseRange:
    """A phase encoder's declared value range."""

    name: str
    v_min: float
    v_max: float
    raw: RawRef


def _groups(groups: Mapping[str, Iterable[str]]) -> Mapping[str, tuple[str, ...]]:
    return MappingProxyType({k: tuple(v) for k, v in groups.items()})


@dataclass(frozen=True)
class OntologyDecl:
    """The ontology as declared: canonical term -> aliases, distinctions, raw schema rows."""

    entities: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    relations: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    distinct: tuple[tuple[str, str], ...] = ()
    schema_rows: tuple[tuple[str, str, str], ...] = ()

    def __post_init__(self) -> None:
        _set(self, "entities", _groups(self.entities))
        _set(self, "relations", _groups(self.relations))
        _set(self, "distinct", tuple(tuple(p) for p in self.distinct))
        _set(self, "schema_rows", tuple(tuple(r) for r in self.schema_rows))


class DeclSource(StrEnum):
    """Who supplied a declaration. Inferred declarations are advisory until accepted."""

    USER = "user"
    IMPORTED = "imported"
    INFERRED = "inferred"


class DeclKey(StrEnum):
    """What a declaration states about its subject.

    Subjects are node ids, ``""`` for a schema-wide setting (language,
    acyclicity), or a namespace (``"entity"``, ``"relation"``) for a naming
    convention (``CASE``, ``SEPARATOR``, ``NUMBER``). Values are node ids for ``TYPE``,
    ``INVERSE_OF``, ``DOMAIN``, ``RANGE`` and ``DISJOINT``, a module name for ``MODULE``,
    ``"true"``/``"false"`` for flags, a number as written for ``MINIMUM``, ``MAXIMUM``,
    ``MIN_COUNT`` and ``MAX_COUNT``, and text otherwise. A ``DISJOINT`` subject is the
    group, ``disjoint:<index>``; ``MATERIALIZE`` values are ``inverse``, ``symmetric``
    or ``transitive``.
    """

    KIND = "kind"
    TYPE = "type"
    DESCRIPTION = "description"
    LANGUAGE = "language"
    STANDALONE = "standalone"
    INVERSE_OF = "inverse_of"
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    TRANSITIVE = "transitive"
    SUBCLASS_PREDICATE = "subclass_predicate"
    TYPE_PREDICATE = "type_predicate"
    ROOT = "root"
    ACYCLIC = "acyclic"
    MODULE = "module"
    CASE = "case"
    SEPARATOR = "separator"
    NUMBER = "number"
    CONVENTION_EXCEPTION = "convention_exception"
    IRREFLEXIVE = "irreflexive"
    DOMAIN = "domain"
    RANGE = "range"
    DATATYPE = "datatype"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    ALLOWED_VALUE = "allowed_value"
    UNIT = "unit"
    PERIOD = "period"
    QUALIFIER = "qualifier"
    MIN_COUNT = "min_count"
    MAX_COUNT = "max_count"
    EQUALITY = "equality"
    COMPLETE = "complete"
    MATERIALIZE = "materialize"
    INHERITED = "inherited"
    DISJOINT = "disjoint"
    CLOSED_TYPING = "closed_typing"


@dataclass(frozen=True)
class Declaration:
    """One statement of the audit schema sidecar, with who supplied it and where."""

    subject: str
    key: DeclKey
    value: str
    source: DeclSource
    raw: RawRef


@dataclass(frozen=True)
class CompetencyQuestion:
    """A question the graph must answer, with its expected answers (stored tokens).

    ``head`` and ``object`` are the given terms; the one left out is the
    answer. With both given the question asks whether the fact is stated, and
    ``absent`` asks that it is not. ``exhaustive`` makes an answer outside
    ``expected`` a failure.
    """

    question_id: str
    relation: str
    source: DeclSource
    raw: RawRef
    question: str = ""
    head: str | None = None
    object: str | None = None
    expected: tuple[str, ...] = ()
    absent: bool = False
    exhaustive: bool = False

    def __post_init__(self) -> None:
        _tuple(self, "expected")


@dataclass(frozen=True)
class SemanticsDecl:
    """Metadata the store does not record, from the audit schema sidecar.

    ``version`` is None when no usable sidecar was read; ``declarations`` is
    then empty, and rules needing this metadata report it unavailable. Inferred
    declarations are kept but take no effect: node annotation, capabilities and
    rules use ``accepted()``, so a guess changes nothing until a person sets its
    source to ``user`` or ``imported``.
    """

    version: str | None = None
    declarations: tuple[Declaration, ...] = ()
    competency: tuple[CompetencyQuestion, ...] = ()

    def __post_init__(self) -> None:
        _tuple(self, "declarations", "competency")

    def accepted(self) -> SemanticsDecl:
        """Only the declarations a person supplied or imported; inferred ones are advisory."""
        return SemanticsDecl(self.version,
                             tuple(d for d in self.declarations if d.source is not DeclSource.INFERRED),
                             tuple(q for q in self.competency if q.source is not DeclSource.INFERRED))

    def of(self, key: DeclKey) -> tuple[Declaration, ...]:
        return tuple(d for d in self.declarations if d.key is key)

    def find(self, key: DeclKey, subject: str) -> Declaration | None:
        """The first declaration of ``key`` about ``subject``, or None."""
        return next((d for d in self.of(key) if d.subject == subject), None)

    def subjects(self, key: DeclKey) -> tuple[str, ...]:
        """Subjects declaring ``key``, sorted, each once."""
        return tuple(sorted({d.subject for d in self.of(key)}))

    def values(self, key: DeclKey, subject: str) -> tuple[str, ...]:
        return tuple(d.value for d in self.of(key) if d.subject == subject)

    def flag(self, key: DeclKey, subject: str) -> bool:
        """True when ``subject`` declares flag ``key`` true."""
        d = self.find(key, subject)
        return d is not None and d.value == "true"

    def inverse_pairs(self) -> tuple[tuple[str, str], ...]:
        """Declared inverse relations as sorted pairs, whichever side declared them."""
        return tuple(sorted({tuple(sorted((d.subject, d.value)))  # type: ignore[misc]
                             for d in self.of(DeclKey.INVERSE_OF)}))


@dataclass(frozen=True)
class ArtifactRecord:
    """An input file as read: presence, size and raw digest (``None`` if unreadable)."""

    name: str
    present: bool
    required: bool
    size: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class Capabilities:
    """What the snapshot can support. A missing capability makes rules unavailable."""

    node_kinds: bool = False
    descriptions: bool = False
    hierarchy: bool = False
    typed_constraints: bool = False
    provenance: bool = False
    source_text: bool = False
    numeric_conflicts_recorded: bool = False
    semantic_verification: bool = False
    relation_properties: bool = False
    naming_conventions: bool = False
    competency_questions: bool = False
    adapter: str = ""
    adapter_version: str = ""


@dataclass(frozen=True)
class AuditSnapshot:
    """Everything an audit reads, deeply immutable. Built by an adapter, never by a rule."""

    snapshot_id: str
    nodes: tuple[Node, ...]
    assertions: tuple[Assertion, ...]
    views: Mapping[GraphView, tuple[AssertionRow, ...]]
    ontology: OntologyDecl
    semantics: SemanticsDecl = field(default_factory=SemanticsDecl)
    evidence: tuple[EvidenceLink, ...] = ()
    sources: tuple[SourceText, ...] = ()
    numeric_slots: tuple[NumericSlot, ...] = ()
    numeric_conflicts: tuple[NumericConflict, ...] = ()
    phase_ranges: tuple[PhaseRange, ...] = ()
    capabilities: Capabilities = field(default_factory=Capabilities)
    artifacts: tuple[ArtifactRecord, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    metadata: Mapping[str, FrozenJson] = field(default_factory=dict)
    # parsed artifact content by file name, for artifact-level rules
    parsed: Mapping[str, FrozenJson] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _tuple(self, "nodes", "assertions", "evidence", "sources", "numeric_slots",
               "numeric_conflicts", "phase_ranges", "artifacts", "diagnostics")
        _set(self, "views", MappingProxyType({GraphView(k): tuple(v) for k, v in self.views.items()}))
        _set(self, "metadata", freeze_mapping(self.metadata))
        _set(self, "parsed", freeze_mapping(self.parsed))

    @cached_property
    def node_index(self) -> Mapping[str, Node]:
        return MappingProxyType({n.node_id: n for n in self.nodes})

    @cached_property
    def assertion_index(self) -> Mapping[str, Assertion]:
        return MappingProxyType({a.assertion_id: a for a in self.assertions})

    def view(self, name: GraphView) -> tuple[AssertionRow, ...]:
        """Rows of one view, in artifact order; empty when the view is absent."""
        return self.views.get(name, ())

    def has_view(self, name: GraphView) -> bool:
        return name in self.views


@dataclass(frozen=True)
class Finding:
    """One detected issue with a reproducible witness. Never mutates the input."""

    rule_id: str
    rule_version: str
    category: str
    severity: Severity
    message: str
    basis: str
    view: GraphView | None = None
    affected: tuple[str, ...] = ()
    locations: tuple[RawRef, ...] = ()
    witness: Mapping[str, FrozenJson] = field(default_factory=dict)
    suggestion: str | None = None
    heuristic_score: float | None = None  # uncalibrated unless validated
    discriminator: str = ""  # separates distinct issues on the same target
    # the configured suppression that matched (reason, rule version); never part of the id
    suppression: Mapping[str, FrozenJson] | None = None

    def __post_init__(self) -> None:
        _tuple(self, "affected", "locations")
        _set(self, "witness", freeze_mapping(self.witness))
        if self.suppression is not None:
            _set(self, "suppression", freeze_mapping(self.suppression))
        if self.heuristic_score is not None and not math.isfinite(self.heuristic_score):
            raise ValueError("heuristic_score must be finite")
        json.dumps(thaw(self.witness), allow_nan=False)  # strict JSON, or raise here

    @property
    def finding_id(self) -> str:
        """Stable across reruns: see :func:`knowlytix.knowledge.kgcheck.identity.finding_id`."""
        return finding_id(self.rule_id, self.rule_version, self.view.value if self.view else None,
                          self.affected, tuple(f"{r.artifact}#{r.pointer}" for r in self.locations),
                          self.discriminator)


class RuleResultError(ValueError):
    """A rule result whose status contradicts its content."""


def _is_count(value: object) -> bool:
    return value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 0)


@dataclass(frozen=True)
class RuleResult:
    """One rule execution. Construction enforces what each status may claim."""

    rule_id: str
    rule_version: str
    status: RuleStatus
    findings: tuple[Finding, ...] = ()
    eligible: int | None = None
    tested: int | None = None
    missing_prerequisites: tuple[str, ...] = ()
    error: str | None = None
    runtime_s: float = 0.0
    details_truncated: bool = False
    # measured quantities with their denominators, e.g. {"coverage": {"numerator": 7,
    # "denominator": 9, "value": 0.78}}; strict JSON like a finding witness
    metrics: Mapping[str, FrozenJson] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _tuple(self, "findings", "missing_prerequisites")
        _set(self, "metrics", freeze_mapping(self.metrics))
        json.dumps(thaw(self.metrics), allow_nan=False)
        problem = _common_problem(self) or _STATUS_CHECKS[self.status](self)
        if problem:
            raise RuleResultError(f"{self.status.value}: {problem}")


def _common_problem(r: RuleResult) -> str | None:
    if not (_is_count(r.eligible) and _is_count(r.tested)):
        return "eligible and tested must be nonnegative integers"
    if r.tested is not None and r.eligible is not None and r.tested > r.eligible:
        return "tested exceeds eligible"
    if any(f.rule_id != r.rule_id for f in r.findings):
        return "a finding belongs to a different rule"
    if r.status in _DEFINITE and (r.error or r.missing_prerequisites):
        return "a definite outcome carries no error and no missing prerequisites"
    return None


def _check_passed(r: RuleResult) -> str | None:
    if r.findings:
        return "a passed rule has no findings"
    if r.eligible is None or r.eligible < 1 or r.tested != r.eligible:
        return "passed needs at least one eligible target, all of them tested"
    return None


def _check_findings(r: RuleResult) -> str | None:
    if not r.findings:
        return "needs at least one finding"
    if r.tested is None or r.tested != r.eligible:
        return "findings means every eligible target was tested; use partial otherwise"
    return None


def _check_not_applicable(r: RuleResult) -> str | None:
    if r.findings or r.eligible != 0:
        return "means zero eligible targets and no findings"
    return None


def _check_unavailable(r: RuleResult) -> str | None:
    if r.findings or not r.missing_prerequisites:
        return "needs missing prerequisites and carries no findings"
    return None


def _check_error(r: RuleResult) -> str | None:
    return None if r.error else "needs an error message"


def _check_partial(r: RuleResult) -> str | None:
    if r.tested is not None and r.eligible is not None and r.tested >= r.eligible \
            and not r.details_truncated and not r.error:
        return "a complete, untruncated run is passed or findings, not partial"
    return None


def _check_disabled(r: RuleResult) -> str | None:
    return "carries no findings" if r.findings else None


_DEFINITE = frozenset({RuleStatus.PASSED, RuleStatus.FINDINGS, RuleStatus.NOT_APPLICABLE})
_STATUS_CHECKS = {
    RuleStatus.PASSED: _check_passed,
    RuleStatus.FINDINGS: _check_findings,
    RuleStatus.NOT_APPLICABLE: _check_not_applicable,
    RuleStatus.UNAVAILABLE: _check_unavailable,
    RuleStatus.ERROR: _check_error,
    RuleStatus.PARTIAL: _check_partial,
    RuleStatus.DISABLED: _check_disabled,
}


@dataclass(frozen=True)
class FindingCounts:
    """Exact finding totals of a report; ``listed`` is how many have details."""

    total: int
    listed: int
    by_severity: dict[str, int]
    by_category: dict[str, int]


def _exact_counts(r: RuleResult) -> tuple[int, Mapping[str, int], Mapping[str, int]]:
    m = r.metrics
    if isinstance(m.get("findings_total"), int) and isinstance(m.get("findings_by_category"), Mapping):
        return m["findings_total"], m.get("findings_by_severity", {}), m["findings_by_category"]
    sev: dict[str, int] = {}
    cat: dict[str, int] = {}
    for f in r.findings:
        sev[f.severity.value] = sev.get(f.severity.value, 0) + 1
        cat[f.category] = cat.get(f.category, 0) + 1
    return len(r.findings), sev, cat


def _merge(into: dict[str, int], counts: Mapping[str, int]) -> None:
    for key, n in counts.items():
        into[key] = into.get(key, 0) + int(n)


class DuplicateFindingError(ValueError):
    """Two findings of one report share an id; the rule must set a discriminator."""


@dataclass(frozen=True)
class AuditReport:
    """Rule results for one snapshot. Serialize with ``kgcheck.reporting``.

    Results are kept in rule-id order, so a report does not depend on the order
    in which rules ran. Finding ids must be unique within the report.
    """

    snapshot_id: str
    contract_version: str
    results: tuple[RuleResult, ...]
    diagnostics: tuple[Diagnostic, ...] = ()
    capabilities: Capabilities = field(default_factory=Capabilities)
    artifacts: tuple[ArtifactRecord, ...] = ()
    profile: str | None = None
    views: tuple[GraphView, ...] = ()

    def __post_init__(self) -> None:
        _set(self, "results", tuple(sorted(self.results, key=lambda r: r.rule_id)))
        _tuple(self, "diagnostics", "artifacts", "views")
        ids = [f.finding_id for f in self.findings()]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise DuplicateFindingError(f"duplicate finding ids: {dupes}")

    @classmethod
    def for_snapshot(cls, snapshot: AuditSnapshot, results: Iterable[RuleResult], *,
                     profile: str | None = None,
                     views: Iterable[GraphView] = ()) -> AuditReport:
        """A report carrying the snapshot's identity, inputs and read diagnostics."""
        from knowlytix.knowledge.kgcheck.contract import contract_version
        return cls(snapshot.snapshot_id, contract_version(), tuple(results), snapshot.diagnostics,
                   snapshot.capabilities, snapshot.artifacts, profile, tuple(views))

    def findings(self) -> tuple[Finding, ...]:
        """All findings, by rule id then finding id."""
        return tuple(f for r in self.results
                     for f in sorted(r.findings, key=lambda f: f.finding_id))

    def by_status(self) -> dict[RuleStatus, tuple[str, ...]]:
        """Rule ids grouped by status, statuses in enum order."""
        return {s: tuple(r.rule_id for r in self.results if r.status is s)
                for s in RuleStatus if any(r.status is s for r in self.results)}

    def finding_counts(self) -> FindingCounts:
        """Totals over every detected finding, including those truncated from the details.

        A rule result's metrics carry its exact counts (the engine records them
        before truncation); a result without them counts its listed findings.
        """
        total, listed = 0, 0
        severity: dict[str, int] = {}
        category: dict[str, int] = {}
        for r in self.results:
            exact = _exact_counts(r)
            total += exact[0]
            listed += len(r.findings)
            _merge(severity, exact[1])
            _merge(category, exact[2])
        return FindingCounts(total, listed, dict(sorted(severity.items())),
                             dict(sorted(category.items())))

    def is_complete(self) -> bool:
        """True when every enabled rule ran to a definite outcome."""
        open_ = {RuleStatus.UNAVAILABLE, RuleStatus.ERROR, RuleStatus.PARTIAL}
        return not any(r.status in open_ for r in self.results)

    # Conveniences over kgcheck.gate and kgcheck.reporting (imported lazily: they import this module).
    def evaluate_gate(self, policy: GatePolicy | None = None) -> GateDecision:
        """The gate decision under ``policy`` (default: fail on errors, profile's required rules)."""
        from knowlytix.knowledge.kgcheck.gate import evaluate_gate
        return evaluate_gate(self, policy)

    def write_json(self, path: str | Path, run: RunInfo | None = None,
                   gate: GateDecision | None = None) -> Path:
        from knowlytix.knowledge.kgcheck.reporting.json_report import write_json
        return write_json(self, path, run, gate)

    def write_markdown(self, path: str | Path, gate: GateDecision | None = None, *,
                       excerpts: bool = True) -> Path:
        from knowlytix.knowledge.kgcheck.reporting.markdown_report import write_markdown
        return write_markdown(self, path, gate, excerpts=excerpts)
