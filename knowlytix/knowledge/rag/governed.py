# SPDX-License-Identifier: Apache-2.0
"""Governed retrieval --- policy enforcement around triple-mediated RAG.

The retrieval channel is the highest-risk read surface an agent has: a query that
binds broadly can surface records the workflow was never authorized to see. This
module makes retrieval a controlled system operation rather than an autonomous
search. The agent does not search the store; it submits an intent-scoped request
to a :class:`GovernedRetriever`, which binds the query, retrieves grounded facts
through the GMS pipeline, and enforces a :class:`RetrievalContract` before, during
and after retrieval:

* **before** -- the bound head's sensitivity zone is checked against the contract;
  a query that binds to a blocked zone is denied, not searched.
* **during** -- retrieval reuses :class:`~knowlytix.knowledge.rag.pipeline.RagPipeline`
  (head-scoped, gate-calibrated, provenance-linked); the accept, relevance and
  self-verification gates already decide what is grounded, and the pipeline
  abstains rather than guess.
* **after** -- every retrieved fact's zone is re-checked, restricted fields are
  redacted, relations the purpose does not need are dropped (least-context), an
  optional disclosure guard tests whether the answer exposes a protected attribute
  no admitted fact supports, and an immutable audit record is emitted.

Nothing here re-implements retrieval or the gates. It composes ``RagPipeline``,
:class:`~knowlytix.knowledge.rag.value_polarity.ValuePolarityChecker` and
:class:`~knowlytix.knowledge.geode.provenance.ProvenanceLedger`. The sensitivity
*zone* of a record is metadata the GMS store does not itself carry, so it is
supplied by a :class:`SensitivityMap`: GMS enforces the contract, it does not
originate the labels. An entity with no label is treated as maximally sensitive
(fail-closed), so an unclassified record is withheld rather than leaked.

Usage::

    from knowlytix.knowledge.rag import RagPipeline
    from knowlytix.knowledge.rag.governed import (
        GovernedRetriever, RetrievalContract, SensitivityMap)

    pipe = RagPipeline.from_store(store, rag_config)
    smap = SensitivityMap.load("data/gms_governed_store/sensitivity_map.json")
    contract = RetrievalContract.load("data/complaint_contract.json")
    gov = GovernedRetriever(pipe, contract, smap,
                            audit_sink=lambda rec: log.write(json.dumps(rec) + "\\n"))
    result = gov.retrieve("overdraft fee dispute policy for checking account")
    result.decision   # "accept" | "deny" | "abstain"
    result.answer     # redacted, contract-scoped
    result.audit      # immutable RetrievalAuditRecord
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Callable, Sequence

if TYPE_CHECKING:
    from knowlytix.knowledge.geode.provenance import ProvenanceLedger
    from knowlytix.knowledge.rag.pipeline import RagAnswer, RagPipeline
    from knowlytix.knowledge.rag.value_polarity import ValuePolarityChecker

__all__ = [
    "SENSITIVITY_ORDER",
    "SensitivityMap",
    "RetrievalContract",
    "ProtectedProbe",
    "PolarityDisclosureGuard",
    "ClassifierDisclosureGuard",
    "DEFAULT_STANCE_VALUES",
    "GovernedFact",
    "DeniedFact",
    "DisclosureFinding",
    "RetrievalAuditRecord",
    "GovernedResult",
    "GovernedRetriever",
    "REDACTORS",
]


# ---------------------------------------------------------------------------
# Sensitivity ordering and field redactors
# ---------------------------------------------------------------------------

SENSITIVITY_ORDER: tuple[str, ...] = (
    "public",
    "internal",
    "confidential",
    "restricted",
    "regulated",
)
"""Total order on sensitivity levels, least to most sensitive. A contract's
``max_sensitivity`` admits every level at or below it in this order."""


def _sensitivity_rank(level: str) -> int:
    """Rank of *level* in :data:`SENSITIVITY_ORDER`; unknown levels rank as the
    most sensitive (fail-closed) so an unrecognized label is never treated as
    less sensitive than it might be."""
    try:
        return SENSITIVITY_ORDER.index(level)
    except ValueError:
        return len(SENSITIVITY_ORDER)


def _last4(value: str) -> str:
    """Trailing four digits of *value*, or ``****`` if it has fewer than four."""
    digits = re.sub(r"\D", "", value)
    return "****" + digits[-4:] if len(digits) >= 4 else "****"


def _year_only(value: str) -> str:
    """First four-digit run in *value* (a year), coarsening a full date."""
    m = re.search(r"\d{4}", value)
    return m.group(0) if m else "[REDACTED_DATE]"


def _region_only(value: str) -> str:
    """Coarsest geographic token of an address: the last purely-alphabetic token
    (a state or city), skipping numeric tokens such as a ZIP or apartment number
    that would disclose finer location than a region. Falls back to a placeholder
    when no alphabetic token is present."""
    parts = [p for p in re.split(r"[ _,]+", value.strip()) if p.isalpha()]
    return parts[-1] if parts else "[REDACTED_LOCATION]"


REDACTORS: dict[str, Callable[[str], str]] = {
    "mask": lambda v: "[REDACTED]",
    "mask_ssn": lambda v: "[REDACTED_SSN]",
    "last4": _last4,
    "year_only": _year_only,
    "region_only": _region_only,
}
"""Named field-level redactors. A contract's ``redact_relations`` maps a relation
to one of these names; the retriever applies it to the fact's tail before the
fact reaches the agent. The special mode ``"drop"`` (handled in
:class:`GovernedRetriever`, not here) suppresses the fact entirely."""


# ---------------------------------------------------------------------------
# Sensitivity metadata
# ---------------------------------------------------------------------------

@dataclass
class SensitivityMap:
    """Maps a store entity to a sensitivity *zone* and a zone to a level.

    The GMS store carries no classification field, so the zone of a record is
    supplied here (per-document tagging keyed on the entity name). Enforcement is
    the contract's job; this object only reports labels.

    Parameters
    ----------
    entity_zone : dict[str, str]
        Entity name -> zone name (e.g. ``"case_c12345" -> "case"``).
    zone_sensitivity : dict[str, str]
        Zone name -> sensitivity level in :data:`SENSITIVITY_ORDER`.
    default_zone : str
        Zone assigned to an entity absent from *entity_zone*. Defaults to
        ``"unclassified"``.
    default_sensitivity : str
        Sensitivity of the default zone. Defaults to ``"regulated"`` (the most
        sensitive) so unlabeled records fail closed.
    """

    entity_zone: dict[str, str] = field(default_factory=dict)
    zone_sensitivity: dict[str, str] = field(default_factory=dict)
    default_zone: str = "unclassified"
    default_sensitivity: str = "regulated"

    def zone_of(self, entity: str) -> str:
        """Zone of *entity*, or :attr:`default_zone` if unlabeled."""
        return self.entity_zone.get(entity, self.default_zone)

    def sensitivity_of(self, entity: str) -> str:
        """Sensitivity level of *entity* via its zone (fail-closed)."""
        zone = self.zone_of(entity)
        if zone == self.default_zone and entity not in self.entity_zone:
            return self.default_sensitivity
        return self.zone_sensitivity.get(zone, self.default_sensitivity)

    def to_dict(self) -> dict:
        return {
            "entity_zone": self.entity_zone,
            "zone_sensitivity": self.zone_sensitivity,
            "default_zone": self.default_zone,
            "default_sensitivity": self.default_sensitivity,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SensitivityMap":
        return cls(
            entity_zone=dict(d.get("entity_zone", {})),
            zone_sensitivity=dict(d.get("zone_sensitivity", {})),
            default_zone=d.get("default_zone", "unclassified"),
            default_sensitivity=d.get("default_sensitivity", "regulated"),
        )

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "SensitivityMap":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


# ---------------------------------------------------------------------------
# Retrieval contract
# ---------------------------------------------------------------------------

@dataclass
class RetrievalContract:
    """The explicit, per-workflow envelope a retrieval must stay inside.

    A retrieval is admissible only where every dimension passes: the bound zone
    is allowed, the record's sensitivity is at or below the ceiling, any flag the
    zone requires is granted, and the relation is one the business purpose needs.
    Restricted fields that survive are redacted.

    Parameters
    ----------
    workflow : str
        Identifier of the workflow this contract governs.
    business_purpose : str
        Human-readable purpose, recorded in every audit entry.
    allowed_zones : frozenset[str]
        Allowlist of zones (from a :class:`SensitivityMap`) the workflow may read.
        A zone not listed is denied; the allowlist, not a blocklist, is the
        control.
    needed_relations : frozenset[str] or None
        Least-context allowlist of relations the purpose requires. When set, a
        retrieved fact whose relation is absent is dropped as not-needed. ``None``
        disables least-context filtering (all relations pass).
    redact_relations : dict[str, str]
        Relation -> redactor name (see :data:`REDACTORS`) or ``"drop"``. Applied
        to admitted facts before they reach the agent.
    max_sensitivity : str
        Highest sensitivity level admitted, inclusive (see
        :data:`SENSITIVITY_ORDER`).
    zone_flags : dict[str, str]
        Zone -> grant flag the caller must hold to read that zone (e.g.
        ``{"aml": "aml_authorized"}``). A zone with no entry needs no flag.
    """

    workflow: str = ""
    business_purpose: str = ""
    allowed_zones: frozenset[str] = field(default_factory=frozenset)
    needed_relations: frozenset[str] | None = None
    redact_relations: dict[str, str] = field(default_factory=dict)
    max_sensitivity: str = "confidential"
    zone_flags: dict[str, str] = field(default_factory=dict)

    def permits_zone(
        self,
        zone: str,
        sensitivity: str,
        granted_flags: frozenset[str],
    ) -> tuple[bool, str]:
        """Whether a record in *zone* at *sensitivity* may be read.

        Returns ``(True, "")`` when admissible, else ``(False, reason)`` naming
        the first failing dimension: zone allowlist, sensitivity ceiling, or a
        missing zone flag.
        """
        if zone not in self.allowed_zones:
            return False, f"zone '{zone}' not in workflow allowlist"
        if _sensitivity_rank(sensitivity) > _sensitivity_rank(self.max_sensitivity):
            return False, (
                f"sensitivity '{sensitivity}' exceeds ceiling "
                f"'{self.max_sensitivity}'"
            )
        need = self.zone_flags.get(zone)
        if need is not None and need not in granted_flags:
            return False, f"zone '{zone}' requires flag '{need}' (not granted)"
        return True, ""

    def needs_relation(self, relation: str) -> bool:
        """Whether *relation* is within the least-context allowlist (always True
        when :attr:`needed_relations` is ``None``)."""
        return self.needed_relations is None or relation in self.needed_relations

    def to_dict(self) -> dict:
        return {
            "workflow": self.workflow,
            "business_purpose": self.business_purpose,
            "allowed_zones": sorted(self.allowed_zones),
            "needed_relations": (
                None if self.needed_relations is None
                else sorted(self.needed_relations)
            ),
            "redact_relations": self.redact_relations,
            "max_sensitivity": self.max_sensitivity,
            "zone_flags": self.zone_flags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RetrievalContract":
        nr = d.get("needed_relations")
        return cls(
            workflow=d.get("workflow", ""),
            business_purpose=d.get("business_purpose", ""),
            allowed_zones=frozenset(d.get("allowed_zones", [])),
            needed_relations=None if nr is None else frozenset(nr),
            redact_relations=dict(d.get("redact_relations", {})),
            max_sensitivity=d.get("max_sensitivity", "confidential"),
            zone_flags=dict(d.get("zone_flags", {})),
        )

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "RetrievalContract":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


# ---------------------------------------------------------------------------
# Disclosure guard (indirect-leakage backstop)
# ---------------------------------------------------------------------------

@dataclass
class ProtectedProbe:
    """A protected attribute the answer must not disclose or infer.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. ``"financial_distress"``).
    head : str
        The entity the attribute would attach to (e.g. ``"customer_alice"``).
    relation : str
        The store relation that would carry it (e.g.
        ``"has_days_balance_negative"``); its presence in the admitted set is a
        direct disclosure.
    pole : str
        A short value phrase naming the sensitive inference (e.g.
        ``"financial distress"``), tested against the answer in u-space.
    stored : str
        A neutral reference value for the polarity check (e.g. ``"unknown"``).
    """

    name: str
    head: str
    relation: str
    pole: str
    stored: str = "unknown"


@dataclass
class DisclosureFinding:
    """A protected attribute the disclosure guard judged exposed."""

    probe: str
    kind: str          # "direct" (in admitted set) | "inferred" (u-space)
    detail: str
    tension: float | None = None


class PolarityDisclosureGuard:
    """Backstop for indirect leakage, built on :class:`ValuePolarityChecker`.

    Two signals per probe. **Direct**: the protected relation is present in the
    admitted fact set -- the attribute reached the agent literally, which
    least-context should have prevented. **Inferred**: the u-space polarity
    tension between the answer text and the protected pole falls at or below the
    entailment cut, i.e. the answer states the sensitive inference even though no
    admitted fact carries it. The inferred signal is coarse (sentence-level u-space
    tension); it is a secondary net, with least-context the primary control.

    Parameters
    ----------
    checker : ValuePolarityChecker
        Supplies the polarity-tuned u-encoder and, when calibrated, the
        entailment cut ``tau_ent``. When the u-space channel is degenerate
        (``checker.cuts is None``) only the direct signal fires.
    probes : Sequence[ProtectedProbe]
        The protected attributes to test.
    """

    def __init__(
        self,
        checker: "ValuePolarityChecker",
        probes: Sequence[ProtectedProbe],
    ):
        self.checker = checker
        self.probes = list(probes)

    def scan(
        self,
        answer: str,
        admitted_facts: "Sequence[tuple[str, str, str]]",
    ) -> list[DisclosureFinding]:
        """Findings for *answer* given the facts admitted this retrieval.

        *admitted_facts* is a sequence of ``(head, relation, tail)``. The direct
        signal fires when a protected relation is in the admitted set; the inferred
        signal tests the answer prose against each protected pole in u-space.
        """
        admitted_relations = {r for _h, r, _t in admitted_facts}
        findings: list[DisclosureFinding] = []
        cuts = getattr(self.checker, "cuts", None)
        for p in self.probes:
            if p.relation in admitted_relations:
                findings.append(DisclosureFinding(
                    probe=p.name, kind="direct",
                    detail=f"protected relation '{p.relation}' in admitted set"))
                continue
            if not answer or cuts is None:
                continue
            try:
                t = self.checker._tension(answer, p.pole)
            except Exception:  # noqa: BLE001 - the inferred net is best-effort
                continue
            if t <= cuts.tau_ent:
                findings.append(DisclosureFinding(
                    probe=p.name, kind="inferred",
                    detail=f"answer entails protected pole '{p.pole}'",
                    tension=round(float(t), 4)))
        return findings


DEFAULT_STANCE_VALUES = frozenset({
    "forbidden", "permitted", "required", "optional", "issued", "denied",
})
"""Categorical stance values a policy attribute can hold. An admitted fact whose
tail is one of these carries a stance the output can contradict."""


class ClassifierDisclosureGuard:
    """Output-consistency guard backed by a trained classifier.

    Mechanism-agnostic: given a ``classify`` callable that maps a
    ``"Policy: <attribute> is <stance>. Claim: <answer>"`` string to a
    ``(label, confidence)`` over ``supported | contradicted | uncertain``, this
    guard flags an answer that contradicts a stance the admitted facts assert. The
    concrete classifier (e.g. a Qwen LoRA head) lives in the application; the guard
    only orchestrates, so knowlytix carries no training dependency.

    On the Chapter 13 held-out DoE test set a Qwen classifier behind this guard
    scored contradiction F1 0.83 and recall 1.0, against 0.62 / 0.80 for the
    geometric :class:`PolarityDisclosureGuard` on prose -- so this is the selected
    disclosure scanner, with least-context the primary control for protected
    attributes that must never be retrieved at all.

    Parameters
    ----------
    classify : Callable[[str], tuple[str, float]]
        Maps the reference+claim string to ``(label, confidence)``.
    relation_phrase : dict[str, str] or None
        Relation -> readable phrase for the reference premise; a relation absent
        here falls back to its de-underscored name.
    stance_values : Iterable[str] or None
        Tails treated as stances (default :data:`DEFAULT_STANCE_VALUES`).
    tau : float
        Minimum confidence for a ``contradicted`` verdict to fire (calibrated
        operating point; 0.0 acts on any contradiction).
    probes : Sequence[ProtectedProbe]
        Protected attributes; a probe relation present in the admitted set is a
        direct disclosure, independent of the classifier.
    """

    def __init__(
        self,
        classify: Callable[[str], tuple[str, float]],
        relation_phrase: dict[str, str] | None = None,
        *,
        stance_values=None,
        tau: float = 0.0,
        probes: Sequence[ProtectedProbe] = (),
    ):
        self.classify = classify
        self.relation_phrase = dict(relation_phrase or {})
        self.stance_values = frozenset(stance_values or DEFAULT_STANCE_VALUES)
        self.tau = tau
        self.probes = list(probes)

    def _phrase(self, relation: str) -> str:
        return self.relation_phrase.get(
            relation, (relation[4:] if relation.startswith("has_")
                       else relation).replace("_", " "))

    def scan(
        self,
        answer: str,
        admitted_facts: "Sequence[tuple[str, str, str]]",
    ) -> list[DisclosureFinding]:
        """Findings for *answer* given the admitted ``(head, relation, tail)`` facts."""
        findings: list[DisclosureFinding] = []
        admitted_relations = {r for _h, r, _t in admitted_facts}
        seen: set[str] = set()
        for _h, rel, tail in admitted_facts:
            if str(tail).strip().lower() not in self.stance_values or rel in seen:
                continue
            seen.add(rel)
            premise = f"Policy: {self._phrase(rel)} is {tail}. Claim: {answer}"
            try:
                label, conf = self.classify(premise)
            except Exception:  # noqa: BLE001 - a classifier failure is not a leak
                continue
            if label == "contradicted" and conf >= self.tau:
                findings.append(DisclosureFinding(
                    probe=rel, kind="policy_contradiction",
                    detail=f"answer contradicts stored stance '{tail}' for {rel}",
                    tension=round(float(conf), 4)))
        for p in self.probes:
            if p.relation in admitted_relations:
                findings.append(DisclosureFinding(
                    probe=p.name, kind="direct",
                    detail=f"protected relation '{p.relation}' in admitted set"))
        return findings


# ---------------------------------------------------------------------------
# Result and audit records
# ---------------------------------------------------------------------------

@dataclass
class GovernedFact:
    """A fact admitted to the agent, after zoning and field redaction.

    The tail is the value the agent actually receives; the pre-redaction value is
    never carried here or into the audit record.
    """

    head: str
    relation: str
    tail: str
    zone: str
    sensitivity: str
    score: float = 0.0
    confidence: float = 0.0
    location: str = ""
    redacted: bool = False


@dataclass
class DeniedFact:
    """A retrieved fact withheld from the agent, with the withholding reason."""

    head: str
    relation: str
    zone: str
    sensitivity: str
    reason: str


@dataclass
class RetrievalAuditRecord:
    """Immutable record of one governed retrieval decision.

    Answers, for after-the-fact review: why the agent retrieved this data, whether
    it was allowed, what it saw (redacted), and what was withheld. All admitted
    tails are already redacted, so the record is safe to persist.
    """

    timestamp: str
    workflow: str
    business_purpose: str
    query: str
    decision: str
    reason: str
    bound_heads: list[str] = field(default_factory=list)
    admitted: list[dict] = field(default_factory=list)
    denied: list[dict] = field(default_factory=list)
    redactions: list[dict] = field(default_factory=list)
    disclosure: list[dict] = field(default_factory=list)
    route: str = ""
    notice: str = ""
    confidence: float = 0.0
    granted_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GovernedResult:
    """Outcome of a governed retrieval.

    Parameters
    ----------
    decision : str
        ``"accept"`` (grounded, contract-scoped answer), ``"deny"`` (a policy
        block: the query or every fact fell outside the contract, or a disclosure
        was caught), or ``"abstain"`` (the pipeline found nothing groundable).
    reason : str
        Explanation, distinguishing a policy denial from a search failure.
    answer : str
        The answer the agent may use: redacted, contract-scoped, or a denial
        notice when suppressed.
    facts : list[GovernedFact]
        Admitted, redacted facts.
    denied : list[DeniedFact]
        Retrieved facts withheld by the contract.
    disclosure : list[DisclosureFinding]
        Protected attributes the disclosure guard flagged.
    audit : RetrievalAuditRecord
        The immutable audit record for this retrieval.
    raw : RagAnswer or None
        The underlying pipeline answer, for callers that need the gate detail.
    """

    decision: str
    reason: str
    answer: str = ""
    facts: list[GovernedFact] = field(default_factory=list)
    denied: list[DeniedFact] = field(default_factory=list)
    disclosure: list[DisclosureFinding] = field(default_factory=list)
    audit: RetrievalAuditRecord | None = None
    raw: "RagAnswer | None" = None


# Pipeline abstention reasons that are search failures, not policy denials.
_SEARCH_FAILURE_NOTICES = {"no_binding", "no_match"}
_DENIAL_NOTICE = (
    "I cannot access that data for this workflow."
)


# ---------------------------------------------------------------------------
# GovernedRetriever
# ---------------------------------------------------------------------------

class GovernedRetriever:
    """Policy-enforcing wrapper around a triple-mediated ``RagPipeline``.

    Composes an existing pipeline with a :class:`RetrievalContract` and a
    :class:`SensitivityMap`; optionally a :class:`PolarityDisclosureGuard` for the
    indirect-leakage backstop and a :class:`ProvenanceLedger` for span-level
    provenance. The pipeline's own gates decide what is *groundable*; this class
    decides what is *permitted*, and records both.

    Parameters
    ----------
    pipeline : RagPipeline
        The triple-mediated retrieval pipeline (built by the caller with its LLM).
    contract : RetrievalContract
        The workflow's retrieval envelope.
    sensitivity : SensitivityMap
        Entity -> zone -> sensitivity labels.
    granted_flags : frozenset[str]
        Flags the caller holds (satisfying ``contract.zone_flags``).
    disclosure_guard : PolarityDisclosureGuard or None
        Optional indirect-leakage backstop.
    ledger : ProvenanceLedger or None
        Optional span provenance; when absent the retriever falls back to the
        location each ``RetrievedFact`` already carries.
    audit_sink : Callable[[dict], None] or None
        Called with each audit record as a dict (e.g. to append to a log).
    synthesizer : Callable[[str, list[GovernedFact]], str] or None
        Produces the answer from ONLY the admitted, redacted facts. This is what
        makes least-context real: the pipeline retrieves with generation OFF, so
        the answering model never sees an ungoverned or unredacted fact. A grounded
        synthesizer should return ``""`` when no admitted fact answers the query
        (the retriever then abstains). When ``None``, the answer is a deterministic
        rendering of the admitted facts.
    suppress_on_disclosure : bool
        When True (default) a flagged disclosure suppresses the answer and makes
        the decision a denial.
    """

    def __init__(
        self,
        pipeline: "RagPipeline",
        contract: RetrievalContract,
        sensitivity: SensitivityMap,
        *,
        granted_flags: frozenset[str] = frozenset(),
        disclosure_guard: "PolarityDisclosureGuard | None" = None,
        ledger: "ProvenanceLedger | None" = None,
        audit_sink: Callable[[dict], None] | None = None,
        synthesizer: "Callable[[str, list], str] | None" = None,
        suppress_on_disclosure: bool = True,
    ):
        self.pipeline = pipeline
        self.contract = contract
        self.sensitivity = sensitivity
        self.granted_flags = frozenset(granted_flags)
        self.disclosure_guard = disclosure_guard
        self.ledger = ledger
        self.audit_sink = audit_sink
        self.synthesizer = synthesizer
        self.suppress_on_disclosure = suppress_on_disclosure

    # -- public API --------------------------------------------------------

    def retrieve(self, query: str) -> GovernedResult:
        """Run one governed retrieval for *query* and return a
        :class:`GovernedResult`. Emits an audit record via ``audit_sink``."""
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        bound_heads = self._bound_heads(query)  # recorded for audit, not a gate

        # --- during: retrieve with GENERATION OFF, so the answering model
        # never sees an ungoverned or unredacted fact. The answer is synthesized
        # afterwards from the admitted, redacted facts alone (least-context). -----
        ans = self.pipeline.query(query, generate=False)

        # --- after: zone filter + least-context + field redaction ---------
        # Enforcement is on the ACTUAL retrieved facts (their exact heads/zones),
        # not on the noisy query parse: a fact from a blocked zone is withheld; a
        # relation the purpose does not need is dropped; a restricted field is
        # redacted. A query that reaches only blocked zones ends with no admitted
        # facts and a blocked-zone denial -> an explicit policy denial.
        facts: list[GovernedFact] = []
        denied: list[DeniedFact] = []
        redactions: list[dict] = []
        n_blocked = 0
        for src in getattr(ans, "sources", None) or []:
            head = getattr(src, "head", "")
            rel = getattr(src, "relation", "")
            tail = str(getattr(src, "tail", ""))
            zone = self.sensitivity.zone_of(head)
            sens = self.sensitivity.sensitivity_of(head)

            ok, why = self.contract.permits_zone(zone, sens, self.granted_flags)
            if not ok:
                n_blocked += 1
                denied.append(DeniedFact(head, rel, zone, sens, why))
                continue
            if not self.contract.needs_relation(rel):
                denied.append(DeniedFact(
                    head, rel, zone, sens, "relation not needed for purpose"))
                continue

            red_tail, was_redacted, mode = self._redact(rel, tail)
            if mode == "drop":
                denied.append(DeniedFact(head, rel, zone, sens, "field dropped"))
                continue
            if was_redacted:
                redactions.append({"head": head, "relation": rel, "mode": mode})
            facts.append(GovernedFact(
                head=head, relation=rel, tail=red_tail, zone=zone,
                sensitivity=sens,
                score=float(getattr(src, "score", 0.0) or 0.0),
                confidence=float(getattr(src, "confidence", 0.0) or 0.0),
                location=self._provenance(head, rel, tail, src),
                redacted=was_redacted))

        # --- map pipeline outcome to a governed decision ------------------
        decision, reason = self._decide(ans, facts, n_blocked)

        # --- synthesize the answer from ONLY the admitted, redacted facts -
        answer = ""
        if decision == "accept" and facts:
            answer = self._synthesize(query, facts)
            if not answer.strip():
                # nothing admitted answered the query. If blocked-zone facts were
                # withheld, the answer required data the workflow may not read -- an
                # explicit policy denial; otherwise it is a plain search failure.
                if n_blocked:
                    decision, reason = "deny", (
                        f"policy denial: the query's answer requires blocked-zone "
                        f"data ({n_blocked} fact(s) withheld)")
                else:
                    decision, reason = "abstain", "no admitted fact answers the query"
        if decision == "deny":
            answer = _DENIAL_NOTICE

        # --- disclosure backstop -----------------------------------------
        findings: list[DisclosureFinding] = []
        if self.disclosure_guard is not None and answer:
            admitted_facts = [(f.head, f.relation, f.tail) for f in facts]
            findings = self.disclosure_guard.scan(answer, admitted_facts)
            if findings and self.suppress_on_disclosure:
                decision = "deny"
                reason = ("output-disclosure block: "
                          + "; ".join(f"{f.probe}({f.kind})" for f in findings))
                answer = _DENIAL_NOTICE

        audit = RetrievalAuditRecord(
            timestamp=ts, workflow=self.contract.workflow,
            business_purpose=self.contract.business_purpose, query=query,
            decision=decision, reason=reason, bound_heads=bound_heads,
            admitted=[{"head": f.head, "relation": f.relation, "tail": f.tail,
                       "zone": f.zone, "sensitivity": f.sensitivity,
                       "location": f.location, "redacted": f.redacted}
                      for f in facts],
            denied=[{"head": d.head, "relation": d.relation, "zone": d.zone,
                     "sensitivity": d.sensitivity, "reason": d.reason}
                    for d in denied],
            redactions=redactions,
            disclosure=[{"probe": f.probe, "kind": f.kind, "detail": f.detail,
                         "tension": f.tension} for f in findings],
            route=str(getattr(ans, "route", "")),
            notice=str(getattr(ans, "notice", "") or ""),
            confidence=float(getattr(ans, "confidence", 0.0) or 0.0),
            granted_flags=sorted(self.granted_flags))
        self._emit(audit)

        return GovernedResult(
            decision=decision, reason=reason, answer=answer, facts=facts,
            denied=denied, disclosure=findings, audit=audit, raw=ans)

    # -- internals ---------------------------------------------------------

    def _synthesize(self, query: str, facts: "list[GovernedFact]") -> str:
        """Answer from ONLY the admitted, redacted facts. Delegates to a
        caller-supplied ``synthesizer`` (a grounded LLM that may abstain by
        returning ``""``); otherwise renders the facts deterministically."""
        if self.synthesizer is not None:
            try:
                return self.synthesizer(query, facts) or ""
            except Exception:  # noqa: BLE001 - a synthesis failure is not a leak
                return ""
        lines = "; ".join(f"{f.relation.removeprefix('has_').replace('_', ' ')}: "
                          f"{f.tail}" for f in facts)
        return f"Based on the admitted facts -- {lines}." if lines else ""

    def _bound_heads(self, query: str) -> list[str]:
        """Head entities the query binds to, via the pipeline's parse/bind."""
        try:
            ex = self.pipeline.extract(query)
        except Exception:  # noqa: BLE001 - a bind failure is not a leak
            return []
        heads: list[str] = []
        for bt in getattr(ex, "bound_triples", []) or []:
            h = getattr(bt, "head", None)
            if h and not str(h).startswith("?") and h not in heads:
                heads.append(h)  # skip unbound query variables (?x0)
        return heads

    def _redact(self, relation: str, tail: str) -> tuple[str, bool, str]:
        """Apply the contract redactor for *relation*. Returns
        ``(tail, was_redacted, mode)``; mode is ``""`` when no rule applies and
        ``"drop"`` when the field is to be suppressed."""
        mode = self.contract.redact_relations.get(relation)
        if mode is None:
            return tail, False, ""
        if mode == "drop":
            return tail, True, "drop"
        redactor = REDACTORS.get(mode)
        if redactor is None:
            return "[REDACTED]", True, mode
        return redactor(tail), True, mode

    def _provenance(self, head: str, rel: str, tail: str, src) -> str:
        """Span location for a fact: the ledger if available, else the location
        the RetrievedFact already carries."""
        if self.ledger is not None:
            try:
                p = self.ledger.resolve(head, rel, tail)
                loc = p.location() if p is not None else ""
                if loc:
                    return loc
            except Exception:  # noqa: BLE001
                pass
        return str(getattr(src, "location", "") or "")

    def _decide(self, ans, facts, n_blocked) -> tuple[str, str]:
        """Map the pipeline outcome and the contract filter to a decision.

        A blocked-zone hit with nothing admitted is an explicit **policy denial**
        (the query reached only data the workflow may not read); an empty retrieval
        or a below-gate abstention is a **search failure**. The two are reported
        distinctly, per the make-denial-explicit principle.
        """
        if facts:
            return "accept", "within contract"
        if n_blocked:
            return "deny", (
                f"policy denial: retrieval reached only blocked zones "
                f"({n_blocked} fact(s) withheld)")
        decision = str(getattr(ans, "decision", "abstain"))
        notice = str(getattr(ans, "notice", "") or "")
        if decision == "abstain" and notice in _SEARCH_FAILURE_NOTICES:
            return "abstain", f"search failure ({notice})"
        return "abstain", "search failure (no admissible facts retrieved)"

    def _deny(self, ts, query, bound_heads, reason, *, denied) -> GovernedResult:
        audit = RetrievalAuditRecord(
            timestamp=ts, workflow=self.contract.workflow,
            business_purpose=self.contract.business_purpose, query=query,
            decision="deny", reason=reason, bound_heads=bound_heads,
            denied=[{"head": d.head, "relation": d.relation, "zone": d.zone,
                     "sensitivity": d.sensitivity, "reason": d.reason}
                    for d in denied],
            granted_flags=sorted(self.granted_flags))
        self._emit(audit)
        return GovernedResult(
            decision="deny", reason=reason, answer=_DENIAL_NOTICE,
            denied=denied, audit=audit)

    def _emit(self, audit: RetrievalAuditRecord) -> None:
        if self.audit_sink is not None:
            try:
                self.audit_sink(audit.to_dict())
            except Exception as exc:  # noqa: BLE001
                print(f"  [GovernedRetriever] audit sink error: {exc}")
