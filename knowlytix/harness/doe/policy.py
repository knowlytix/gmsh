# SPDX-License-Identifier: Apache-2.0
"""GMSH Runtime Policy Engine --- pre-execution policy checking for agent testing.

Provides a GMS-backed policy engine that validates every tool call before
execution.  Policy rules are encoded as GMS triples so that the same
geometric primitives (geodesic distance, tension energy, graph queries)
used for knowledge verification also enforce runtime constraints.

Rules can be loaded from YAML or added programmatically.  The engine
integrates with :class:`~harness.doe.gateway.ToolGateway` via the
``before_tool`` hook.

Usage::

    from knowlytix.harness.doe.policy import PolicyEngine, PolicyRule

    engine = PolicyEngine(
        store=my_store,
        rules=[
            PolicyRule(
                rule_id="RULE-001",
                action="submit_escalation",
                requirement="human_approval",
                rule_type="requires",
            ),
        ],
        tracing=tracing_gms,
        run_id="run_001",
    )
    decision = engine.check("submit_escalation", {"ticket": 42})
    if not decision.allowed:
        print(decision.reason, decision.blocking_rules)

YAML format::

    rules:
      - rule_id: "RULE-001"
        action: "submit_escalation"
        requirement: "human_approval"
        rule_type: "requires"
        context: "*"
        severity: "error"
      - rule_id: "RULE-002"
        action: "delete_*"
        requirement: "admin_context"
        rule_type: "blocks"
        context: "production"
        severity: "error"
"""

from __future__ import annotations

import fnmatch
import uuid
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import yaml

from knowlytix.doc.store import GMSExpertStore
from knowlytix.doc.extract import ExtractedTriple

if TYPE_CHECKING:
    from knowlytix.doc.learn import RuntimeLearner
    from .tracing import DOETracingGMS


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PolicyRule:
    """A single policy rule governing tool or action execution.

    Parameters
    ----------
    rule_id : str
        Unique identifier for this rule (e.g. ``"RULE-001"``).
    action : str
        Tool name or action pattern matched via ``fnmatch`` (e.g.
        ``"delete_*"``).
    requirement : str
        What is required or blocked: ``"human_approval"``,
        ``"data_loaded"``, ``"admin_context"``, etc.
    rule_type : str
        One of ``"requires"`` (action needs the requirement),
        ``"blocks"`` (action is blocked when requirement matches),
        or ``"allows"`` (explicitly permit the action).
    context : str
        Scenario context pattern matched via ``fnmatch``.  Defaults to
        ``"*"`` (all contexts).
    severity : str
        ``"error"`` for hard blocks, ``"warning"`` for log-only.
    """

    rule_id: str = ""
    action: str = ""
    requirement: str = ""
    rule_type: str = "requires"
    context: str = "*"
    severity: str = "error"

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary.

        Returns
        -------
        dict
            Flat dictionary representation of this rule.
        """
        return {
            "rule_id": self.rule_id,
            "action": self.action,
            "requirement": self.requirement,
            "rule_type": self.rule_type,
            "context": self.context,
            "severity": self.severity,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PolicyRule":
        """Deserialize from a plain dictionary.

        Parameters
        ----------
        d : dict
            Dictionary with keys matching ``PolicyRule`` fields.

        Returns
        -------
        PolicyRule
            Reconstructed rule.
        """
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


@dataclass
class PolicyDecision:
    """Result of a policy check against a proposed action.

    Parameters
    ----------
    allowed : bool
        Whether the action is permitted.
    action : str
        The action that was checked.
    rules_checked : list[str]
        IDs of all rules that were evaluated.
    blocking_rules : list[str]
        IDs of rules that blocked the action.
    gms_scores : dict[str, float]
        Geometric scores collected during the check (plausibility,
        tension, etc.).
    reason : str
        Human-readable explanation of the decision.
    """

    allowed: bool = True
    action: str = ""
    rules_checked: list[str] = field(default_factory=list)
    blocking_rules: list[str] = field(default_factory=list)
    gms_scores: dict[str, float] = field(default_factory=dict)
    reason: str = ""


# ---------------------------------------------------------------------------
# PolicyEngine
# ---------------------------------------------------------------------------

class PolicyEngine:
    """Pre-execution policy checking using GMS geometric verification.

    Policy rules are encoded as GMS triples in the store::

        ("submit_escalation", "requires", "human_approval")
        ("delete_data", "blocks", "production")

    The engine checks each tool call against these rules using:

    - ``query_triples()`` for explicit rule lookup
    - ``score_triple()`` for geometric plausibility
    - ``tension_energy()`` for contradiction detection

    Parameters
    ----------
    store : GMSExpertStore
        GMS store used for policy graph queries and geometric scoring.
    rules : list[PolicyRule] or None
        Initial set of policy rules.  Additional rules can be loaded
        from YAML via :meth:`load_rules_from_yaml`.
    tracing : DOETracingGMS or None
        If provided, every policy decision is traced as GMS triples.
    run_id : str
        Identifier for the current run (used in tracing).
    judge : GMSJudge or None
        If provided, calibrated thresholds are read from
        ``judge._thresholds`` and override the class-level defaults:

        - ``_thresholds["policy_geodesic"]`` -> :attr:`PLAUSIBILITY_THRESHOLD`
        - ``_thresholds["tau_contra"]``      -> :attr:`TENSION_THRESHOLD`

        Any key absent from ``judge._thresholds`` falls through to the
        hand-picked default, mirroring the partial-calibration policy in
        :class:`VerificationRouter`.
    """

    PLAUSIBILITY_THRESHOLD = 1.5
    """Default maximum geodesic distance for a 'plausible' action in context.

    Class-level fallback; per-instance value (set in :meth:`__init__`) is
    overridden when a calibrated judge supplies ``policy_geodesic``.
    """

    TENSION_THRESHOLD = 1.7
    """Default tension energy above which a strong contradiction is detected.

    Class-level fallback; per-instance value (set in :meth:`__init__`) is
    overridden when a calibrated judge supplies ``tau_contra``.
    """

    def __init__(
        self,
        store: GMSExpertStore,
        rules: list[PolicyRule] | None = None,
        tracing: "DOETracingGMS | None" = None,
        run_id: str = "",
        judge: Any | None = None,
        plausibility_relation: str | None = None,
    ):
        self.store = store
        self.tracing = tracing
        self.run_id = run_id
        self._rules: list[PolicyRule] = []
        self._learner: "RuntimeLearner | None" = None

        # Layer-3 probe: when ``plausibility_relation`` is set, Layer 3
        # scores ``(context, rel, action)`` -- the agent-centric
        # convention used by the policy governance store
        # (e.g. ``rel="uses_tool"``).  When unset, falls back to the
        # legacy ``(action, "allowed_in", context)`` probe.
        self._plausibility_relation = plausibility_relation

        # Per-instance thresholds: start at class defaults, override from
        # the judge's calibrated values when available.  Mirrors
        # VerificationRouter (verifiers.py:1310-1328).
        self.PLAUSIBILITY_THRESHOLD = type(self).PLAUSIBILITY_THRESHOLD
        self.TENSION_THRESHOLD = type(self).TENSION_THRESHOLD
        self._judge = judge
        self._calibration_source = {"plausibility": "default",
                                    "tension": "default"}
        if judge is not None and getattr(judge, "_thresholds", None):
            thr = judge._thresholds
            if thr.get("policy_geodesic") is not None:
                self.PLAUSIBILITY_THRESHOLD = float(thr["policy_geodesic"])
                self._calibration_source["plausibility"] = "judge.policy_geodesic"
            if thr.get("tau_contra") is not None:
                self.TENSION_THRESHOLD = float(thr["tau_contra"])
                self._calibration_source["tension"] = "judge.tau_contra"
            # Otherwise: per Ch9 §9.2.2, the 3-class tension fit refuses
            # to emit a cut when the contradict class is empty or the
            # classes don't separate. The default TENSION_THRESHOLD
            # stays; we do not synthesise a fallback.

        # Build a RuntimeLearner for encoding rules as GMS triples
        self._init_learner()

        if rules:
            self.encode_rules(rules)

    # -- Initialization helpers --------------------------------------------

    def _init_learner(self) -> None:
        """Create a RuntimeLearner from the store if possible."""
        try:
            from knowlytix.doc.learn import RuntimeLearner as _RL
            if hasattr(self.store, "config") and self.store.config is not None:
                self._learner = _RL(self.store, self.store.config.learn)
            else:
                # Attempt with default learn config
                from knowlytix.doc.config import LearnConfig
                self._learner = _RL(self.store, LearnConfig())
        except Exception:
            self._learner = None

    def _learn(self, triples: list[ExtractedTriple]) -> Any:
        """Encode triples into the GMS store via the RuntimeLearner.

        Parameters
        ----------
        triples : list[ExtractedTriple]
            Triples to learn.

        Returns
        -------
        Any
            The ``LearnResult`` if successful, None otherwise.
        """
        if not self._learner or not triples:
            return None
        try:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return self._learner.learn_triples(triples, device)
        except Exception as exc:
            print(f"  [PolicyEngine] Learn error: {exc}")
            return None

    # -- Rule loading and encoding -----------------------------------------

    def load_rules_from_yaml(self, path: str) -> list[PolicyRule]:
        """Load policy rules from a YAML file and encode them into GMS.

        The file must contain a top-level ``rules`` key with a list of
        rule objects.  See module docstring for the expected format.

        Parameters
        ----------
        path : str
            Path to the YAML file.

        Returns
        -------
        list[PolicyRule]
            The loaded and encoded policy rules.

        Raises
        ------
        FileNotFoundError
            If the YAML file does not exist.
        ValueError
            If the file does not contain a ``rules`` key.
        """
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)

        if not isinstance(data, dict) or "rules" not in data:
            raise ValueError(
                f"YAML file must contain a top-level 'rules' key: {path}"
            )

        loaded = [PolicyRule.from_dict(entry) for entry in data["rules"]]
        self.encode_rules(loaded)
        return loaded

    def encode_rules(self, rules: list[PolicyRule]) -> None:
        """Encode policy rules as GMS triples and store internally.

        For each rule the following triples are learned:

        - ``(rule.action, rule.rule_type, rule.requirement)``
        - ``(rule.rule_id, "is_type", "policy_rule")``
        - ``(rule.rule_id, "applies_to", rule.action)``
        - ``(rule.rule_id, "severity", rule.severity)``

        Parameters
        ----------
        rules : list[PolicyRule]
            Rules to encode.
        """
        triples: list[ExtractedTriple] = []
        for rule in rules:
            # Assign an ID if missing
            if not rule.rule_id:
                rule.rule_id = f"rule_{uuid.uuid4().hex[:8]}"

            # Primary rule triple
            triples.append(ExtractedTriple(
                head=rule.action,
                relation=rule.rule_type,
                tail=rule.requirement,
            ))
            # Metadata triples
            triples.append(ExtractedTriple(
                head=rule.rule_id,
                relation="is_type",
                tail="policy_rule",
            ))
            triples.append(ExtractedTriple(
                head=rule.rule_id,
                relation="applies_to",
                tail=rule.action,
            ))
            triples.append(ExtractedTriple(
                head=rule.rule_id,
                relation="severity",
                tail=rule.severity,
            ))

            self._rules.append(rule)

        self._learn(triples)

    # -- Main check method -------------------------------------------------

    def check(
        self,
        action: str,
        args: dict,
        context: str = "",
    ) -> PolicyDecision:
        """Check whether an action is permitted under current policy.

        Performs four layers of verification:

        1. **Explicit rule matching** via fnmatch against stored rules.
        2. **GMS graph lookup** via ``query_triples()`` for the action entity.
        3. **Geometric plausibility** via ``score_triple()`` for the
           action in the given context.
        4. **Contradiction detection** via ``tension_energy()`` between
           the action and any blocking rule entities.

        Parameters
        ----------
        action : str
            Tool name or action identifier to check.
        args : dict
            Arguments that will be passed to the tool (currently used
            for future extensions; reserved).
        context : str
            Scenario context string matched against rule context patterns.
            Empty string matches rules with ``context="*"`` only.

        Returns
        -------
        PolicyDecision
            Structured decision including pass/fail, blocking rules,
            geometric scores, and a human-readable reason.
        """
        rules_checked: list[str] = []
        blocking_rules: list[str] = []
        gms_scores: dict[str, float] = {}
        reasons: list[str] = []
        explicitly_allowed = False

        # -----------------------------------------------------------------
        # 1. Explicit rule matching
        # -----------------------------------------------------------------
        matched = self._match_rules(action, context)
        for rule in matched:
            rules_checked.append(rule.rule_id)

            if rule.rule_type == "allows":
                explicitly_allowed = True

            elif rule.rule_type == "blocks":
                if rule.severity == "error":
                    blocking_rules.append(rule.rule_id)
                    reasons.append(
                        f"Rule {rule.rule_id}: '{action}' blocked by "
                        f"'{rule.requirement}' in context '{context or '*'}'"
                    )
                else:
                    reasons.append(
                        f"Warning (rule {rule.rule_id}): '{action}' soft-blocked "
                        f"by '{rule.requirement}'"
                    )

            elif rule.rule_type == "requires":
                if rule.severity == "error":
                    blocking_rules.append(rule.rule_id)
                    reasons.append(
                        f"Rule {rule.rule_id}: '{action}' requires "
                        f"'{rule.requirement}' (not met)"
                    )
                else:
                    reasons.append(
                        f"Warning (rule {rule.rule_id}): '{action}' should "
                        f"have '{rule.requirement}'"
                    )

        # If explicitly allowed and no hard blocks, short-circuit
        if explicitly_allowed and not blocking_rules:
            decision = PolicyDecision(
                allowed=True,
                action=action,
                rules_checked=rules_checked,
                blocking_rules=[],
                gms_scores=gms_scores,
                reason="Explicitly allowed by policy rule",
            )
            self._trace_decision(decision)
            return decision

        # -----------------------------------------------------------------
        # 2. GMS graph lookup
        # -----------------------------------------------------------------
        action_entity = self._resolve_entity(action)
        if action_entity is not None:
            try:
                graph_triples = self.store.query_triples(head=action_entity)
                for triple in graph_triples:
                    # query_triples returns (head, relation, tail) tuples
                    # per docgms/DESIGN.md §4.4; accept dict/attr forms too
                    # so alternative store impls stay compatible.
                    if isinstance(triple, tuple) and len(triple) == 3:
                        _, rel, tail = triple
                    elif isinstance(triple, dict):
                        rel = triple.get("relation", "")
                        tail = triple.get("tail", "")
                    else:
                        rel = getattr(triple, "relation", "")
                        tail = getattr(triple, "tail", "")

                    if rel == "blocks":
                        rule_id = f"gms_graph:{action_entity}->blocks->{tail}"
                        if rule_id not in blocking_rules:
                            blocking_rules.append(rule_id)
                            reasons.append(
                                f"GMS graph: '{action_entity}' blocks '{tail}'"
                            )
                    elif rel == "requires":
                        rule_id = f"gms_graph:{action_entity}->requires->{tail}"
                        if rule_id not in blocking_rules:
                            blocking_rules.append(rule_id)
                            reasons.append(
                                f"GMS graph: '{action_entity}' requires '{tail}'"
                            )
            except Exception as exc:
                reasons.append(f"GMS graph lookup error: {exc}")

        # -----------------------------------------------------------------
        # 3. Geometric plausibility (strict entity resolution: a fallback
        # to the raw name would score against uninitialised embeddings)
        # -----------------------------------------------------------------
        action_entity_strict = self._resolve_entity_strict(action)
        if context:
            if action_entity_strict is None:
                reasons.append(
                    f"Layer 3 skipped: action '{action}' not in GMS store"
                )
            else:
                context_entity = self._resolve_entity_strict(context)
                if context_entity is None:
                    reasons.append(
                        f"Layer 3 skipped: context '{context}' not in GMS store"
                    )
                else:
                    action_entity = action_entity_strict
                    try:
                        if self._plausibility_relation is not None:
                            # Agent-centric: score (context, rel, action)
                            probe_rel = self._plausibility_relation
                            score = self.store.score_triple(
                                context_entity, probe_rel, action_entity
                            )
                        else:
                            probe_rel = "allowed_in"
                            score = self.store.score_triple(
                                action_entity, probe_rel, context_entity
                            )
                        if score is not None:
                            gms_scores["plausibility"] = float(score)
                            if float(score) > self.PLAUSIBILITY_THRESHOLD:
                                rule_id = f"gms_plausibility:{action_entity}"
                                blocking_rules.append(rule_id)
                                reasons.append(
                                    f"GMS plausibility: '{action}' in context "
                                    f"'{context}' scored {score:.4f} "
                                    f"via '{probe_rel}' "
                                    f"(threshold {self.PLAUSIBILITY_THRESHOLD})"
                                )
                        else:
                            reasons.append(
                                f"Layer 3 skipped: relation '{probe_rel}' "
                                f"not in GMS store"
                            )
                    except Exception as exc:
                        reasons.append(
                            f"GMS plausibility scoring error: {exc}"
                        )

        # -----------------------------------------------------------------
        # 4. Contradiction check via tension_energy (strict resolution:
        # a graph-derived blocking rule ID like
        # "gms_graph:foo->blocks->bar" would otherwise be scored as if
        # it were a real entity)
        # -----------------------------------------------------------------
        if action_entity_strict is not None:
            for br in list(blocking_rules):
                # Only check tension for explicit rule IDs (not graph-derived)
                rule_entity = self._resolve_entity_strict(br)
                if rule_entity is not None:
                    try:
                        tension = self.store.tension_energy(
                            action_entity_strict, rule_entity
                        )
                        if tension is not None:
                            gms_scores[f"tension:{br}"] = float(tension)
                            if float(tension) > self.TENSION_THRESHOLD:
                                reasons.append(
                                    f"Strong contradiction: tension("
                                    f"'{action}', '{br}') = {tension:.4f} "
                                    f"(threshold {self.TENSION_THRESHOLD})"
                                )
                    except Exception:
                        pass  # tension check is supplementary

        # -----------------------------------------------------------------
        # 5. Build decision
        # -----------------------------------------------------------------
        allowed = len(blocking_rules) == 0

        if not allowed:
            reason = "; ".join(reasons) if reasons else "Blocked by policy"
        elif reasons:
            # Allowed but with informational notes (e.g. Layer 3 skipped,
            # plausibility score below threshold).  Surface them so silent
            # gates in the four-layer pipeline are visible in transcripts.
            reason = "Action permitted; notes: " + "; ".join(reasons)
        else:
            reason = "Action permitted: no blocking rules matched"

        decision = PolicyDecision(
            allowed=allowed,
            action=action,
            rules_checked=rules_checked,
            blocking_rules=blocking_rules,
            gms_scores=gms_scores,
            reason=reason,
        )

        # -----------------------------------------------------------------
        # 6. Trace
        # -----------------------------------------------------------------
        self._trace_decision(decision)

        return decision

    # -- Internal helpers ---------------------------------------------------

    def _match_rules(self, action: str, context: str) -> list[PolicyRule]:
        """Return all rules whose action and context patterns match.

        Parameters
        ----------
        action : str
            Tool name or action identifier.
        context : str
            Scenario context string.  Empty string is treated as
            matching only the wildcard ``"*"`` pattern.

        Returns
        -------
        list[PolicyRule]
            All matching rules in the order they were registered.
        """
        matched: list[PolicyRule] = []
        ctx = context if context else ""
        for rule in self._rules:
            if not fnmatch.fnmatch(action, rule.action):
                continue
            # Context matching: rule.context="*" matches everything;
            # otherwise require fnmatch match on actual context.
            if rule.context == "*" or (ctx and fnmatch.fnmatch(ctx, rule.context)):
                matched.append(rule)
        return matched

    def _resolve_entity(self, name: str) -> str | None:
        """Resolve an entity name through GMS fuzzy matching.

        Falls back to the raw name if fuzzy matching returns None,
        ensuring graceful degradation when the entity is not yet in
        the graph.  Use :meth:`_resolve_entity_strict` for callers that
        need a guarantee the result exists in the store (e.g. Layer 3-4
        geometric scoring, where a fallback name would produce a
        meaningless score against random embedding noise).

        Parameters
        ----------
        name : str
            Entity name to resolve.

        Returns
        -------
        str or None
            Resolved entity name, or None if the store has no match
            and the entity cannot be used for geometric queries.
        """
        try:
            resolved = self.store.fuzzy_match_entity(name)
            return resolved if resolved is not None else name
        except Exception:
            return name

    def _resolve_entity_strict(self, name: str) -> str | None:
        """Resolve an entity only if it actually exists in the store.

        Unlike :meth:`_resolve_entity`, returns ``None`` (not the raw
        name) when the entity is absent.  Used by Layers 3-4 of
        :meth:`check` so geometric primitives never run on names that
        have no embedding.
        """
        try:
            resolved = self.store.fuzzy_match_entity(name)
            if resolved is not None:
                return resolved
            adapter = getattr(self.store, "adapter", None)
            if adapter is not None and name in getattr(
                adapter, "entity_to_idx", {}
            ):
                return name
            return None
        except Exception:
            return None

    def _trace_decision(self, decision: PolicyDecision) -> None:
        """Record a policy decision via the tracing store.

        Parameters
        ----------
        decision : PolicyDecision
            The decision to trace.
        """
        if self.tracing is None:
            return
        try:
            self.tracing.trace_policy_decision(
                run_id=self.run_id,
                action=decision.action,
                allowed=decision.allowed,
                blocking_rules=decision.blocking_rules,
                gms_scores=decision.gms_scores,
            )
        except Exception as exc:
            print(f"  [PolicyEngine] Tracing error: {exc}")


# ---------------------------------------------------------------------------
# Module-level YAML loader
# ---------------------------------------------------------------------------

def load_policy_rules(path: str) -> list[PolicyRule]:
    """Load policy rules from a YAML file.

    The file should contain a top-level ``rules`` key with a list of
    rule objects::

        rules:
          - rule_id: "RULE-001"
            action: "submit_escalation"
            requirement: "human_approval"
            rule_type: "requires"
            context: "*"
            severity: "error"

    Parameters
    ----------
    path : str
        Path to the YAML file.

    Returns
    -------
    list[PolicyRule]
        Deserialized policy rules.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If the file does not contain a ``rules`` key.
    """
    with open(path, "r") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict) or "rules" not in data:
        raise ValueError(
            f"YAML file must contain a top-level 'rules' key: {path}"
        )

    return [PolicyRule.from_dict(entry) for entry in data["rules"]]
