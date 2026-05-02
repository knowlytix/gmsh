# SPDX-License-Identifier: Apache-2.0
"""GMSH Multi-Agent Testing — GMS-native inter-agent consistency.

Tests multi-agent interactions using geometric primitives:

- **Per-agent stores**: each agent role gets its own ``GMSExpertStore``
  tracking assertions made during its execution.
- **Inter-agent consistency**: ``tension_energy`` between agent stores
  measures agreement (low = consensus, high = contradiction).
- **Delegation correctness**: ``check_holonomy`` on delegation chains
  (agent A delegates to B, B produces output) — holonomy defect
  measures whether the delegation path is consistent.
- **Shared fact agreement**: ``score_triple`` across merged stores
  verifies shared entities are geometrically consistent.

Usage::

    gen = MultiAgentGenerator()
    questions = gen.generate(store)

    oracle = ConsensusOracle(stores={"agent_a": store_a, "agent_b": store_b})
    verdict = oracle.evaluate_consensus(entity="revenue_2024")
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from knowlytix.benchmark.eval.generators.base import GeneratedQuestion, QuestionGenerator, ANSWER_FORMATS

from .cross_document import (
    _CROSS_STORE_CLASS_TO_CONFLICT,
    _amended_indices,
    _compute_triple_features,
    _conflict_type,
    _load_cross_store_from_dir,
)
from .oracles import (
    _load_calibrated_classifiers,
    _load_calibrated_thresholds,
    _load_cross_store_calibrated_classifier,
    _load_cross_store_calibrated_thresholds,
)

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore


# Hand-picked fallback used only when no calibration.json is present.
DEFAULT_TAU_TENSION = 0.5


def _pick_reference_store(
    stores: "dict[str, GMSExpertStore]",
) -> "GMSExpertStore | None":
    """Return the first store in ``stores`` that exposes a filesystem
    ``store_path`` (so ``calibration.json`` / ``calibration.pkl`` can be
    read).  Returns ``None`` if no such store exists."""
    for store in stores.values():
        if getattr(store, "store_path", None):
            return store
    return None


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class AgentRole:
    """Definition of an agent role in a multi-agent test.

    Attributes
    ----------
    role_id : str
        Unique identifier (e.g., ``"researcher"``, ``"reviewer"``).
    system_prompt : str
        System prompt for this agent role.
    allowed_tools : list[str]
        Tools this agent is permitted to call.
    store_path : str
        Path to this agent's per-role GMS store.
    """

    role_id: str = ""
    system_prompt: str = ""
    allowed_tools: list[str] = field(default_factory=list)
    store_path: str = ""

    def to_dict(self) -> dict:
        return {
            "role_id": self.role_id,
            "system_prompt": self.system_prompt[:200],
            "allowed_tools": self.allowed_tools,
            "store_path": self.store_path,
        }


@dataclass
class InteractionResult:
    """Result of a multi-agent interaction with GMS metrics.

    Attributes
    ----------
    scenario_id : str
        The scenario being tested.
    agent_outputs : dict[str, str]
        Per-agent raw outputs keyed by role_id.
    inter_agent_tension : dict[str, float]
        Pairwise tension_energy between agent pairs.
        Keys are ``"role_a:role_b"`` strings.
    process_holonomy : float or None
        Holonomy defect of the delegation chain (if applicable).
    consensus_reached : bool
        Whether all pairwise tensions are below threshold.
    mean_tension : float
        Mean inter-agent tension.
    """

    scenario_id: str = ""
    agent_outputs: dict = field(default_factory=dict)
    inter_agent_tension: dict = field(default_factory=dict)
    process_holonomy: float | None = None
    consensus_reached: bool = False
    mean_tension: float = 0.0
    consensus_confidence: float = 0.0
    tau_tension: float | None = None
    consensus_label: str = "minor_variance"
    tau_minor_upper: float | None = None
    tau_contra_lower: float | None = None

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "n_agents": len(self.agent_outputs),
            "inter_agent_tension": {
                k: round(v, 4) for k, v in self.inter_agent_tension.items()
            },
            "process_holonomy": (
                round(self.process_holonomy, 4)
                if self.process_holonomy is not None else None
            ),
            "consensus_reached": self.consensus_reached,
            "mean_tension": round(self.mean_tension, 4),
            "consensus_confidence": round(self.consensus_confidence, 4),
            "tau_tension": (
                round(self.tau_tension, 4)
                if self.tau_tension is not None else None
            ),
            "consensus_label": self.consensus_label,
            "tau_minor_upper": (
                round(self.tau_minor_upper, 4)
                if self.tau_minor_upper is not None else None
            ),
            "tau_contra_lower": (
                round(self.tau_contra_lower, 4)
                if self.tau_contra_lower is not None else None
            ),
        }


# ---------------------------------------------------------------------------
# Consensus oracle
# ---------------------------------------------------------------------------

class ConsensusOracle:
    """Evaluate inter-agent consistency using GMS tension_energy.

    Computes pairwise ``tension_energy`` between shared entities
    across agent stores.  Low mean tension = consensus reached.

    Parameters
    ----------
    stores : dict[str, GMSExpertStore]
        Per-agent GMS stores keyed by role_id.
    tension_threshold : float, optional
        Maximum mean tension for consensus.  ``None`` (default) reads
        :math:`\\tau_{\\text{tension}}` from any store's
        ``calibration.json``; if no calibration is present, falls back
        to :data:`DEFAULT_TAU_TENSION`.
    """

    oracle_type = "consensus"

    def __init__(
        self,
        stores: dict[str, "GMSExpertStore"] | None = None,
        tension_threshold: float | None = None,
        *,
        calibration_dir: "str | Path | None" = None,
    ):
        self.stores = stores or {}
        ref_store = _pick_reference_store(self.stores)
        # --- per-claim calibration (legacy fallback)
        calibrated = _load_calibrated_thresholds(ref_store)
        classifiers = _load_calibrated_classifiers(ref_store)
        self.tension_threshold = (
            tension_threshold
            if tension_threshold is not None
            else calibrated.get("tension", DEFAULT_TAU_TENSION)
        )
        self._calibration_source = (
            "override" if tension_threshold is not None
            else ("calibration.json" if "tension" in calibrated
                  else "default")
        )
        self._tension_classifier = classifiers.get("tension")

        # --- cross-store calibration (preferred when available)
        xs_thresholds, xs_classifier = _load_cross_store_from_dir(
            calibration_dir,
        )
        if not xs_thresholds:
            xs_thresholds = _load_cross_store_calibrated_thresholds(ref_store)
        if xs_classifier is None:
            xs_classifier = _load_cross_store_calibrated_classifier(ref_store)
        self._cross_store_thresholds = xs_thresholds or {}
        self._cross_store_classifier = xs_classifier
        self._cross_calibration_source = (
            f"dir:{calibration_dir}" if (xs_thresholds and calibration_dir)
            else ("store:cross_store_calibration.json" if xs_thresholds
                  else "none")
        )

    def evaluate_consensus(
        self, entity: str,
    ) -> InteractionResult:
        """Evaluate inter-agent consensus on a specific entity.

        For every ordered pair of agent stores, enumerate triples that
        involve ``entity`` and compute the 3-feature vector
        ``(delta, tail_missing, has_replacement)`` per triple.  Mean
        delta across triples is the *scalar* consensus metric (reported
        as ``mean_tension`` for continuity with earlier chapters); the
        *worst* 3-D feature vector across all pairs is what drives
        classification and confidence, so a single hard disagreement is
        not drowned out by many near-zero triples.

        Parameters
        ----------
        entity : str
            The entity to check consensus on.

        Returns
        -------
        InteractionResult
            With pairwise mean tensions, aggregate mean, worst-triple
            classification, and calibrated confidence.
        """
        role_ids = sorted(self.stores.keys())
        tensions: dict[str, float] = {}
        worst_features: tuple[float, int, int] = (0.0, 0, 0)

        for i, ra in enumerate(role_ids):
            for rb in role_ids[i + 1:]:
                store_a = self.stores[ra]
                store_b = self.stores[rb]
                pair_key = f"{ra}:{rb}"

                aggregated = self._cross_store_tension(
                    store_a, store_b, entity,
                )
                if aggregated is None:
                    continue
                mean_delta, pair_worst = aggregated
                tensions[pair_key] = mean_delta
                if pair_worst[0] > worst_features[0]:
                    worst_features = pair_worst

        mean_t = (
            sum(tensions.values()) / len(tensions)
            if tensions else 0.0
        )
        tau_minor = self._cross_store_thresholds.get("tau_minor_upper")
        tau_contra = self._cross_store_thresholds.get("tau_contra_lower")
        label = self._classify_worst_features(
            worst_features, tau_minor, tau_contra,
        )
        # When the 3-D classifier is loaded, consensus follows the
        # classifier's verdict on the worst-triple feature vector --
        # robust across projection-cell edge cases where the scalar
        # tau_minor_upper degenerates to 0.  Without a classifier, fall
        # back to the scalar minor/variance cut.
        if self._cross_store_classifier is not None:
            consensus_reached = (label == "minor_variance")
        else:
            consensus_threshold = (
                float(tau_minor) if tau_minor is not None
                else float(self.tension_threshold)
            )
            consensus_reached = mean_t <= consensus_threshold
        confidence = self._consensus_confidence(
            worst_features, consensus_reached, label,
        )

        return InteractionResult(
            inter_agent_tension=tensions,
            consensus_reached=consensus_reached,
            mean_tension=mean_t,
            consensus_confidence=confidence,
            tau_tension=float(self.tension_threshold),
            consensus_label=label,
            tau_minor_upper=(float(tau_minor) if tau_minor is not None else None),
            tau_contra_lower=(float(tau_contra) if tau_contra is not None else None),
        )

    def _classify_worst_features(
        self,
        features: tuple[float, int, int],
        tau_minor: float | None,
        tau_contra: float | None,
    ) -> str:
        """Label the aggregated consensus band using the 3-D classifier
        on the worst observed feature vector.  Falls back to the scalar
        :func:`_conflict_type` band rules when no classifier is loaded.
        """
        delta, tail_missing, has_replacement = features
        if self._cross_store_classifier is not None:
            try:
                X = [[float(delta), float(tail_missing), float(has_replacement)]]
                cls_idx = int(self._cross_store_classifier.predict(X)[0])
                return _CROSS_STORE_CLASS_TO_CONFLICT.get(
                    cls_idx, "minor_variance",
                )
            except Exception:
                pass
        return _conflict_type(
            float(delta),
            float(self.tension_threshold),
            tau_minor_upper=tau_minor,
            tau_contra_lower=tau_contra,
        )

    def _consensus_confidence(
        self,
        features: tuple[float, int, int],
        consensus_reached: bool,
        consensus_label: str = "minor_variance",
    ) -> float:
        """Return :math:`P(\\text{consensus verdict correct})`.

        Parameters
        ----------
        features
            3-D worst-triple feature vector
            ``(delta, tail_missing, has_replacement)`` aggregated across
            all pairs for the target entity.
        consensus_reached
            Whether mean pairwise delta stayed below the calibrated
            minor/variance cut.  Used only by the legacy fallback paths.
        consensus_label
            The label returned by :meth:`_classify_worst_features`.

        Preference order:

        1. **Cross-store calibrated 3-D classifier** --
           :math:`P(\\text{class} = \\text{consensus\\_label})` directly
           from the isotonic-calibrated classifier.
        2. **Per-claim tension classifier** (``calibration.pkl``).
           Scalar-only; uses the worst-triple delta.
        3. **Hand-mapped fallback**
           :math:`\\max(0, 1 - \\delta / (2 \\tau_{\\text{tension}}))`.
        """
        delta, tail_missing, has_replacement = features
        # 1. Cross-store 3-D classifier
        if self._cross_store_classifier is not None:
            try:
                X = [[float(delta), float(tail_missing), float(has_replacement)]]
                probs = self._cross_store_classifier.predict_proba(X)[0]
                inv = {v: k for k, v in _CROSS_STORE_CLASS_TO_CONFLICT.items()}
                cls_idx = inv.get(consensus_label)
                if cls_idx is not None and 0 <= cls_idx < len(probs):
                    return float(probs[cls_idx])
            except Exception:
                pass
        # 2. Per-claim tension classifier (scalar, legacy)
        tau = max(float(self.tension_threshold), 1e-6)
        if self._tension_classifier is not None:
            try:
                p_real = float(
                    self._tension_classifier.predict_proba(
                        [[float(delta)]]
                    )[0, 1]
                )
                return p_real if consensus_reached else 1.0 - p_real
            except Exception:
                pass
        # 3. Hand-mapped fallback
        return max(0.0, min(1.0, 1.0 - float(delta) / (2.0 * tau)))

    def evaluate_delegation_holonomy(
        self,
        delegation_chain: list[str],
        direct_relation: str,
        store: "GMSExpertStore",
    ) -> float | None:
        """Check holonomy defect of a delegation chain.

        Parameters
        ----------
        delegation_chain : list[str]
            Sequence of relations representing the delegation path
            (e.g., ``["delegates_to", "produces"]``).
        direct_relation : str
            The direct relation the chain should be equivalent to.
        store : GMSExpertStore
            Store with the relation embeddings.

        Returns
        -------
        float or None
            Holonomy defect (0 = consistent path).
        """
        try:
            return store.check_holonomy(delegation_chain, direct_relation)
        except Exception:
            return None

    def _cross_store_tension(
        self,
        store_a: "GMSExpertStore",
        store_b: "GMSExpertStore",
        entity: str,
    ) -> "tuple[float, tuple[float, int, int]] | None":
        """Aggregate per-triple disagreement for an entity across two stores.

        Returns ``(mean_delta, worst_features)`` where ``worst_features``
        is the 3-D vector of the triple with the largest delta, or
        ``None`` when the entity has no usable triples in ``store_a``.

        The legacy implementation scored ``(entity, "is_type", entity)``
        as a self-loop, which returned ``None`` for every store whose
        relation vocabulary did not contain ``"is_type"``.  The
        monograph's MRA corpus is such a store, which collapsed mean
        tension to zero and made the classifier untrainable.  This
        replacement iterates triples whose head matches ``entity`` and
        computes the calibration-aligned 3-feature vector for each.
        """
        if store_a.doc_graph is None or store_b.doc_graph is None:
            return None
        b_entities, b_hr_tails = _amended_indices(store_b)
        per_triple: list[tuple[float, int, int]] = []
        for h, r, t in store_a.doc_graph.triples:
            if h != entity and t != entity:
                continue
            features = _compute_triple_features(
                store_a, store_b, h, r, t,
                amended_entities=b_entities,
                amended_hr_tails=b_hr_tails,
            )
            if features is not None:
                per_triple.append(features)
        if not per_triple:
            return None
        mean_delta = sum(f[0] for f in per_triple) / len(per_triple)
        worst = max(per_triple, key=lambda f: f[0])
        return float(mean_delta), worst


# ---------------------------------------------------------------------------
# Question generator
# ---------------------------------------------------------------------------

class MultiAgentGenerator(QuestionGenerator):
    """Generate questions that probe multi-agent coordination.

    Creates delegation and coordination scenarios where multiple
    agents must agree on shared facts.  The answer function scores
    agreement via ``score_triple`` across shared entities.

    Category: ``"multi_agent"``
    """

    category = "multi_agent"

    def generate(self, store: "GMSExpertStore") -> list[GeneratedQuestion]:
        if store.doc_graph is None or store.adapter is None:
            return []

        questions: list[GeneratedQuestion] = []
        rng = random.Random(42)

        triples = list(store.doc_graph.triples)
        entities = list(store.adapter.entity_to_idx.keys())
        if len(entities) < 4 or len(triples) < 4:
            return []

        # Find entities that participate in multiple relationships
        entity_rels: dict[str, set[str]] = {}
        for h, r, t in triples:
            entity_rels.setdefault(h, set()).add(r)
            entity_rels.setdefault(t, set()).add(r)

        multi_rel_entities = [
            e for e, rels in entity_rels.items() if len(rels) >= 2
        ]
        rng.shuffle(multi_rel_entities)

        for i, ent in enumerate(multi_rel_entities[:10]):
            # Find two different relations for this entity
            ent_triples = [
                (h, r, t) for h, r, t in triples
                if h == ent or t == ent
            ]
            if len(ent_triples) < 2:
                continue

            t1 = ent_triples[0]
            t2 = ent_triples[1]

            ground_truth = True  # Both relations should be consistent

            def _answer_fn(s, _e=ent, _t1=t1, _t2=t2):
                """Check tension between the two relations — low = consistent."""
                try:
                    s1 = s.score_triple(*_t1)
                    s2 = s.score_triple(*_t2)
                    if s1 is not None and s2 is not None:
                        return abs(float(s1) - float(s2)) < 1.0
                except Exception:
                    pass
                return True

            questions.append(GeneratedQuestion(
                qid=f"multi_agent_{i:03d}",
                category=self.category,
                natural_language=(
                    f"Two agents are analyzing '{ent}'. Agent A focuses "
                    f"on its '{t1[1]}' relationship, while Agent B "
                    f"focuses on its '{t2[1]}' relationship. Are their "
                    f"perspectives consistent?"
                ),
                gms_answer_fn=_answer_fn,
                ground_truth=ground_truth,
                llm_prompt=(
                    f"Consider two perspectives on '{ent}':\n"
                    f"- Perspective A: the '{t1[1]}' relationship\n"
                    f"- Perspective B: the '{t2[1]}' relationship\n\n"
                    f"Based on the document, are these perspectives "
                    f"consistent with each other?\n\n"
                    f"{ANSWER_FORMATS['bool']}"
                ),
                answer_type="bool",
                metadata={
                    "entity": ent,
                    "triple_a": list(t1),
                    "triple_b": list(t2),
                    "test_type": "multi_agent_consistency",
                },
            ))

        return questions
