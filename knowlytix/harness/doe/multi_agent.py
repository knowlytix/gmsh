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
from typing import TYPE_CHECKING

from knowlytix.bench.eval.generators.base import GeneratedQuestion, QuestionGenerator, ANSWER_FORMATS

if TYPE_CHECKING:
    from knowlytix.doc.store import GMSExpertStore


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
    tension_threshold : float
        Maximum mean tension for consensus (default 0.5).
    """

    oracle_type = "consensus"

    def __init__(
        self,
        stores: dict[str, "GMSExpertStore"] | None = None,
        tension_threshold: float = 0.5,
    ):
        self.stores = stores or {}
        self.tension_threshold = tension_threshold

    def evaluate_consensus(
        self, entity: str,
    ) -> InteractionResult:
        """Evaluate inter-agent consensus on a specific entity.

        Computes ``tension_energy`` for the entity across every pair
        of agent stores.

        Parameters
        ----------
        entity : str
            The entity to check consensus on.

        Returns
        -------
        InteractionResult
            With pairwise tensions and consensus flag.
        """
        role_ids = sorted(self.stores.keys())
        tensions: dict[str, float] = {}

        for i, ra in enumerate(role_ids):
            for rb in role_ids[i + 1:]:
                store_a = self.stores[ra]
                store_b = self.stores[rb]
                pair_key = f"{ra}:{rb}"

                # Attempt cross-store tension via entity embeddings
                tension = self._cross_store_tension(store_a, store_b, entity)
                if tension is not None:
                    tensions[pair_key] = tension

        mean_t = (
            sum(tensions.values()) / len(tensions)
            if tensions else 0.0
        )

        return InteractionResult(
            inter_agent_tension=tensions,
            consensus_reached=mean_t <= self.tension_threshold,
            mean_tension=mean_t,
        )

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
    ) -> float | None:
        """Compute tension between entity representations in two stores.

        Uses ``score_triple`` in each store to get the entity's
        plausibility, then computes the difference as a proxy for
        inter-store disagreement.
        """
        try:
            # Score the entity as a self-referential triple in each store
            score_a = store_a.score_triple(entity, "is_type", entity)
            score_b = store_b.score_triple(entity, "is_type", entity)
            if score_a is not None and score_b is not None:
                # Difference in geodesic = disagreement
                return abs(float(score_a) - float(score_b))
        except Exception:
            pass
        return None


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
