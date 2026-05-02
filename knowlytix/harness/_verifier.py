# SPDX-License-Identifier: Apache-2.0
"""LogicVerifier: Domain-GMS geometric verification of proposed actions + findings.

Public home (GMS-171) — relocated from the private
`agents.ds_agent.agents.base.LogicVerifier` class so `knowlytix-harness`
imports cleanly from the mirror's public cut-list. Internal callers
keep working via the re-export at the top of
`agents/ds_agent/agents/base.py` in the private monorepo.

Verification levels:
1. Action plausibility  — is (agent, should_call, tool) geometrically close?
2. Triple plausibility  — is a proposed domain triple consistent with existing knowledge?
3. Contradiction gate   — does a proposed triple introduce tension > tau_contra?
4. Holonomy consistency — are relation paths logically consistent?

Scores are geometric distances on S^{m-1}. Lower = more plausible.
Tension energy: 0=agree, sqrt(2)=irrelevant, 2=contradict.
"""

from __future__ import annotations

from knowlytix.knowledge.config import VerifyConfig
from knowlytix.knowledge.extract import ExtractedTriple
from knowlytix.knowledge.store import GMSExpertStore


class LogicVerifier:
    """Uses Domain GMS geometry to verify proposed actions and findings."""

    def __init__(self, store: GMSExpertStore, verify_config: VerifyConfig | None = None):
        self.store = store
        self.config = verify_config or VerifyConfig()

    # ── Action verification ──

    def rank_tools(self, context_entity: str, relation: str = "should_call",
                   top_k: int = 5) -> list[tuple[str, float]]:
        """Rank candidate tools by geodesic distance from context.

        Uses link_predict: (context_entity, should_call, ?) → ranked tools.
        Lower score = more plausible action.
        """
        return self.store.link_predict(context_entity, relation, top_k)

    def verify_action(self, context_entity: str, tool_name: str,
                      relation: str = "should_call") -> dict:
        """Verify a proposed tool call is geometrically plausible.

        Returns:
            {"plausible": bool, "score": float|None, "reason": str}
        """
        score = self.store.score_triple(context_entity, relation, tool_name)
        if score is None:
            return {"plausible": True, "score": None, "reason": "novel_action"}

        plausible = score <= self.config.min_plausibility
        reason = "plausible" if plausible else "implausible"
        return {"plausible": plausible, "score": score, "reason": reason}

    # ── Triple verification ──

    def verify_triple(self, head: str, relation: str, tail: str) -> dict:
        """Score plausibility of a proposed domain triple.

        Returns:
            {"plausible": bool, "score": float|None, "reason": str}
        """
        score = self.store.score_triple(head, relation, tail)
        if score is None:
            return {"plausible": True, "score": None, "reason": "novel_triple"}

        plausible = score <= self.config.min_plausibility
        reason = "plausible" if plausible else "implausible"
        return {"plausible": plausible, "score": score, "reason": reason}

    def check_tension(self, entity_a: str, entity_b: str) -> dict:
        """Check tension energy between two entities.

        Returns:
            {"contradicts": bool, "tension": float|None, "level": str}
        """
        te = self.store.tension_energy(entity_a, entity_b)
        if te is None:
            return {"contradicts": False, "tension": None, "level": "unknown"}

        if te > self.config.tau_contra:
            level = "contradiction"
        elif te > 1.414:
            level = "irrelevant"
        elif te > self.config.tau_ent:
            level = "weak"
        else:
            level = "agreement"

        return {
            "contradicts": te > self.config.tau_contra,
            "tension": round(te, 4),
            "level": level,
        }

    def check_consistency(self, relation_path: list[str],
                          direct_relation: str) -> dict:
        """Check holonomy consistency of a relation path.

        Returns:
            {"consistent": bool|None, "defect": float|None}
        """
        defect = self.store.check_holonomy(relation_path, direct_relation)
        if defect is None:
            return {"consistent": None, "defect": None}

        consistent = defect <= self.config.tau_path
        return {"consistent": consistent, "defect": round(defect, 4)}

    # ── Batch verification for domain triples ──

    def verify_domain_triples(self, triples: list[ExtractedTriple]) -> dict:
        """Verify a batch of proposed domain triples before learning.

        Returns:
            {"accepted": list[ExtractedTriple],
             "rejected": list[dict],
             "n_accepted": int, "n_rejected": int}
        """
        accepted = []
        rejected = []

        for t in triples:
            v = self.verify_triple(t.head, t.relation, t.tail)

            if not v["plausible"] and v["score"] is not None:
                rejected.append({
                    "triple": (t.head, t.relation, t.tail),
                    "reason": "implausible",
                    "score": v["score"],
                })
                continue

            te = self.check_tension(t.head, t.tail)
            if te["contradicts"]:
                rejected.append({
                    "triple": (t.head, t.relation, t.tail),
                    "reason": "contradiction",
                    "tension": te["tension"],
                })
                continue

            accepted.append(t)

        return {
            "accepted": accepted,
            "rejected": rejected,
            "n_accepted": len(accepted),
            "n_rejected": len(rejected),
        }
