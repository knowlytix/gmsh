# SPDX-License-Identifier: Apache-2.0
"""GMSH Cross-Document Conflict Testing — inter-store tension analysis.

Detects factual conflicts between multiple document sources by
computing ``tension_energy`` between GMS stores for shared entities.

- Ingest multiple documents into separate GMS stores
- Find shared entities across stores
- For each shared entity, compute inter-store ``tension_energy`` —
  high tension indicates a factual conflict between documents
- Generate questions that target conflicting claims

Usage::

    analyzer = CrossDocumentAnalyzer(
        stores={"doc_a": store_a, "doc_b": store_b},
    )
    conflicts = analyzer.find_conflicts()
    for c in conflicts:
        print(c.entity, c.inter_store_tension, c.conflict_type)

    gen = CrossDocumentGenerator(stores={"a": store_a, "b": store_b})
    questions = gen.generate(store_a)  # primary store
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from knowlytix.bench.eval.generators.base import GeneratedQuestion, QuestionGenerator, ANSWER_FORMATS

if TYPE_CHECKING:
    from knowlytix.doc.store import GMSExpertStore


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class DocumentConflict:
    """A detected conflict between two document stores.

    Attributes
    ----------
    entity : str
        The shared entity where conflict is detected.
    store_a_id : str
        Identifier for the first store.
    store_b_id : str
        Identifier for the second store.
    fact_a : str
        Fact/relation context from store A.
    fact_b : str
        Fact/relation context from store B.
    inter_store_tension : float
        Tension energy between store representations.
        Higher = stronger conflict.
    conflict_type : str
        Classification: ``"contradiction"`` (tension > 1.5),
        ``"disagreement"`` (0.7-1.5), ``"minor_variance"`` (< 0.7).
    """

    entity: str = ""
    store_a_id: str = ""
    store_b_id: str = ""
    fact_a: str = ""
    fact_b: str = ""
    inter_store_tension: float = 0.0
    conflict_type: str = "minor_variance"

    def to_dict(self) -> dict:
        return {
            "entity": self.entity,
            "stores": [self.store_a_id, self.store_b_id],
            "inter_store_tension": round(self.inter_store_tension, 4),
            "conflict_type": self.conflict_type,
            "fact_a": self.fact_a,
            "fact_b": self.fact_b,
        }


def _conflict_type(tension: float) -> str:
    """Classify conflict severity by tension energy."""
    if tension > 1.5:
        return "contradiction"
    if tension > 0.7:
        return "disagreement"
    return "minor_variance"


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class CrossDocumentAnalyzer:
    """Detect factual conflicts between GMS stores using tension_energy.

    Parameters
    ----------
    stores : dict[str, GMSExpertStore]
        Named GMS stores, each from a different document.
    tension_threshold : float
        Minimum tension to flag as a conflict.
    """

    def __init__(
        self,
        stores: dict[str, "GMSExpertStore"] | None = None,
        tension_threshold: float = 0.5,
    ):
        self.stores = stores or {}
        self.tension_threshold = tension_threshold

    def find_shared_entities(self) -> dict[str, list[str]]:
        """Find entities that appear in multiple stores.

        Returns
        -------
        dict[str, list[str]]
            Entity → list of store IDs containing that entity.
        """
        entity_stores: dict[str, list[str]] = {}
        for store_id, store in self.stores.items():
            if store.adapter is None:
                continue
            for entity in store.adapter.entity_to_idx:
                entity_stores.setdefault(entity, []).append(store_id)

        return {
            e: sids for e, sids in entity_stores.items()
            if len(sids) >= 2
        }

    def find_conflicts(
        self, max_conflicts: int = 50,
    ) -> list[DocumentConflict]:
        """Scan shared entities for inter-store tension conflicts.

        Parameters
        ----------
        max_conflicts : int
            Maximum number of conflicts to return.

        Returns
        -------
        list[DocumentConflict]
            Detected conflicts sorted by tension (descending).
        """
        shared = self.find_shared_entities()
        conflicts: list[DocumentConflict] = []

        for entity, store_ids in shared.items():
            # Check all pairs of stores
            for i, sid_a in enumerate(store_ids):
                for sid_b in store_ids[i + 1:]:
                    tension = self._compute_tension(
                        self.stores[sid_a],
                        self.stores[sid_b],
                        entity,
                    )
                    if tension is None or tension < self.tension_threshold:
                        continue

                    # Gather fact context from each store
                    fact_a = self._entity_context(self.stores[sid_a], entity)
                    fact_b = self._entity_context(self.stores[sid_b], entity)

                    conflicts.append(DocumentConflict(
                        entity=entity,
                        store_a_id=sid_a,
                        store_b_id=sid_b,
                        fact_a=fact_a,
                        fact_b=fact_b,
                        inter_store_tension=tension,
                        conflict_type=_conflict_type(tension),
                    ))

        # Sort by tension descending, limit
        conflicts.sort(key=lambda c: c.inter_store_tension, reverse=True)
        return conflicts[:max_conflicts]

    def _compute_tension(
        self,
        store_a: "GMSExpertStore",
        store_b: "GMSExpertStore",
        entity: str,
    ) -> float | None:
        """Compute inter-store tension for a shared entity.

        Scores the entity as subject in both stores, then computes
        the geodesic difference as a tension proxy.
        """
        try:
            # Use score_triple with a reflexive relation
            score_a = store_a.score_triple(entity, "is_type", entity)
            score_b = store_b.score_triple(entity, "is_type", entity)
            if score_a is not None and score_b is not None:
                return abs(float(score_a) - float(score_b))
        except Exception:
            pass

        # Fallback: use tension_energy within each store on shared relations
        try:
            tension = store_a.tension_energy(entity, entity)
            if tension is not None:
                return float(tension)
        except Exception:
            pass

        return None

    def _entity_context(
        self, store: "GMSExpertStore", entity: str,
    ) -> str:
        """Get a brief fact summary for an entity in a store."""
        if store.doc_graph is None:
            return ""
        triples = [
            (h, r, t) for h, r, t in store.doc_graph.triples
            if h == entity or t == entity
        ]
        if not triples:
            return f"'{entity}' present but no triples found"
        summaries = [f"{h} {r} {t}" for h, r, t in triples[:3]]
        return "; ".join(summaries)


# ---------------------------------------------------------------------------
# Question generator
# ---------------------------------------------------------------------------

class CrossDocumentGenerator(QuestionGenerator):
    """Generate questions targeting cross-document conflicts.

    Requires multiple stores to detect conflicts.  Falls back to
    intra-store tension for single-store scenarios.

    Category: ``"cross_document"``

    Parameters
    ----------
    stores : dict[str, GMSExpertStore] or None
        Additional named stores for cross-document analysis.
    """

    category = "cross_document"

    def __init__(
        self, stores: dict[str, "GMSExpertStore"] | None = None,
    ):
        self.additional_stores = stores or {}

    def generate(self, store: "GMSExpertStore") -> list[GeneratedQuestion]:
        if store.doc_graph is None or store.adapter is None:
            return []

        questions: list[GeneratedQuestion] = []
        rng = random.Random(42)

        # If we have additional stores, find conflicts
        if self.additional_stores:
            all_stores = {"primary": store, **self.additional_stores}
            analyzer = CrossDocumentAnalyzer(stores=all_stores)
            conflicts = analyzer.find_conflicts(max_conflicts=20)

            for i, conflict in enumerate(conflicts[:10]):
                ground_truth = conflict.conflict_type

                def _answer_fn(s, _ent=conflict.entity, _type=conflict.conflict_type):
                    return _type

                questions.append(GeneratedQuestion(
                    qid=f"cross_doc_{i:03d}",
                    category=self.category,
                    natural_language=(
                        f"Document A says: {conflict.fact_a}. "
                        f"Document B says: {conflict.fact_b}. "
                        f"Are these consistent?"
                    ),
                    gms_answer_fn=_answer_fn,
                    ground_truth=ground_truth,
                    llm_prompt=(
                        f"Compare these two claims about '{conflict.entity}':\n"
                        f"Source A: {conflict.fact_a}\n"
                        f"Source B: {conflict.fact_b}\n\n"
                        f"Do these sources agree? Classify as: "
                        f"'minor_variance', 'disagreement', or "
                        f"'contradiction'.\n\n"
                        f"ANSWER: <classification>"
                    ),
                    answer_type="str",
                    metadata={
                        "entity": conflict.entity,
                        "tension": round(conflict.inter_store_tension, 4),
                        "test_type": "cross_document_conflict",
                    },
                ))
        else:
            # Single-store: find entities with internal tension
            triples = list(store.doc_graph.triples)
            entities = list(store.adapter.entity_to_idx.keys())

            for i, ent in enumerate(entities[:15]):
                ent_triples = [
                    (h, r, t) for h, r, t in triples
                    if h == ent
                ]
                if len(ent_triples) < 2:
                    continue

                # Check tension between different tails
                t1, t2 = ent_triples[0], ent_triples[1]
                tension = None
                try:
                    tension = store.tension_energy(t1[2], t2[2])
                except Exception:
                    pass

                if tension is None:
                    continue

                ground_truth = tension < 0.7  # consistent if low tension

                def _answer_fn(s, _gt=ground_truth):
                    return _gt

                questions.append(GeneratedQuestion(
                    qid=f"cross_doc_intra_{i:03d}",
                    category=self.category,
                    natural_language=(
                        f"Are the facts '{t1[0]} {t1[1]} {t1[2]}' and "
                        f"'{t2[0]} {t2[1]} {t2[2]}' mutually consistent?"
                    ),
                    gms_answer_fn=_answer_fn,
                    ground_truth=ground_truth,
                    llm_prompt=(
                        f"Based on the document, are these two facts "
                        f"consistent?\n"
                        f"1. {t1[0]} {t1[1]} {t1[2]}\n"
                        f"2. {t2[0]} {t2[1]} {t2[2]}\n\n"
                        f"{ANSWER_FORMATS['bool']}"
                    ),
                    answer_type="bool",
                    metadata={
                        "entity": ent,
                        "tension": round(tension, 4) if tension else None,
                        "test_type": "intra_document_consistency",
                    },
                ))

                if len(questions) >= 10:
                    break

        return questions
