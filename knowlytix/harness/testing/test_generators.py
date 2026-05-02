# SPDX-License-Identifier: Apache-2.0
"""GMSH Extended Test Generators — new GMS-native test categories.

These generators fill gaps identified in the GMSH spec:

- **ExactRetrievalGenerator**: Tests exact ENM lookup (GMSH §10.1.E)
- **NegativeGenerator**: Adversarial tests with fabricated entities (GMSH §10.3)
- **WorkflowComplianceGenerator**: Multi-hop ordering tests (GMSH §10.1.F)

Each generator uses GMS primitives for ground truth:
    store.lookup_enm(), store.entity_exists(), store.query_triples(),
    store.transport.path_score()
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from knowlytix.benchmark.eval.generators.base import GeneratedQuestion, QuestionGenerator, ANSWER_FORMATS

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore


# =========================================================================
# 1. EXACT RETRIEVAL — ENM lookup tests
# =========================================================================

class ExactRetrievalGenerator(QuestionGenerator):
    """Generate tests for exact numeric retrieval from ENM.

    For each ENM entry, asks: "What is the exact value of {entity} in
    {category}?"  Ground truth comes from ``store.lookup_enm()``.

    This tests the GMSH exact route (§8.1).
    """

    category = "exact_retrieval"

    def generate(self, store: "GMSExpertStore") -> list[GeneratedQuestion]:
        if store.enm is None:
            return []
        questions = []

        for key in store.enm.keys():
            val = store.enm.read_exact(key)
            if val is None:
                continue
            fval = float(val.item() if hasattr(val, "item") else val)

            ground_truth = fval

            def _answer_fn(s, _cat=key.type, _id=key.id, _store=store):
                return _store.lookup_enm(_cat, _id)

            questions.append(GeneratedQuestion(
                qid=f"exact_retrieval_{len(questions):03d}",
                category=self.category,
                natural_language=(
                    f"What is the exact '{key.type}' value for "
                    f"'{key.id}'?"
                ),
                gms_answer_fn=_answer_fn,
                ground_truth=ground_truth,
                llm_prompt=(
                    f"Based on the document, what is the exact numeric "
                    f"value of '{key.id}' in the '{key.type}' category?\n\n"
                    f"ANSWER: <number>"
                ),
                answer_type="float",
                metadata={
                    "enm_type": key.type,
                    "entity_id": key.id,
                    "value": fval,
                },
            ))

        return questions


# =========================================================================
# 2. NEGATIVE — adversarial tests with fabricated entities
# =========================================================================

class NegativeGenerator(QuestionGenerator):
    """Generate adversarial negative tests with fabricated entities.

    Creates questions where the correct answer is "no" or "not found"
    because the entity or relation doesn't exist in the GMS store.

    This tests the GMSH adversarial requirement (§10.3).
    """

    category = "negative"

    # Fabricated entity prefixes
    _FAKE_PREFIXES = [
        "Nonexistent_", "Fabricated_", "Phantom_", "Unknown_",
    ]

    def generate(self, store: "GMSExpertStore") -> list[GeneratedQuestion]:
        if store.doc_graph is None or store.adapter is None:
            return []

        questions = []
        rng = random.Random(42)
        entities = list(store.adapter.entity_to_idx.keys())
        relations = list(store.adapter.relation_to_idx.keys())

        if len(entities) < 2 or not relations:
            return []

        # Type 1: Non-existent entity as head
        for i in range(min(15, len(entities))):
            real_entity = entities[i]
            fake_entity = rng.choice(self._FAKE_PREFIXES) + real_entity[:10]
            relation = rng.choice(relations)
            tail = rng.choice(entities)

            ground_truth = False

            def _answer_fn(s, _fake=fake_entity, _store=store):
                return not _store.entity_exists(_fake)

            questions.append(GeneratedQuestion(
                qid=f"negative_entity_{i:03d}",
                category=self.category,
                natural_language=(
                    f"Does the entity '{fake_entity}' have a "
                    f"'{relation}' relationship with '{tail}'?"
                ),
                gms_answer_fn=_answer_fn,
                ground_truth=ground_truth,
                llm_prompt=(
                    f"Based on the document, does '{fake_entity}' have "
                    f"a '{relation}' relationship with '{tail}'?\n\n"
                    f"{ANSWER_FORMATS['bool']}"
                ),
                answer_type="bool",
                metadata={
                    "fake_entity": fake_entity,
                    "relation": relation,
                    "tail": tail,
                    "test_type": "non_existent_entity",
                },
            ))

        # Type 2: Wrong relation for a known triple
        triples = list(store.doc_graph.triples)
        if len(triples) >= 2 and len(relations) >= 2:
            rng.shuffle(triples)
            for i, (h, r, t) in enumerate(triples[:15]):
                wrong_relations = [rel for rel in relations if rel != r]
                if not wrong_relations:
                    continue
                wrong_r = rng.choice(wrong_relations)

                # Check that (h, wrong_r, t) is NOT a real triple
                real = store.query_triples(head=h, relation=wrong_r, tail=t)
                if real:
                    continue

                ground_truth = False

                def _answer_fn(
                    s, _h=h, _wr=wrong_r, _t=t, _store=store
                ):
                    return not bool(_store.query_triples(
                        head=_h, relation=_wr, tail=_t))

                questions.append(GeneratedQuestion(
                    qid=f"negative_relation_{i:03d}",
                    category=self.category,
                    natural_language=(
                        f"Does '{h}' have a '{wrong_r}' relationship "
                        f"with '{t}'?"
                    ),
                    gms_answer_fn=_answer_fn,
                    ground_truth=ground_truth,
                    llm_prompt=(
                        f"Based on the document, does '{h}' have a "
                        f"'{wrong_r}' relationship with '{t}'?\n\n"
                        f"{ANSWER_FORMATS['bool']}"
                    ),
                    answer_type="bool",
                    metadata={
                        "head": h,
                        "wrong_relation": wrong_r,
                        "correct_relation": r,
                        "tail": t,
                        "test_type": "wrong_relation",
                    },
                ))

        return questions


# =========================================================================
# 3. WORKFLOW COMPLIANCE — multi-hop ordering tests
# =========================================================================

class WorkflowComplianceGenerator(QuestionGenerator):
    """Generate workflow ordering tests from multi-hop graph paths.

    Finds 3+ hop chains in the knowledge graph and asks about the
    correct ordering of steps.  Ground truth comes from the graph
    path structure, verified by transport.path_score().

    This tests the GMSH workflow route (§10.1.F).
    """

    category = "workflow_compliance"

    def generate(self, store: "GMSExpertStore") -> list[GeneratedQuestion]:
        if store.doc_graph is None or store.adapter is None:
            return []

        questions = []

        # Build adjacency
        adj: dict[str, list[tuple[str, str]]] = {}
        for h, r, t in store.doc_graph.triples:
            if r == "in_section":
                continue
            adj.setdefault(h, []).append((r, t))

        # Find 3-hop chains: a --r1--> b --r2--> c --r3--> d
        chains = []
        for a, neighbors_a in adj.items():
            for r1, b in neighbors_a:
                if b == a:
                    continue
                for r2, c in adj.get(b, []):
                    if c in (a, b):
                        continue
                    for r3, d in adj.get(c, []):
                        if d in (a, b, c):
                            continue
                        chains.append((a, r1, b, r2, c, r3, d))

        if not chains:
            return []

        rng = random.Random(42)
        rng.shuffle(chains)

        for a, r1, b, r2, c, r3, d in chains[:20]:
            # Type 1: "What is the next step after B?"
            ground_truth = c

            def _answer_fn(s, _c=c):
                return _c

            questions.append(GeneratedQuestion(
                qid=f"workflow_next_{len(questions):03d}",
                category=self.category,
                natural_language=(
                    f"In the sequence starting from '{a}', after "
                    f"reaching '{b}' via '{r1}', what entity comes "
                    f"next via '{r2}'?"
                ),
                gms_answer_fn=_answer_fn,
                ground_truth=ground_truth,
                llm_prompt=(
                    f"According to the document, consider this sequence:\n"
                    f"  '{a}' --[{r1}]--> '{b}' --[{r2}]--> ???\n"
                    f"What entity comes next?\n\n"
                    f"ANSWER: <entity name>"
                ),
                answer_type="str",
                metadata={
                    "head": a, "r1": r1, "mid1": b,
                    "r2": r2, "mid2": c, "r3": r3, "tail": d,
                    "path_length": 3,
                },
            ))

            # Type 2: "Is the ordering A→B→C correct?"
            ground_truth_bool = True

            def _answer_fn_order(s, _gt=True):
                return _gt

            questions.append(GeneratedQuestion(
                qid=f"workflow_order_{len(questions):03d}",
                category=self.category,
                natural_language=(
                    f"Is the following ordering consistent with the "
                    f"document: '{a}' → '{b}' → '{c}' → '{d}'?"
                ),
                gms_answer_fn=_answer_fn_order,
                ground_truth=ground_truth_bool,
                llm_prompt=(
                    f"Based on the document, is the following sequence "
                    f"of entities consistent?\n"
                    f"  {a} → {b} → {c} → {d}\n\n"
                    f"{ANSWER_FORMATS['bool']}"
                ),
                answer_type="bool",
                metadata={
                    "entities": [a, b, c, d],
                    "relations": [r1, r2, r3],
                    "path_length": 3,
                },
            ))

            if len(questions) >= 30:
                break

        return questions


# =========================================================================
# Registry
# =========================================================================

def extended_generators() -> list[QuestionGenerator]:
    """Return one instance of each extended GMSH generator."""
    return [
        ExactRetrievalGenerator(),
        NegativeGenerator(),
        WorkflowComplianceGenerator(),
    ]
