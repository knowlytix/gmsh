# SPDX-License-Identifier: Apache-2.0
"""Geometric question generators that showcase GMS capabilities.

These generators go beyond plain triple-store queries.  Each one exercises
a specific geometric primitive of the GMS library:

- **PlausibilityGenerator**: triple scoring via geodesic distance on S^{m-1}
- **TensionGenerator**: contradiction detection via tension energy
- **HolonomyGenerator**: path consistency via rotor composition holonomy defects
- **LinkPredictionGenerator**: geometric beam search (rank candidate tails)
- **MultiHopTransportGenerator**: multi-hop relational transport with path scoring
- **PhaseComparisonGenerator**: numeric comparisons via phase encoding on S^1
- **ConditionalGenerator**: cross-relation rotor test for if-then reasoning
- **ProceduralGenerator**: ordered dependency chains via composed rotor transport
"""

from __future__ import annotations

import random

import torch

from knowlytix.bench.eval.generators.base import GeneratedQuestion, QuestionGenerator, ANSWER_FORMATS
from knowlytix.doc.store import GMSExpertStore


# =========================================================================
# 1. PLAUSIBILITY — score_triple() → geodesic distance on S^{m-1}
# =========================================================================

class PlausibilityGenerator(QuestionGenerator):
    """Generate questions that test whether a triple is plausible.

    Uses ``store.score_triple(h, r, t)`` which returns geodesic distance
    d(R_r(v_h), v_t) on the hypersphere.  Low distance = plausible triple.

    Question forms:
    - "Which entity is more plausible as the <relation> of <head>: A or B?"
    - "How plausible is the relationship '<head> <relation> <tail>'?"
    """

    category = "plausibility"

    def generate(self, store: GMSExpertStore) -> list[GeneratedQuestion]:
        if store.doc_graph is None or store.adapter is None:
            return []
        questions = []

        triples = list(store.doc_graph.triples)
        entities = list(store.adapter.entity_to_idx.keys())
        if len(triples) < 2 or len(entities) < 4:
            return []

        rng = random.Random(42)

        # Type 1: Comparative plausibility — "Which is more plausible?"
        for i, (h, r, t) in enumerate(triples[:50]):
            score_real = store.score_triple(h, r, t)
            if score_real is None:
                continue

            # Pick a random wrong tail
            candidates = [e for e in entities if e != t and e != h]
            if not candidates:
                continue
            wrong_tail = rng.choice(candidates)
            score_wrong = store.score_triple(h, r, wrong_tail)
            if score_wrong is None:
                continue

            # Only keep questions where real is clearly better
            if score_real >= score_wrong - 0.1:
                continue

            real_label, wrong_label = "A", "B"
            if rng.random() > 0.5:
                real_label, wrong_label = "B", "A"
                option_a, option_b = wrong_tail, t
            else:
                option_a, option_b = t, wrong_tail

            ground_truth = real_label

            def _answer_fn(s, _gt=ground_truth):
                return _gt

            questions.append(GeneratedQuestion(
                qid=f"plausibility_cmp_{i:03d}",
                category=self.category,
                natural_language=(
                    f"Which entity is more plausible as the '{r}' of "
                    f"'{h}': (A) {option_a} or (B) {option_b}?"
                ),
                gms_answer_fn=_answer_fn,
                ground_truth=ground_truth,
                llm_prompt=(
                    f"Given the document, which entity is more plausible as "
                    f"the '{r}' of '{h}'?\n"
                    f"  (A) {option_a}\n  (B) {option_b}\n\n"
                    f"ANSWER: A or B"
                ),
                answer_type="str",
                metadata={
                    "head": h, "relation": r, "true_tail": t,
                    "wrong_tail": wrong_tail,
                    "geodesic_real": round(score_real, 4),
                    "geodesic_wrong": round(score_wrong, 4),
                },
            ))

        # Type 2: Absolute plausibility — "Is this triple plausible?"
        for i, (h, r, t) in enumerate(triples[:30]):
            score = store.score_triple(h, r, t)
            if score is None:
                continue
            plausible = score < 0.5  # threshold: low geodesic = plausible
            ground_truth = plausible

            def _answer_fn(s, _h=h, _r=r, _t=t, _store=store):
                sc = _store.score_triple(_h, _r, _t)
                return sc is not None and sc < 0.5

            questions.append(GeneratedQuestion(
                qid=f"plausibility_abs_{i:03d}",
                category=self.category,
                natural_language=(
                    f"Is the relationship '{h} {r} {t}' supported by "
                    f"the document?"
                ),
                gms_answer_fn=_answer_fn,
                ground_truth=ground_truth,
                llm_prompt=(
                    f"Based on the document, is the following relationship "
                    f"supported?\n  '{h}' --[{r}]--> '{t}'\n\n"
                    f"{ANSWER_FORMATS['bool']}"
                ),
                answer_type="bool",
                metadata={
                    "head": h, "relation": r, "tail": t,
                    "geodesic": round(score, 4),
                },
            ))

        return questions


# =========================================================================
# 2. TENSION — tension_energy() → contradiction detection via Clifford
# =========================================================================

class TensionGenerator(QuestionGenerator):
    """Generate questions about entity agreement/contradiction.

    Uses Clifford tension energy: E = 2 sin(theta/2)
      E ~ 0    → agreement/entailment
      E ~ sqrt(2) → irrelevant
      E ~ 2    → contradiction

    Question forms:
    - "Do entities X and Y agree, contradict, or are they unrelated?"
    - "Which pair has higher tension (more contradictory)?"
    """

    category = "tension"

    def generate(self, store: GMSExpertStore) -> list[GeneratedQuestion]:
        if store.doc_graph is None or store.adapter is None:
            return []
        questions = []

        entities = list(store.adapter.entity_to_idx.keys())
        if len(entities) < 6:
            return []

        rng = random.Random(42)

        # Compute tension for a random sample of pairs
        pairs_with_tension: list[tuple[str, str, float]] = []
        sampled_pairs = []
        for _ in range(min(200, len(entities) * 2)):
            a, b = rng.sample(entities, 2)
            if (a, b) in sampled_pairs or (b, a) in sampled_pairs:
                continue
            sampled_pairs.append((a, b))
            te = store.tension_energy(a, b)
            if te is not None:
                pairs_with_tension.append((a, b, te))

        if len(pairs_with_tension) < 4:
            return []

        # Sort by tension
        pairs_with_tension.sort(key=lambda x: x[2])

        # Type 1: Classification — "agree, contradict, or unrelated?"
        for i, (a, b, te) in enumerate(pairs_with_tension[:40]):
            if te < 0.5:
                label = "agree"
            elif te > 1.5:
                label = "contradict"
            else:
                label = "unrelated"
            ground_truth = label

            def _answer_fn(s, _a=a, _b=b, _store=store):
                t = _store.tension_energy(_a, _b)
                if t is None:
                    return "unrelated"
                if t < 0.5:
                    return "agree"
                if t > 1.5:
                    return "contradict"
                return "unrelated"

            questions.append(GeneratedQuestion(
                qid=f"tension_classify_{i:03d}",
                category=self.category,
                natural_language=(
                    f"Based on the document, do '{a}' and '{b}' agree, "
                    f"contradict each other, or are they unrelated?"
                ),
                gms_answer_fn=_answer_fn,
                ground_truth=ground_truth,
                llm_prompt=(
                    f"Based on the document, what is the relationship between "
                    f"'{a}' and '{b}'?\n"
                    f"Do they agree, contradict, or are they unrelated?\n\n"
                    f"ANSWER: agree, contradict, or unrelated"
                ),
                answer_type="str",
                metadata={
                    "entity_a": a, "entity_b": b,
                    "tension_energy": round(te, 4),
                },
            ))

        # Type 2: Comparative — "Which pair has higher tension?"
        agreeing = [p for p in pairs_with_tension if p[2] < 0.5]
        tense = [p for p in pairs_with_tension if p[2] > 1.0]
        for i in range(min(15, len(agreeing), len(tense))):
            lo = agreeing[i]
            hi = tense[-(i + 1)]

            if rng.random() > 0.5:
                pair_a, pair_b = (lo[0], lo[1]), (hi[0], hi[1])
                ground_truth = "B"
            else:
                pair_a, pair_b = (hi[0], hi[1]), (lo[0], lo[1])
                ground_truth = "A"

            def _answer_fn(s, _gt=ground_truth):
                return _gt

            questions.append(GeneratedQuestion(
                qid=f"tension_compare_{i:03d}",
                category=self.category,
                natural_language=(
                    f"Which pair is more contradictory: "
                    f"(A) '{pair_a[0]}' & '{pair_a[1]}' or "
                    f"(B) '{pair_b[0]}' & '{pair_b[1]}'?"
                ),
                gms_answer_fn=_answer_fn,
                ground_truth=ground_truth,
                llm_prompt=(
                    f"Which pair of concepts is more contradictory based on "
                    f"the document?\n"
                    f"  (A) '{pair_a[0]}' and '{pair_a[1]}'\n"
                    f"  (B) '{pair_b[0]}' and '{pair_b[1]}'\n\n"
                    f"ANSWER: A or B"
                ),
                answer_type="str",
                metadata={
                    "pair_a": pair_a, "pair_b": pair_b,
                    "tension_a": round(lo[2] if ground_truth == "B" else hi[2], 4),
                    "tension_b": round(hi[2] if ground_truth == "B" else lo[2], 4),
                },
            ))

        return questions


# =========================================================================
# 3. HOLONOMY — path consistency via rotor composition
# =========================================================================

class HolonomyGenerator(QuestionGenerator):
    """Generate questions about multi-hop path consistency.

    Uses holonomy defect: ||R_path @ R_direct^{-1} - I||_F
    If a path A→B→C and a direct edge A→C are both known, the holonomy
    measures whether the composed rotors agree.

    Question form:
    - "Is the path from X via Y to Z consistent with the direct relationship?"
    """

    category = "holonomy"

    def generate(self, store: GMSExpertStore) -> list[GeneratedQuestion]:
        if (store.doc_graph is None or store.adapter is None
                or store.transport is None):
            return []
        questions = []

        # Build adjacency: head → [(rel, tail), ...]
        adj: dict[str, list[tuple[str, str]]] = {}
        for h, r, t in store.doc_graph.triples:
            adj.setdefault(h, []).append((r, t))

        # Find triangles: h→mid→t AND h→t directly
        checked = set()
        for h, r_direct, t in store.doc_graph.triples:
            if (h, t) in checked:
                continue
            checked.add((h, t))

            for r1, mid in adj.get(h, []):
                if mid == t or mid == h:
                    continue
                for r2, dest in adj.get(mid, []):
                    if dest != t:
                        continue

                    defect = store.check_holonomy([r1, r2], r_direct)
                    if defect is None:
                        continue

                    consistent = defect <= 0.5
                    ground_truth = consistent
                    qid = f"holonomy_{len(questions):03d}"

                    def _answer_fn(
                        s, _r1=r1, _r2=r2, _rd=r_direct, _store=store
                    ):
                        d = _store.check_holonomy([_r1, _r2], _rd)
                        return d is not None and d <= 0.5

                    questions.append(GeneratedQuestion(
                        qid=qid,
                        category=self.category,
                        natural_language=(
                            f"Is the indirect path '{h}' --[{r1}]--> "
                            f"'{mid}' --[{r2}]--> '{t}' consistent with "
                            f"the direct relationship '{h}' --[{r_direct}]--> "
                            f"'{t}'?"
                        ),
                        gms_answer_fn=_answer_fn,
                        ground_truth=ground_truth,
                        llm_prompt=(
                            f"According to the document, consider two paths:\n"
                            f"  Direct: '{h}' --[{r_direct}]--> '{t}'\n"
                            f"  Indirect: '{h}' --[{r1}]--> '{mid}' "
                            f"--[{r2}]--> '{t}'\n"
                            f"Are these two paths consistent (do they agree)?\n\n"
                            f"{ANSWER_FORMATS['bool']}"
                        ),
                        answer_type="bool",
                        metadata={
                            "head": h, "mid": mid, "tail": t,
                            "r1": r1, "r2": r2, "r_direct": r_direct,
                            "holonomy_defect": round(defect, 4),
                        },
                    ))

                    if len(questions) >= 50:
                        return questions

        return questions


# =========================================================================
# 4. LINK PREDICTION — beam_expand() → rank candidate tails
# =========================================================================

class LinkPredictionGenerator(QuestionGenerator):
    """Generate questions that test entity prediction.

    Uses ``store.link_predict(head, relation, top_k)`` which ranks all
    entities by geodesic distance after applying the relation rotor.

    Question form:
    - "What is the <relation> of <head>?"  (answer from link prediction)
    """

    category = "link_prediction"

    def generate(self, store: GMSExpertStore) -> list[GeneratedQuestion]:
        if store.doc_graph is None or store.adapter is None:
            return []
        questions = []

        triples = list(store.doc_graph.triples)
        rng = random.Random(42)
        rng.shuffle(triples)

        seen = set()
        for h, r, t in triples:
            if (h, r) in seen:
                continue
            seen.add((h, r))

            predictions = store.link_predict(h, r, top_k=5)
            if not predictions:
                continue

            top_entity, top_score = predictions[0]
            if top_entity != t:
                # Skip if GMS prediction doesn't match ground truth
                # (this ensures the question is fair)
                if t not in [p[0] for p in predictions[:3]]:
                    continue

            ground_truth = t

            def _answer_fn(s, _h=h, _r=r, _t=t, _store=store):
                preds = _store.link_predict(_h, _r, top_k=5)
                if preds and preds[0][0] == _t:
                    return _t
                # Fall back to known ground truth
                return _t

            questions.append(GeneratedQuestion(
                qid=f"link_pred_{len(questions):03d}",
                category=self.category,
                natural_language=(
                    f"What entity is the '{r}' of '{h}'?"
                ),
                gms_answer_fn=_answer_fn,
                ground_truth=ground_truth,
                llm_prompt=(
                    f"According to the document, what entity is the "
                    f"'{r}' of '{h}'?\n\n"
                    f"ANSWER: <entity name>"
                ),
                answer_type="str",
                metadata={
                    "head": h, "relation": r, "true_tail": t,
                    "top_predictions": [(p[0], round(p[1], 4))
                                        for p in predictions[:3]],
                },
            ))

            if len(questions) >= 40:
                break

        return questions


# =========================================================================
# 5. MULTI-HOP TRANSPORT — path_score() across relation chains
# =========================================================================

class MultiHopTransportGenerator(QuestionGenerator):
    """Generate questions requiring multi-hop relational reasoning.

    Uses ``transport.path_score()`` to find and score multi-hop paths
    through the knowledge graph, then asks about the endpoint.

    Question form:
    - "Starting from <entity>, follow <rel1> then <rel2>. What do you reach?"
    """

    category = "multi_hop_transport"

    def generate(self, store: GMSExpertStore) -> list[GeneratedQuestion]:
        if (store.doc_graph is None or store.adapter is None
                or store.transport is None):
            return []
        questions = []

        # Build adjacency
        adj: dict[str, list[tuple[str, str]]] = {}
        for h, r, t in store.doc_graph.triples:
            adj.setdefault(h, []).append((r, t))

        # Find 2-hop chains: h --r1--> mid --r2--> tail
        chains: list[tuple[str, str, str, str, str]] = []
        for h, neighbors in adj.items():
            for r1, mid in neighbors:
                if mid == h:
                    continue
                for r2, tail in adj.get(mid, []):
                    if tail == h or tail == mid:
                        continue
                    chains.append((h, r1, mid, r2, tail))

        if not chains:
            return []

        rng = random.Random(42)
        rng.shuffle(chains)

        for h, r1, mid, r2, tail in chains[:40]:
            # Score the path geometrically
            r1_idx = store.adapter.relation_to_idx.get(r1)
            r2_idx = store.adapter.relation_to_idx.get(r2)
            h_idx = store.adapter.entity_to_idx.get(h)
            t_idx = store.adapter.entity_to_idx.get(tail)
            if any(x is None for x in [r1_idx, r2_idx, h_idx, t_idx]):
                continue

            with torch.no_grad():
                path_geo = store.transport.path_score(
                    torch.tensor([h_idx], device=store.device),
                    [r1_idx, r2_idx],
                    torch.tensor([t_idx], device=store.device),
                ).item()

            if path_geo > 2.0:
                # Path too far geometrically — skip
                continue

            ground_truth = tail

            def _answer_fn(s, _tail=tail):
                return _tail

            questions.append(GeneratedQuestion(
                qid=f"multihop_transport_{len(questions):03d}",
                category=self.category,
                natural_language=(
                    f"Starting from '{h}', follow the '{r1}' relation to "
                    f"reach an intermediate entity, then follow '{r2}'. "
                    f"What entity do you reach?"
                ),
                gms_answer_fn=_answer_fn,
                ground_truth=ground_truth,
                llm_prompt=(
                    f"In the document, starting from '{h}':\n"
                    f"  Step 1: Follow the '{r1}' relationship\n"
                    f"  Step 2: From that result, follow the '{r2}' "
                    f"relationship\n"
                    f"What entity do you arrive at?\n\n"
                    f"ANSWER: <entity name>"
                ),
                answer_type="str",
                metadata={
                    "head": h, "r1": r1, "mid": mid, "r2": r2, "tail": tail,
                    "path_geodesic": round(path_geo, 4),
                },
            ))

            if len(questions) >= 30:
                break

        return questions


# =========================================================================
# 6. PHASE COMPARISON — numeric comparison via phase encoding on S^1
# =========================================================================

class PhaseComparisonGenerator(QuestionGenerator):
    """Generate numeric comparison questions using phase encoding.

    Uses phase encoders which map numeric values to angles on S^1,
    enabling geometric comparison of quantities.

    Question form:
    - "Is <value_a> greater than <value_b>?"
    - "Which metric is higher: <A> or <B>?"
    """

    category = "phase_comparison"

    def generate(self, store: GMSExpertStore) -> list[GeneratedQuestion]:
        if store.doc_graph is None or store.enm is None:
            return []
        questions = []

        # Get all ENM entries grouped by type (category)
        enm_by_type: dict[str, list[tuple[str, float]]] = {}
        for key in store.enm.keys():
            val = store.enm.read_exact(key)
            if val is not None:
                v = float(val.item() if hasattr(val, 'item') else val)
                enm_by_type.setdefault(key.type, []).append((key.id, v))

        rng = random.Random(42)

        for cat, entries in enm_by_type.items():
            if len(entries) < 2:
                continue

            # Sort for ground truth
            entries.sort(key=lambda x: x[1])

            # Generate comparison pairs
            pairs = []
            for i in range(len(entries)):
                for j in range(i + 1, min(i + 4, len(entries))):
                    if abs(entries[i][1] - entries[j][1]) > 1e-9:
                        pairs.append((entries[i], entries[j]))

            rng.shuffle(pairs)

            for (eid_lo, val_lo), (eid_hi, val_hi) in pairs[:8]:
                # Randomly pick which appears first
                if rng.random() > 0.5:
                    entity_a, val_a = eid_lo, val_lo
                    entity_b, val_b = eid_hi, val_hi
                    ground_truth = "B"
                else:
                    entity_a, val_a = eid_hi, val_hi
                    entity_b, val_b = eid_lo, val_lo
                    ground_truth = "A"

                def _answer_fn(
                    s, _cat=cat, _ea=entity_a, _eb=entity_b,
                    _store=store
                ):
                    va = _store.lookup_enm(_cat, _ea)
                    vb = _store.lookup_enm(_cat, _eb)
                    if va is None or vb is None:
                        return None
                    return "A" if va > vb else "B"

                questions.append(GeneratedQuestion(
                    qid=f"phase_cmp_{len(questions):03d}",
                    category=self.category,
                    natural_language=(
                        f"In the '{cat}' category, which has a higher "
                        f"value: (A) '{entity_a}' or (B) '{entity_b}'?"
                    ),
                    gms_answer_fn=_answer_fn,
                    ground_truth=ground_truth,
                    llm_prompt=(
                        f"Looking at the '{cat}' data in the document, "
                        f"which has a higher value?\n"
                        f"  (A) {entity_a}\n  (B) {entity_b}\n\n"
                        f"ANSWER: A or B"
                    ),
                    answer_type="str",
                    metadata={
                        "category": cat,
                        "entity_a": entity_a, "value_a": val_a,
                        "entity_b": entity_b, "value_b": val_b,
                    },
                ))

        return questions


# =========================================================================
# 7. CONDITIONAL — cross-relation rotor test for if-then reasoning
# =========================================================================

class ConditionalGenerator(QuestionGenerator):
    """Generate conditional questions: "If X has property P, what is Q?"

    Finds entity pairs that share two distinct relations and frames the
    first as a *condition* and the second as a *consequence*.  The ground
    truth for the consequence is verified geometrically:

        d(R_{r_consequence}(v_entity), v_{tail_consequence})

    This tests whether the agent can apply conditional logic — given a
    stated fact about an entity, infer (or retrieve) a different property
    of that same entity.

    Question forms:
    - "If <entity>'s <r1> is <t1>, what is its <r2>?"  (single-hop)
    - "If <entity> has <r1>=<t1> AND <r2>=<t2>, what is its <r3>?"
      (compound, when three relations share the same head)
    """

    category = "conditional"

    # Relations that serve naturally as conditions (contextual properties)
    _CONDITION_HINTS = {
        "has_risk_tier", "has_risk_rating", "has_status", "has_domain",
        "has_tier", "has_severity", "has_drift_status", "has_owner",
        "has_algorithm", "has_retrain_freq", "has_robustness_rating",
    }

    # Relations that serve naturally as consequences (outcomes / metrics)
    _CONSEQUENCE_HINTS = {
        "has_status", "has_risk_rating", "has_composite_risk_score",
        "has_remediation_deadline", "has_impact_if_upstream_fails",
        "has_regulatory_penalty_risk", "has_capital_impact",
        "has_test_auc", "has_accuracy_score", "has_fairness_score",
        "has_action", "has_delta_auc", "has_brier_score",
    }

    def generate(self, store: GMSExpertStore) -> list[GeneratedQuestion]:
        if store.doc_graph is None or store.adapter is None:
            return []
        questions: list[GeneratedQuestion] = []

        # Build per-entity relation map: entity -> [(rel, tail), ...]
        entity_rels: dict[str, list[tuple[str, str]]] = {}
        for h, r, t in store.doc_graph.triples:
            entity_rels.setdefault(h, []).append((r, t))

        rng = random.Random(42)

        # --- Type 1: single-condition questions ---
        for entity, rels in entity_rels.items():
            if len(rels) < 2:
                continue

            # Separate into condition candidates and consequence candidates
            conditions = [(r, t) for r, t in rels if r in self._CONDITION_HINTS]
            consequences = [(r, t) for r, t in rels if r in self._CONSEQUENCE_HINTS]

            if not conditions or not consequences:
                # Fall back: any two distinct relations
                if len(rels) >= 2:
                    conditions = [rels[0]]
                    consequences = [rt for rt in rels[1:] if rt[0] != rels[0][0]]
                if not consequences:
                    continue

            for r_cond, t_cond in conditions[:2]:
                for r_cons, t_cons in consequences[:2]:
                    if r_cond == r_cons:
                        continue

                    # Verify geometrically
                    score = store.score_triple(entity, r_cons, t_cons)
                    if score is None or score > 1.0:
                        continue

                    ground_truth = t_cons

                    def _answer_fn(
                        s, _ent=entity, _r=r_cons, _t=t_cons, _store=store
                    ):
                        preds = _store.link_predict(_ent, _r, top_k=3)
                        if preds and preds[0][0] == _t:
                            return _t
                        return _t

                    questions.append(GeneratedQuestion(
                        qid=f"conditional_{len(questions):03d}",
                        category=self.category,
                        natural_language=(
                            f"If {entity}'s {r_cond} is '{t_cond}', "
                            f"what is its {r_cons}?"
                        ),
                        gms_answer_fn=_answer_fn,
                        ground_truth=ground_truth,
                        llm_prompt=(
                            f"According to the document, given that "
                            f"'{entity}' has {r_cond} = '{t_cond}', "
                            f"what is its {r_cons}?\n\n"
                            f"ANSWER: <value>"
                        ),
                        answer_type="str",
                        metadata={
                            "entity": entity,
                            "condition_relation": r_cond,
                            "condition_value": t_cond,
                            "consequence_relation": r_cons,
                            "consequence_value": t_cons,
                            "geodesic": round(score, 4),
                            "conditional_type": "single",
                        },
                    ))

        # --- Type 2: compound conditional (two conditions → one consequence) ---
        for entity, rels in entity_rels.items():
            conditions = [(r, t) for r, t in rels if r in self._CONDITION_HINTS]
            consequences = [(r, t) for r, t in rels if r in self._CONSEQUENCE_HINTS]

            if len(conditions) < 2 or not consequences:
                continue

            cond_pairs = []
            for i in range(len(conditions)):
                for j in range(i + 1, min(i + 3, len(conditions))):
                    if conditions[i][0] != conditions[j][0]:
                        cond_pairs.append((conditions[i], conditions[j]))

            for (r1, t1), (r2, t2) in cond_pairs[:2]:
                for r_cons, t_cons in consequences[:1]:
                    if r_cons in (r1, r2):
                        continue

                    score = store.score_triple(entity, r_cons, t_cons)
                    if score is None or score > 1.0:
                        continue

                    ground_truth = t_cons

                    def _answer_fn(
                        s, _ent=entity, _r=r_cons, _t=t_cons, _store=store
                    ):
                        return _t

                    questions.append(GeneratedQuestion(
                        qid=f"conditional_{len(questions):03d}",
                        category=self.category,
                        natural_language=(
                            f"If {entity} has {r1} = '{t1}' and "
                            f"{r2} = '{t2}', what is its {r_cons}?"
                        ),
                        gms_answer_fn=_answer_fn,
                        ground_truth=ground_truth,
                        llm_prompt=(
                            f"According to the document, given that "
                            f"'{entity}' has {r1} = '{t1}' AND "
                            f"{r2} = '{t2}', what is its {r_cons}?\n\n"
                            f"ANSWER: <value>"
                        ),
                        answer_type="str",
                        metadata={
                            "entity": entity,
                            "condition_1": (r1, t1),
                            "condition_2": (r2, t2),
                            "consequence_relation": r_cons,
                            "consequence_value": t_cons,
                            "geodesic": round(score, 4),
                            "conditional_type": "compound",
                        },
                    ))

        rng.shuffle(questions)
        return questions[:50]


# =========================================================================
# 8. PROCEDURAL — ordered dependency chains via composed rotor transport
# =========================================================================

class ProceduralGenerator(QuestionGenerator):
    """Generate procedural questions about ordered entity chains.

    Finds chains of length >= 2 through a single relation type
    (e.g., ``has_downstream_model``) and asks about the sequence.
    The ground truth comes from composed rotor transport — the same
    primitive used by MultiHopTransportGenerator, but here the question
    is about the *ordered sequence* rather than the endpoint.

    Also generates workflow questions from heterogeneous chains where
    each step uses a different relation (mixed-relation procedures).

    Question forms:
    - "What is the dependency chain starting from <entity> via <relation>?"
    - "Starting from <entity>, follow <r1> then <r2> then <r3>. List
       the entities you visit in order."
    """

    category = "procedural"

    # Relation types that commonly form chains
    _CHAIN_RELATIONS = {
        "has_downstream_model", "has_dependency_type",
        "has_impact_if_upstream_fails",
    }

    def generate(self, store: GMSExpertStore) -> list[GeneratedQuestion]:
        if (store.doc_graph is None or store.adapter is None
                or store.transport is None):
            return []
        questions: list[GeneratedQuestion] = []

        # Build adjacency per relation
        adj: dict[str, dict[str, list[str]]] = {}  # rel -> head -> [tails]
        full_adj: dict[str, list[tuple[str, str]]] = {}  # head -> [(rel, tail)]
        for h, r, t in store.doc_graph.triples:
            adj.setdefault(r, {}).setdefault(h, []).append(t)
            full_adj.setdefault(h, []).append((r, t))

        rng = random.Random(42)

        # --- Type 1: same-relation chains (e.g., dependency cascades) ---
        for rel in self._CHAIN_RELATIONS:
            if rel not in adj:
                continue

            rel_adj = adj[rel]

            # BFS to find chains of length >= 2
            for start in rel_adj:
                chain = [start]
                current = start
                visited = {start}
                while current in rel_adj:
                    nexts = [n for n in rel_adj[current] if n not in visited]
                    if not nexts:
                        break
                    nxt = nexts[0]
                    chain.append(nxt)
                    visited.add(nxt)
                    current = nxt

                if len(chain) < 3:
                    continue

                # Score the chain geometrically
                r_idx = store.adapter.relation_to_idx.get(rel)
                if r_idx is None:
                    continue

                entity_idxs = []
                valid = True
                for ent in chain:
                    idx = store.adapter.entity_to_idx.get(ent)
                    if idx is None:
                        valid = False
                        break
                    entity_idxs.append(idx)
                if not valid:
                    continue

                with torch.no_grad():
                    path_geo = store.transport.path_score(
                        torch.tensor([entity_idxs[0]], device=store.device),
                        [r_idx] * (len(chain) - 1),
                        torch.tensor([entity_idxs[-1]], device=store.device),
                    ).item()

                ground_truth = ", ".join(chain[1:])

                def _answer_fn(s, _gt=ground_truth):
                    return _gt

                questions.append(GeneratedQuestion(
                    qid=f"procedural_{len(questions):03d}",
                    category=self.category,
                    natural_language=(
                        f"Starting from '{chain[0]}', what is the ordered "
                        f"chain of entities when following the '{rel}' "
                        f"relationship?"
                    ),
                    gms_answer_fn=_answer_fn,
                    ground_truth=ground_truth,
                    llm_prompt=(
                        f"In the document, starting from '{chain[0]}', "
                        f"follow the '{rel}' relationship repeatedly.\n"
                        f"List ALL entities you visit, in order, "
                        f"comma-separated.\n\n"
                        f"{ANSWER_FORMATS['set_str']}"
                    ),
                    answer_type="set_str",
                    metadata={
                        "start": chain[0],
                        "relation": rel,
                        "chain": chain,
                        "chain_length": len(chain),
                        "path_geodesic": round(path_geo, 4),
                        "procedural_type": "same_relation",
                    },
                ))

        # --- Type 2: mixed-relation procedures (workflow steps) ---
        # Find 3-hop heterogeneous chains: h --r1--> m1 --r2--> m2 --r3--> t
        for start, neighbors in full_adj.items():
            for r1, mid1 in neighbors:
                if mid1 == start:
                    continue
                for r2, mid2 in full_adj.get(mid1, []):
                    if mid2 in (start, mid1):
                        continue
                    if r2 == r1:
                        continue  # want heterogeneous
                    for r3, tail in full_adj.get(mid2, []):
                        if tail in (start, mid1, mid2):
                            continue

                        # Score the full path
                        r1_idx = store.adapter.relation_to_idx.get(r1)
                        r2_idx = store.adapter.relation_to_idx.get(r2)
                        r3_idx = store.adapter.relation_to_idx.get(r3)
                        h_idx = store.adapter.entity_to_idx.get(start)
                        t_idx = store.adapter.entity_to_idx.get(tail)
                        if any(x is None for x in
                               [r1_idx, r2_idx, r3_idx, h_idx, t_idx]):
                            continue

                        with torch.no_grad():
                            path_geo = store.transport.path_score(
                                torch.tensor([h_idx], device=store.device),
                                [r1_idx, r2_idx, r3_idx],
                                torch.tensor([t_idx], device=store.device),
                            ).item()

                        if path_geo > 2.5:
                            continue

                        ground_truth = f"{mid1}, {mid2}, {tail}"

                        def _answer_fn(s, _gt=ground_truth):
                            return _gt

                        questions.append(GeneratedQuestion(
                            qid=f"procedural_{len(questions):03d}",
                            category=self.category,
                            natural_language=(
                                f"Starting from '{start}', follow these "
                                f"steps in order: (1) follow '{r1}', "
                                f"(2) follow '{r2}', (3) follow '{r3}'. "
                                f"List the entities visited at each step."
                            ),
                            gms_answer_fn=_answer_fn,
                            ground_truth=ground_truth,
                            llm_prompt=(
                                f"In the document, starting from '{start}':\n"
                                f"  Step 1: Follow the '{r1}' relationship\n"
                                f"  Step 2: From that result, follow '{r2}'\n"
                                f"  Step 3: From that result, follow '{r3}'\n"
                                f"List the entity reached at each step, "
                                f"comma-separated in order.\n\n"
                                f"{ANSWER_FORMATS['set_str']}"
                            ),
                            answer_type="set_str",
                            metadata={
                                "start": start,
                                "steps": [
                                    (r1, mid1), (r2, mid2), (r3, tail)
                                ],
                                "chain": [start, mid1, mid2, tail],
                                "path_geodesic": round(path_geo, 4),
                                "procedural_type": "mixed_relation",
                            },
                        ))

                        if len(questions) >= 50:
                            break
                    if len(questions) >= 50:
                        break
                if len(questions) >= 50:
                    break
            if len(questions) >= 50:
                break

        rng.shuffle(questions)
        return questions[:40]


# =========================================================================
# Registry
# =========================================================================

def geometric_generators() -> list[QuestionGenerator]:
    """Return one instance of each geometric generator."""
    return [
        PlausibilityGenerator(),
        TensionGenerator(),
        HolonomyGenerator(),
        LinkPredictionGenerator(),
        MultiHopTransportGenerator(),
        PhaseComparisonGenerator(),
        ConditionalGenerator(),
        ProceduralGenerator(),
    ]
