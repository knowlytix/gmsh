# SPDX-License-Identifier: Apache-2.0
"""Base classes for auto-generated benchmark questions.

A generated question is admitted by a *functionality check* (``_validate``):
the question is trusted only if its answer can be recovered from the store and
that recovered answer equals the stated ground truth. By default the recovery is
symbolic --- re-run the generator's ``gms_answer_fn`` against the store's index
and compare. With ``geometric=True`` the check is verified a *second* way: the
answer's fact triple must also be recoverable through the trained GMS geometry.
A question survives only when the symbolic oracle and the geometric path agree
--- the two-way recovery of the testing methodology.

The geometric recovery has two parts, both free of any hand-set threshold:

  * **Ranking (calibration-free).** ``score_triple(head, relation, *)`` must rank
    the true tail strictly first among every tail the relation connects to. This
    is a pure comparison --- no operating point.
  * **Plausibility (calibrated, read from the store).** The true triple's geodesic
    distance must sit under the store's own *calibrated* accept cut, persisted in
    ``calibration.json`` (``thresholds.geodesic``, fitted by ``GMSJudge.calibrate``
    with a false-accept ceiling). The cut is never defaulted or hard-coded; if the
    store carries no calibrated cut, the absolute gate recuses and only the
    calibration-free ranking applies (recorded on ``last_recovery_note``).
"""

import json
import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable
import random


@dataclass
class GeneratedQuestion:
    """A single auto-generated benchmark question."""
    qid: str                                    # e.g. "exact_recall_007"
    category: str                               # question category
    natural_language: str                        # human-readable question
    gms_answer_fn: Callable                     # function(store) -> answer
    ground_truth: Any                           # expected answer
    llm_prompt: str                             # prompt for the LLM
    answer_type: str                            # "float", "int", "bool", "set_str", "str"
    metadata: dict = field(default_factory=dict)  # source keys/triples


# LLM answer format instructions per type
ANSWER_FORMATS = {
    "float": "Report the EXACT value with ALL decimal places. Format: ANSWER: <number>",
    "int": "Format: ANSWER: <integer>",
    "bool": "Format: ANSWER: true or false",
    "set_str": "List ALL items, comma-separated. Format: ANSWER: item1, item2, ...",
    "str": "Format: ANSWER: <exact string>",
}


class QuestionGenerator(ABC):
    """Base class for question generators that mine a GMSStore.

    Parameters
    ----------
    geometric : bool
        When True, ``_validate`` additionally requires the answer's fact triple
        to be recovered by the GMS geometry: ``score_triple`` ranks the true tail
        first (calibration-free) and the true triple's distance is under the
        store's calibrated accept cut (read from the store, never defaulted).
        Default False, so existing (symbolic-only) behavior is unchanged.
    """

    category: str = ""

    def __init__(self, geometric: bool = False):
        self.geometric = geometric
        self.last_recovery_note = None       # transparency: why the absolute gate recused
        # auditable split of how admitted questions were verified (no silent skips)
        self.recovery_stats = {"two_way": 0, "symbolic": 0, "rejected": 0}

    @abstractmethod
    def generate(self, store) -> list[GeneratedQuestion]:
        """Mine the store and produce candidate questions."""
        ...

    def sample(self, candidates: list[GeneratedQuestion],
               max_questions: int, seed: int = 42) -> list[GeneratedQuestion]:
        """Deterministic sampling from candidate pool."""
        if len(candidates) <= max_questions:
            return candidates
        rng = random.Random(seed)
        return rng.sample(candidates, max_questions)

    def _validate(self, store, question: GeneratedQuestion) -> bool:
        """Verify the question is recoverable and its answer matches.

        Step 1 (symbolic, always): re-run ``gms_answer_fn`` and compare to the
        stated ground truth. Step 2 (geometric, only when ``self.geometric``):
        every answer triple the question exposes must be recovered by the GMS
        geometry. A question survives only when both agree.
        """
        try:
            answer = question.gms_answer_fn(store)
            if answer is None:
                return False
            if isinstance(question.ground_truth, float):
                symbolic_ok = abs(answer - question.ground_truth) < 1e-9
            else:
                symbolic_ok = answer == question.ground_truth
            if not symbolic_ok:
                return False
            if not self.geometric:
                return True
            return self._geometric_recovers(store, question)
        except Exception:
            return False

    # -- geometric (two-way) recovery --------------------------------------
    def _geometric_recovers(self, store, question: GeneratedQuestion) -> bool:
        """True iff the GMS geometry recovers every answer triple this question
        exposes. The recovery mode is recorded on ``question.metadata['_recovery']``
        and tallied on ``self.recovery_stats`` so nothing is verified silently:

          * ``two_way``         -- fact triples derived and all recovered;
          * ``two_way:rejected``-- fact triples derived but the geometry refused;
          * ``symbolic:<why>``  -- no fact triple exists (an aggregation such as a
                                   count, an ordering or a boolean comparison, or a
                                   pure-ENM chain), so the symbolic recompute is the
                                   complete check and geometry has no primitive.

        A question is admitted under ``two_way`` or ``symbolic`` but never silently:
        the per-question tag makes the split auditable in the emitted corpus."""
        triples = self._answer_triples(store, question)
        if not triples:
            mode = f"symbolic:{self._symbolic_reason(question)}"
            question.metadata["_recovery"] = mode
            self.recovery_stats["symbolic"] += 1
            return True
        ok = all(self._recover_triple(store, h, r, t) for h, r, t in triples)
        question.metadata["_recovery"] = "two_way" if ok else "two_way:rejected"
        self.recovery_stats["two_way" if ok else "rejected"] += 1
        return ok

    @staticmethod
    def _symbolic_reason(question: GeneratedQuestion) -> str:
        """Why a question carries no geometrically-recoverable fact triple --- so
        the recusal to symbolic-only is justified, not hidden."""
        cat = question.category
        if cat == "counting":
            return "cardinality aggregation; no fact triple"
        if cat == "contradiction":
            return "pass/fail aggregation (count/set/bool); no fact triple"
        if isinstance(question.ground_truth, bool):
            return "boolean comparison; the answer is not a recoverable tail"
        return "no fact triple in the answer (e.g. pure-ENM chain)"

    def _recover_triple(self, store, head, relation, true_tail) -> bool:
        """Geometric recovery of one fact triple, safe for many-to-many relations.

        A head may legitimately hold several tails of one relation (``has_alias``
        carries every alias), so the true tail is *not* required to out-rank the
        head's other true tails. Recovery is judged against the **non-asserted**
        tails of the relation --- the true negatives:

          * **Plausibility (calibrated, per relation):** ``score_triple`` must sit
            under the relation's *learned admissible radius* ``cap_radius(relation)``
            --- the operating point the cap geometry fitted during training. When
            the store is not cap-trained, the persisted global geodesic cut is used;
            with neither, the absolute gate recuses (ranking still applies).
          * **Discrimination (calibration-free):** it must be strictly more
            plausible than every tail the relation connects to that this head does
            *not* assert (a decoy). With no decoys the relation is unambiguous and
            discrimination is vacuous.

        A question with neither a calibrated cut nor any decoy cannot be checked
        geometrically and is refused (the caller records the recusal)."""
        cut = self._calibrated_cut(store, relation)
        try:
            d_true = store.score_triple(head, relation, true_tail)
        except Exception:
            d_true = None
        if d_true is None:                       # unscoreable tail -> not recoverable
            return False
        # plausibility under the calibrated cut
        if cut is not None and d_true > cut:
            return False
        # decoys = tails of this relation NOT asserted for this head (true negatives)
        asserted = {t for _, _, t in store.query_triples(head=head, relation=relation)}
        decoys = {t for _, _, t in store.query_triples(relation=relation)} - asserted
        decoys.discard(true_tail)
        decoy_scores = []
        for t in decoys:
            try:
                d = store.score_triple(head, relation, t)
            except Exception:
                d = None
            if d is not None:
                decoy_scores.append(d)
        if decoy_scores:
            return d_true < min(decoy_scores)
        # no decoys: rely on the calibrated plausibility verdict; refuse only if we
        # had no calibrated cut to lean on (nothing geometric was actually tested).
        return cut is not None

    def _calibrated_cut(self, store, relation):
        """The calibrated plausibility cut for ``relation`` --- never defaulted.

        Preference order, most principled first:
          1. ``cap_radius(relation)`` --- the per-relation admissible radius the cap
             geometry learned during training (the geometry's own operating point);
          2. the store's persisted global ``thresholds.geodesic`` (GMSJudge-fitted
             with a false-accept ceiling), read from ``calibration.json``;
          3. ``None`` --- no calibrated cut; the absolute gate recuses (ranking only).
        """
        try:
            rho = store.cap_radius(relation)
        except Exception:
            rho = None
        if rho is not None:
            self.last_recovery_note = f"cap_radius[{relation}]={rho:.4f} (learned)"
            return float(rho)
        cut = self._persisted_geodesic_cut(store)
        if cut is not None:
            self.last_recovery_note = f"persisted geodesic cut {cut:.4f} (not cap-trained)"
        return cut

    def _persisted_geodesic_cut(self, store):
        """The store's persisted global geodesic accept cut from
        ``calibration.json`` (``thresholds.geodesic``), or None if absent."""
        sp = (getattr(store, "store_path", None)
              or getattr(getattr(store, "config", None), "store_path", None)
              or getattr(getattr(store, "cfg", None), "store_path", None))
        if sp is None:
            return None
        f = pathlib.Path(sp) / "calibration.json"
        if not f.exists():
            return None
        try:
            return float(json.loads(f.read_text())["thresholds"]["geodesic"])
        except Exception:
            return None

    def _answer_triples(self, store, question: GeneratedQuestion):
        """Derive the (head, relation, tail) fact triples whose recovery the
        geometry should confirm. Generators may publish them directly in
        ``metadata['answer_triples']``; otherwise we derive them for value-recall
        questions from the ENM entity and the recalled value by matching the
        graph triple that carries that value."""
        explicit = question.metadata.get("answer_triples")
        if explicit:
            return [tuple(x) for x in explicit]

        gt = question.ground_truth

        # value-recall: (head from enm_id) + the recalled value -> the graph triple
        enm_id = question.metadata.get("enm_id")
        if enm_id is not None and isinstance(gt, (int, float)):
            head = enm_id.split("/")[0]
            value = float(gt)
            out = []
            for h, r, t in store.query_triples(head=head):
                try:
                    if abs(float(t) - value) < 1e-9:
                        out.append((h, r, t))
                except (TypeError, ValueError):
                    continue
            return out

        # multi-hop pattern 1: ground_truth = (head, value, tail_set), metadata
        # carries the second-hop relation. The argmin/argmax (first hop) is a
        # symbolic aggregation the geometry cannot verify, but each second-hop
        # relationship triple (head, relation, tail) is geometrically recoverable.
        rel = question.metadata.get("relation")
        if rel and isinstance(gt, tuple) and len(gt) == 3:
            head, _value, tails = gt
            if isinstance(tails, (set, list, tuple)):
                return [(head, rel, str(t)) for t in tails]

        # cross-reference set overlap: ground_truth is the set of head entities
        # holding BOTH relations. Each member's two relationship triples are
        # geometrically recoverable -- confirming the entity genuinely carries
        # both relations, not just that the symbolic index lists it.
        r1, r2 = question.metadata.get("relation_1"), question.metadata.get("relation_2")
        if r1 and r2 and isinstance(gt, (set, frozenset)):
            out = []
            for h in gt:
                for rel in (r1, r2):
                    out.extend(store.query_triples(head=h, relation=rel))
            return out

        return []
