# SPDX-License-Identifier: Apache-2.0
"""Completeness: the recall complement to claim correctness.

The typed verifier scores *correctness* -- of the claims an answer asserts, are
they grounded in the GMS (precision, via the answer-driven
:class:`GeometricClaimExtractor`). Completeness asks the dual question: of the
facts the answer *should* contain for this question, how many did it cover
(recall)?

The expected facts are the question's ground-truth atoms -- the entities and
the asked value(s) the answer is supposed to state. We keep only atoms that are
*store-grounded* (a known entity, or a numeric value), so margins / booleans /
bookkeeping fields in a generator's ground_truth don't pollute the denominator,
and we test coverage with the same store-bound matching the claim extractor
uses (snake_case/spaced entities; numeric value-equality). Completeness is a
separate axis: a confident, fully-correct-but-partial answer scores high on
correctness and low here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore

__all__ = ["CompletenessReport", "CompletenessEvaluator"]


@dataclass
class CompletenessReport:
    """Coverage of a question's expected (ground-truth) atoms by the answer."""

    expected: int = 0
    covered: int = 0
    missing: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Recall in [0, 1]; 1.0 when nothing was expected (vacuously complete)."""
        return 1.0 if self.expected == 0 else self.covered / self.expected

    def as_dict(self) -> dict:
        return {"expected": self.expected, "covered": self.covered,
                "missing": list(self.missing), "score": round(self.score, 4)}


class CompletenessEvaluator:
    """Score how completely an answer covers a question's expected atoms.

    Parameters
    ----------
    store : GMSExpertStore
        Provides the entity vocabulary used to keep only store-grounded
        expected atoms and to bind matches.
    """

    def __init__(self, store: "GMSExpertStore"):
        from knowlytix.harness.testing.claims import GeometricClaimExtractor
        self.store = store
        # Reuse the answer-driven extractor's store-bound matching (numeric
        # value-equality + snake_case/spaced entity matching).
        self._match = GeometricClaimExtractor(store)

    def evaluate(self, answer_text: str, ground_truth,
                 question_metadata: dict | None = None) -> CompletenessReport:
        """Score how many store-grounded expected atoms the answer covers.

        Args:
            answer_text: the answer to score.
            ground_truth: the question's ground truth, flattened into expected atoms.
            question_metadata: optional metadata contributing head/entity anchors.

        Returns:
            A :class:`CompletenessReport` with expected and covered counts and the
            missing atoms.
        """
        atoms = self._expected_atoms(ground_truth, question_metadata or {})
        if not atoms:
            return CompletenessReport()
        text = answer_text or ""
        low = text.lower()
        covered, missing = 0, []
        for atom in atoms:
            if self._match._mentions_tail(atom, low, text):
                covered += 1
            else:
                missing.append(atom)
        return CompletenessReport(expected=len(atoms), covered=covered,
                                  missing=missing)

    # -- internals ---------------------------------------------------------

    def _expected_atoms(self, ground_truth, meta: dict) -> list[str]:
        """Store-grounded atoms the answer should contain.

        Flattens the ground truth (and the metadata head/entity anchor), then
        keeps only numeric values and known store entities -- so a generator's
        margins, booleans, and schema bookkeeping don't inflate the denominator.
        """
        raw: list = []
        self._flatten(ground_truth, raw)
        for key in ("head", "entity", "entity_id", "base"):
            if meta.get(key) is not None:
                raw.append(meta[key])
        atoms: list[str] = []
        seen: set[str] = set()
        for item in raw:
            s = str(item).strip()
            if not s or s.lower() in seen:
                continue
            if self._is_store_grounded(s):
                atoms.append(s)
                seen.add(s.lower())
        return atoms

    @staticmethod
    def _flatten(obj, out: list) -> None:
        if obj is None or isinstance(obj, bool):
            return
        if isinstance(obj, (list, tuple, set)):
            for x in obj:
                CompletenessEvaluator._flatten(x, out)
        elif isinstance(obj, dict):
            for x in obj.values():
                CompletenessEvaluator._flatten(x, out)
        else:
            out.append(obj)

    def _is_store_grounded(self, s: str) -> bool:
        """Keep numerics (asked values) and store-known entities; drop noise."""
        try:
            float(s)
            return True
        except ValueError:
            pass
        if len(s) <= 1:
            return False
        adapter = getattr(self.store, "adapter", None)
        if adapter is not None:
            ent = getattr(adapter, "entity_to_idx", {}) or {}
            cand = {s, s.replace(" ", "_"), s.replace("_", " ")}
            if any(c in ent for c in cand):
                return True
        match = getattr(self.store, "fuzzy_match_entity", None)
        return bool(match(s)) if callable(match) else False
