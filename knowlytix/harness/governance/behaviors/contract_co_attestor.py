# SPDX-License-Identifier: Apache-2.0
"""Holonomy-based co-attestor for behavioral contracts.

The FSM contract from :mod:`contracts` enforces \emph{structural}
invariants: a transition not in $\delta$ cannot fire, a tool not in
$\Pi(q)$ cannot run.  But many sequences pass the FSM and still look
unusual --- e.g.\ a fairness check before robustness, repeated
preprocessing rounds, or an unusual interleaving of comparison and
diagnostic calls.  These are not structural violations but they may
warrant a human review.

This module provides a second attestor that runs alongside the FSM
with three-state output rather than the FSM's binary pass/fail:

  * **pass** --- FSM accepts and the geometric deviation from a
    canonical reference is below the warning threshold.
  * **flag** --- FSM accepts but holonomy / edit-distance from the
    canonical sequence exceeds the threshold.  The orchestrator
    routes the run to review rather than blocking outright; the
    audit log records the deviation score.
  * **block** --- FSM rejects.  Same as the structural verdict; the
    co-attestor's view is irrelevant once the FSM has fired.

Three-state output (rather than OR-semantics on blocking, as in
:class:`LLMCoAttestor`) reflects the different role each gate plays.
The FSM is structural and binary --- ``cannot do X.''  Holonomy is
behavioral and continuous --- ``this is unusual; have a human
look.''  Conflating the two would either destroy structural
guarantees (block on holonomy alone) or hide the deviation signal
(treat holonomy as advisory only).

The threshold is calibrated from the systematic cohort: pass every
``allowed`` probe through holonomy scoring against a canonical
reference, take the empirical CDF, and pick the threshold at the
upper percentile (default 95).  The bundle records the calibrated
threshold so a different reference produces a different signed
artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from knowlytix.harness.governance.behaviors.contracts import (
    BehavioralContract,
    ProcessEvaluator,
)


@dataclass(frozen=True)
class TwoAttestorContractVerdict:
    """Combined verdict from FSM + holonomy co-attestor.

    ``decision`` is ``pass`` only when both attestors are clean;
    ``flag`` when the FSM accepts but holonomy exceeds threshold;
    ``block`` when the FSM rejects (holonomy is not consulted in
    that case --- the structural verdict is final).
    """

    decision: Literal["pass", "flag", "block"]
    fsm_valid: bool
    fsm_violations: tuple[str, ...]
    holonomy_score: float | None
    holonomy_threshold: float | None
    flagged_reason: str | None


@dataclass
class HolonomyCoAttestor:
    """Continuous-deviation co-attestor for FSM-valid sequences.

    Parameters
    ----------
    reference_sequence : list[str]
        The canonical (approved) tool sequence.  Holonomy is
        measured against this reference.
    threshold : float
        Sequences whose score exceeds this are flagged for review.
        ``calibrate_threshold`` populates this from the cohort.
    store : GMSExpertStore | None
        Optional GMS store with a trained ``precedes`` relation.
        When present, holonomy is computed via the manifold's
        check_holonomy.  Otherwise, falls back to normalized edit
        distance (the fallback in
        :meth:`ProcessEvaluator.check_holonomy`).
    """

    reference_sequence: tuple[str, ...]
    threshold: float = 0.5
    store: object | None = None

    def score(self, actual: list[str]) -> float:
        """Return the holonomy / edit-distance score for ``actual``."""
        ev = ProcessEvaluator(_dummy_contract())
        return ev.check_holonomy(
            list(actual), list(self.reference_sequence), store=self.store,
        )

    def verdict(
        self,
        contract: BehavioralContract,
        actual: list[str],
        evaluator: ProcessEvaluator | None = None,
    ) -> TwoAttestorContractVerdict:
        """Run both attestors and return the combined verdict.

        Order: the FSM runs first.  When the FSM rejects, the
        holonomy score is not computed --- ``decision='block'`` is
        final regardless of how the sequence scores geometrically.
        """
        if evaluator is None:
            evaluator = ProcessEvaluator(contract)

        fsm = evaluator.validate_sequence(actual, contract)
        if not fsm.valid:
            return TwoAttestorContractVerdict(
                decision="block",
                fsm_valid=False,
                fsm_violations=tuple(fsm.violations),
                holonomy_score=None,
                holonomy_threshold=self.threshold,
                flagged_reason=None,
            )

        score = self.score(actual)
        if score > self.threshold:
            return TwoAttestorContractVerdict(
                decision="flag",
                fsm_valid=True,
                fsm_violations=tuple(),
                holonomy_score=score,
                holonomy_threshold=self.threshold,
                flagged_reason=(
                    f"holonomy={score:.4f} exceeds "
                    f"threshold={self.threshold:.4f}"
                ),
            )

        return TwoAttestorContractVerdict(
            decision="pass",
            fsm_valid=True,
            fsm_violations=tuple(),
            holonomy_score=score,
            holonomy_threshold=self.threshold,
            flagged_reason=None,
        )

    def calibrate_threshold(
        self,
        valid_sequences: Iterable[list[str]],
        *,
        percentile: float = 0.95,
    ) -> float:
        """Set the threshold to the ``percentile`` quantile of scores.

        ``valid_sequences`` should be operator-curated approved
        workflows --- not the adversarial cohort's edge-case probes
        (empty sequence, single-tool sequence) which inflate the
        upper percentile and would never flag anything.  The
        :func:`canonical_compliant_workflows` helper provides a
        default set for the data-science contracts.
        """
        scores = sorted(self.score(list(s)) for s in valid_sequences)
        if not scores:
            return self.threshold
        n = len(scores)
        idx = max(0, min(n - 1, int(percentile * n)))
        self.threshold = scores[idx]
        return self.threshold


def canonical_compliant_workflows() -> list[list[str]]:
    """Operator-curated reference workflows for the DS contracts.

    These are full-length, structurally compliant sequences that an
    auditor would consider canonical.  They cover the major
    diagnostic / explanation orderings without including degenerate
    short sequences.  Use as the calibration set for the holonomy
    threshold so the percentile reflects deviations among real
    workflows rather than between full workflows and edge cases.
    """
    return [
        # Canonical: data, preprocess, train, robustness, fairness, leaderboard.
        ["load_dataset", "preprocess_data", "train_model",
         "diagnose_robustness", "diagnose_fairness", "leaderboard"],
        # Same with EDA prefix.
        ["load_dataset", "data_summary", "eda_correlation",
         "preprocess_data", "train_model",
         "diagnose_robustness", "diagnose_fairness", "leaderboard"],
        # SHAP explanation in place of robustness diagnostic.
        ["load_dataset", "preprocess_data", "train_model",
         "explain_shap", "diagnose_fairness", "leaderboard"],
        # Importance + PDP explanation pair.
        ["load_dataset", "preprocess_data", "train_model",
         "explain_importance", "explain_pdp",
         "diagnose_fairness", "leaderboard"],
        # Feature-select before training.
        ["load_dataset", "preprocess_data", "feature_select",
         "train_model", "diagnose_robustness",
         "diagnose_fairness", "leaderboard"],
    ]


def _dummy_contract():
    """A throwaway contract whose only role is to satisfy
    :class:`ProcessEvaluator`'s constructor; ``check_holonomy``
    does not consult it."""
    from knowlytix.harness.governance.behaviors.contracts import advisory_contract
    return advisory_contract()
