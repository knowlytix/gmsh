# SPDX-License-Identifier: Apache-2.0
"""WorkflowPlanner --- GMS-guided step selection for governed agents.

Uses the knowledge graph's workflow triples (``precedes``, ``requires``,
``in_phase``) together with ``link_predict`` and ``check_holonomy`` to
select, order and validate workflow steps.  No geometry is duplicated:
every tensor computation is delegated to the existing :class:`GMSExpertStore`
and :class:`RelationalTransport` primitives.

The planner answers four questions:

1. **What next?**  ``next_step()`` uses ``link_predict(current, "precedes", top_k)``
   then filters candidates whose prerequisites are not yet satisfied.
2. **Full plan?**  ``full_plan()`` iteratively calls ``next_step()`` until the
   goal entity is reached or no more candidates remain.
3. **Is this plan consistent?**  ``validate_plan()`` checks holonomy defect
   for the ``precedes`` chain and verifies all prerequisites.
4. **Is this step ready?**  ``is_step_ready()`` queries the GMS graph for
   ``requires`` relations and checks them against completed steps.

Built in Chapter 9 of "How to Build Policy-Governed AI Agents."
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from knowlytix.knowledge.store import GMSExpertStore

# ------------------------------------------------------------------ #
# Result data classes
# ------------------------------------------------------------------ #

@dataclass
class PlannedStep:
    """One step in a workflow plan."""
    tool: str
    confidence: float
    reason: str


@dataclass
class PlanValidation:
    """Result of validating a workflow plan.

    The baseline fields (``holonomy``, ``prerequisites_satisfied``,
    ``violations``) are always populated.  The calibrated fields are
    populated only after :meth:`WorkflowPlanner.calibrate` has been
    called or a calibration has been loaded from disk.
    """

    holonomy: float
    prerequisites_satisfied: bool
    violations: list[str] = field(default_factory=list)

    # Calibrated decision (populated when thresholds are loaded).
    verdict: str | None = None  # "accept" | "review" | "reject"
    confidence: float | None = None  # probability plan is valid, in [0, 1]
    calibrated_features: dict[str, float] | None = None

    # Second-attestor (policy engine) result; populated when
    # :meth:`WorkflowPlanner.attach_policy` has bound a PlanPolicy.
    policy_verdict: str | None = None  # "accept" | "reject"
    policy_reasons: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ #
# Workflow constants
# ------------------------------------------------------------------ #

# Canonical step ordering (ground truth from the knowledge document).
# Used as a fallback when the GMS graph has no ``precedes`` edges.
CANONICAL_ORDER: list[str] = [
    "load_data",
    "preprocess_data",
    "feature_select",
    "train_model",
    "diagnose_accuracy",
    "diagnose_robustness",
    "diagnose_fairness",
    "explain_importance",
    "explain_pdp",
    "explain_shap",
    "compare_models",
    "release_gate",
    "deploy",
]

# Map of step -> required prior steps (ground truth).
PREREQUISITES: dict[str, list[str]] = {
    "load_data": [],
    "preprocess_data": ["load_data"],
    "feature_select": ["preprocess_data"],
    "train_model": ["preprocess_data"],
    "diagnose_accuracy": ["train_model"],
    "diagnose_robustness": ["diagnose_accuracy"],
    "diagnose_fairness": ["diagnose_robustness"],
    "explain_importance": ["train_model"],
    "explain_pdp": ["train_model"],
    "explain_shap": ["train_model"],
    "compare_models": ["diagnose_accuracy"],
    "release_gate": ["compare_models"],
    "deploy": ["release_gate"],
    "predict": ["train_model"],
}

# Phase assignments
STEP_PHASE: dict[str, str] = {
    "load_data": "Data",
    "preprocess_data": "Data",
    "feature_select": "Data",
    "train_model": "Training",
    "predict": "Training",
    "diagnose_accuracy": "Diagnostics",
    "diagnose_robustness": "Diagnostics",
    "diagnose_fairness": "Diagnostics",
    "explain_importance": "Explanation",
    "explain_pdp": "Explanation",
    "explain_shap": "Explanation",
    "compare_models": "Evaluation",
    "release_gate": "Deployment",
    "deploy": "Deployment",
}


# ------------------------------------------------------------------ #
# WorkflowPlanner
# ------------------------------------------------------------------ #

class WorkflowPlanner:
    """GMS-guided workflow planner for governed agents.

    Wraps :class:`GMSExpertStore` primitives to plan data science
    workflows.  All geometric computation is delegated to the store.

    Parameters
    ----------
    store : GMSExpertStore
        A loaded expert store with workflow triples ingested.
    device : str or torch.device
        Compute device.
    """

    def __init__(
        self,
        store: GMSExpertStore,
        device: str | torch.device = "cpu",
    ):
        self.store = store
        self.device = (
            torch.device(device) if isinstance(device, str) else device
        )
        # Calibration state --- populated by :meth:`calibrate` or
        # :meth:`load_calibration`.  ``_classifier`` is a fitted
        # :class:`sklearn.linear_model.LogisticRegression`; ``_thresholds``
        # stores the accept/reject probability cut-points.
        self._classifier: Any | None = None
        self._thresholds: dict[str, float] = {}
        self._feature_names: list[str] = []
        self._calibration_meta: dict[str, Any] = {}
        # Optional second attestor.
        self._policy: Any | None = None
        # Per-planner cache for score_triple calls (plans in a cohort
        # share many pairs --- avoid repeating the GPU forward pass).
        self._score_cache: dict[tuple[str, str, str], float | None] = {}

    # -------------------------------------------------------------- #
    # Internal helpers
    # -------------------------------------------------------------- #

    def _resolve_step(self, name: str) -> str | None:
        """Resolve a step name via exact match or fuzzy fallback."""
        if self.store.entity_exists(name):
            return name
        return self.store.fuzzy_match_entity(name)

    def _gms_successors(self, step: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Find successor steps using link_predict on 'precedes'."""
        resolved = self._resolve_step(step)
        if resolved is None:
            return []
        return self.store.link_predict(resolved, "precedes", top_k=top_k)

    def _canonical_successors(self, step: str) -> list[str]:
        """Fallback: successors from the canonical ordering."""
        successors = []
        for s, prereqs in PREREQUISITES.items():
            if step in prereqs:
                successors.append(s)
        return successors

    def _get_prerequisites(self, step: str) -> list[str]:
        """Get prerequisites for a step from GMS + fallback.

        Queries the GMS graph for ``(step, requires, ?)`` triples first,
        then falls back to the hardcoded PREREQUISITES table.
        """
        # Try GMS graph first
        resolved = self._resolve_step(step)
        if resolved is not None:
            gms_reqs = self.store.query_triples(head=resolved, relation="requires")
            if gms_reqs:
                return [t for _, _, t in gms_reqs]

        # Fallback to canonical prerequisites
        return PREREQUISITES.get(step, [])

    # -------------------------------------------------------------- #
    # Public API
    # -------------------------------------------------------------- #

    def next_step(
        self,
        completed_steps: list[str],
        context: str = "",
    ) -> PlannedStep:
        """Suggest the next workflow step.

        Uses ``link_predict(last_step, "precedes", top_k)`` to rank
        candidate next steps, then filters out candidates whose
        prerequisites have not been satisfied.

        Parameters
        ----------
        completed_steps : list of str
            Steps already completed (in order).
        context : str
            Optional context hint (e.g., ``"classification_ready"``).

        Returns
        -------
        PlannedStep
            The recommended next step with confidence and reason.
        """
        completed_set = set(completed_steps)

        # If nothing completed yet, start at load_data
        if not completed_steps:
            score = self.store.score_triple("load_data", "precedes", "preprocess_data")
            confidence = math.exp(-score) if score is not None else 0.8
            return PlannedStep(
                tool="load_data",
                confidence=min(confidence, 1.0),
                reason="First step in the workflow; no prerequisites needed.",
            )

        last_step = completed_steps[-1]

        # Try GMS link prediction
        candidates = self._gms_successors(last_step, top_k=15)

        # Filter to known workflow steps and remove already completed
        workflow_steps = set(CANONICAL_ORDER)
        filtered = []
        for entity, dist in candidates:
            # Fuzzy match to canonical names
            matched = None
            for ws in workflow_steps:
                if ws == entity or ws in entity or entity in ws:
                    matched = ws
                    break
            if matched and matched not in completed_set:
                ready, _ = self.is_step_ready(matched, completed_steps)
                if ready:
                    confidence = math.exp(-dist) if dist > 0 else 1.0
                    filtered.append((matched, confidence, dist))

        if filtered:
            # Sort by confidence (descending)
            filtered.sort(key=lambda x: x[1], reverse=True)
            best_step, best_conf, best_dist = filtered[0]
            return PlannedStep(
                tool=best_step,
                confidence=min(best_conf, 1.0),
                reason=(
                    f"link_predict({last_step}, precedes) -> {best_step} "
                    f"(geodesic={best_dist:.4f})"
                ),
            )

        # Fallback: use canonical ordering
        for step in CANONICAL_ORDER:
            if step not in completed_set:
                ready, missing = self.is_step_ready(step, completed_steps)
                if ready:
                    return PlannedStep(
                        tool=step,
                        confidence=0.5,
                        reason=(
                            f"Canonical fallback after {last_step}; "
                            f"prerequisites satisfied."
                        ),
                    )

        # Nothing more to do
        return PlannedStep(
            tool="deploy",
            confidence=0.3,
            reason="All known steps completed; suggesting deployment.",
        )

    def full_plan(
        self,
        goal: str,
        context: str = "",
        max_steps: int = 15,
    ) -> list[PlannedStep]:
        """Generate a full workflow plan toward a goal.

        Iteratively calls :meth:`next_step` to build a plan from scratch
        until the goal step is included or no more candidates remain.

        Parameters
        ----------
        goal : str
            The target step or state (e.g., ``"deployment_ready"``).
        context : str
            Optional context hint.
        max_steps : int
            Safety limit on plan length.

        Returns
        -------
        list of PlannedStep
            Ordered sequence of planned steps.
        """
        plan: list[PlannedStep] = []
        completed: list[str] = []

        # Map goal aliases
        goal_step = goal
        goal_aliases = {
            "deployment_ready": "release_gate",
            "model_trained": "train_model",
            "data_ready": "preprocess_data",
            "evaluated": "compare_models",
            "diagnosed": "diagnose_fairness",
        }
        goal_step = goal_aliases.get(goal, goal)

        for _ in range(max_steps):
            step = self.next_step(completed, context)

            # Avoid infinite loops
            if step.tool in completed:
                break

            plan.append(step)
            completed.append(step.tool)

            # Check if we reached the goal
            if step.tool == goal_step:
                break

        return plan

    def validate_plan(self, plan: list[str]) -> PlanValidation:
        """Validate a workflow plan for prerequisite satisfaction and consistency.

        Always populates the baseline fields (``holonomy``,
        ``prerequisites_satisfied``, ``violations``).  When the planner
        has been calibrated (via :meth:`calibrate` or
        :meth:`load_calibration`), also populates ``verdict``,
        ``confidence`` and ``calibrated_features``.  When a policy
        co-attestor has been attached via :meth:`attach_policy`, also
        populates ``policy_verdict`` and ``policy_reasons``.

        Parameters
        ----------
        plan : list of str
            Ordered list of step names.

        Returns
        -------
        PlanValidation
            Baseline + calibrated + policy verdict fields.
        """
        base = self._baseline_validation(plan)

        # Calibrated decision.
        if self.is_calibrated():
            feats = self.extract_features(plan)
            import numpy as np

            x = np.array(
                [[feats[name] for name in self._feature_names]], dtype=float,
            )
            prob_valid = float(self._classifier.predict_proba(x)[0, 1])
            tau_acc = self._thresholds["tau_accept"]
            tau_rej = self._thresholds["tau_reject"]
            if prob_valid >= tau_acc:
                verdict = "accept"
            elif prob_valid <= tau_rej:
                verdict = "reject"
            else:
                verdict = "review"
            base.verdict = verdict
            base.confidence = prob_valid
            base.calibrated_features = feats

        # Policy co-attestor.
        if self._policy is not None:
            pol = self._policy.check_plan(plan)
            base.policy_verdict = "accept" if pol.allowed else "reject"
            base.policy_reasons = list(pol.reasons)

        return base

    def _compute_plan_holonomy(self, plan: list[str]) -> float:
        """Compute holonomy defect for a plan's ``precedes`` chain.

        Composes the ``precedes`` rotor along all consecutive pairs and
        compares to the direct ``precedes`` rotor.  Uses the store's
        ``check_holonomy`` primitive.
        """
        if len(plan) < 2:
            return 0.0

        # Build the relation path: each consecutive pair uses "precedes"
        relation_path = ["precedes"] * (len(plan) - 1)

        # Check if "precedes" relation exists in the store
        if not self.store.relation_exists("precedes"):
            # Fallback: score each consecutive pair and measure inconsistency
            return self._fallback_holonomy(plan)

        defect = self.store.check_holonomy(relation_path, "precedes")
        if defect is not None:
            return defect

        # Fallback if holonomy computation fails
        return self._fallback_holonomy(plan)

    def _fallback_holonomy(self, plan: list[str]) -> float:
        """Fallback holonomy: score each consecutive pair and penalize bad ordering."""
        scores = []
        penalty = 0.0

        for i in range(len(plan) - 1):
            step_a = plan[i]
            step_b = plan[i + 1]

            # Score the (step_a, precedes, step_b) triple
            score = self.store.score_triple(step_a, "precedes", step_b)
            if score is not None:
                scores.append(score)

            # Penalize prerequisite violations
            prereqs_b = self._get_prerequisites(step_b)
            completed_so_far = set(plan[:i + 1])
            for req in prereqs_b:
                if req not in completed_so_far:
                    penalty += 0.5

        if scores:
            mean_score = sum(scores) / len(scores)
            # Convert to holonomy-like metric: higher = worse
            # Good plans have low mean geodesic -> low holonomy
            holonomy = mean_score * (1.0 + penalty)
        else:
            # Pure prerequisite-based scoring
            total_prereqs = 0
            satisfied = 0
            for i, step in enumerate(plan):
                prereqs = self._get_prerequisites(step)
                completed_so_far = set(plan[:i])
                total_prereqs += len(prereqs)
                satisfied += sum(1 for r in prereqs if r in completed_so_far)

            if total_prereqs > 0:
                violation_rate = 1.0 - (satisfied / total_prereqs)
                holonomy = violation_rate * 2.0  # Scale to [0, 2]
            else:
                holonomy = 0.0

        return holonomy

    def is_step_ready(
        self,
        step: str,
        completed: list[str],
    ) -> tuple[bool, list[str]]:
        """Check if a step's prerequisites are satisfied.

        Queries the GMS graph for ``(step, requires, ?)`` triples
        and checks whether each required step appears in the completed
        list.

        Parameters
        ----------
        step : str
            The step to check.
        completed : list of str
            Steps already completed.

        Returns
        -------
        (ready, missing_prereqs)
            True if all prerequisites are met; list of missing ones.
        """
        prereqs = self._get_prerequisites(step)
        completed_set = set(completed)
        missing = [r for r in prereqs if r not in completed_set]
        return (len(missing) == 0, missing)

    # -------------------------------------------------------------- #
    # Informational
    # -------------------------------------------------------------- #

    def get_workflow_steps(self) -> list[str]:
        """Return the canonical workflow step ordering."""
        return list(CANONICAL_ORDER)

    def get_phase(self, step: str) -> str:
        """Return the phase a step belongs to."""
        return STEP_PHASE.get(step, "Unknown")

    # -------------------------------------------------------------- #
    # Feature extraction (shared between validate_plan and calibrate)
    # -------------------------------------------------------------- #

    # Required canonical steps to reach release_gate via the minimal
    # valid plan (used by the ``coverage`` feature).
    _REQUIRED_CORE_STEPS: tuple[str, ...] = (
        "load_data",
        "preprocess_data",
        "train_model",
        "diagnose_accuracy",
        "compare_models",
        "release_gate",
    )

    # Feature vector order --- kept stable so saved classifiers stay
    # compatible with future loads.
    FEATURE_NAMES: tuple[str, ...] = (
        "holonomy",
        "n_violations",
        "n_unknown_steps",
        "core_coverage",
        "mean_precedes_score",
        "plan_length_delta",
    )

    def extract_features(self, plan: list[str]) -> dict[str, float]:
        """Compute the feature vector used for calibrated plan validation.

        The features below combine the prerequisite-based ``holonomy`` and
        ``violations`` from :meth:`validate_plan` with three signals the
        uncalibrated validator misses: unknown-step count, canonical-core
        coverage and a geometric ordering score from the store's
        ``has_precedes`` operator.  These together separate all six
        failure classes in the cohort.
        """
        # Baseline signals --- reuse the unmodified validate_plan path.
        base = self._baseline_validation(plan)

        holonomy = float(base.holonomy)
        n_violations = float(len(base.violations))

        # Unknown-step count: steps neither in CANONICAL_ORDER nor
        # resolvable via store fuzzy match.
        canonical_set = set(CANONICAL_ORDER) | {"predict"}
        unknown = 0
        for step in plan:
            if step in canonical_set:
                continue
            if self._resolve_step(step) is None:
                unknown += 1
        n_unknown = float(unknown)

        # Core coverage: fraction of _REQUIRED_CORE_STEPS present.
        present = sum(1 for s in self._REQUIRED_CORE_STEPS if s in plan)
        core_coverage = present / len(self._REQUIRED_CORE_STEPS)

        # Mean precedes score on consecutive pairs (try has_precedes first,
        # fall back to precedes).  Lower geodesic = better ordering.
        # Cached per (head, relation, tail) --- most plans share pairs.
        scores: list[float] = []
        relation = self._best_precedes_relation()
        if relation is not None:
            for i in range(len(plan) - 1):
                s = self._cached_score_triple(plan[i], relation, plan[i + 1])
                if s is not None:
                    scores.append(float(s))
        mean_precedes = sum(scores) / len(scores) if scores else 1.0

        # Plan length delta vs the canonical length (12).  Large negative
        # = skipped steps; large positive = hallucinations/inserts.
        plan_length_delta = float(len(plan) - len(CANONICAL_ORDER) + 1)

        return {
            "holonomy": holonomy,
            "n_violations": n_violations,
            "n_unknown_steps": n_unknown,
            "core_coverage": core_coverage,
            "mean_precedes_score": mean_precedes,
            "plan_length_delta": plan_length_delta,
        }

    def _baseline_validation(self, plan: list[str]) -> PlanValidation:
        """Run the uncalibrated validate_plan logic exactly as before.

        Used by :meth:`extract_features` to avoid recursion when the
        calibrated :meth:`validate_plan` is active.
        """
        violations: list[str] = []
        completed_so_far: list[str] = []
        for step in plan:
            prereqs = self._get_prerequisites(step)
            for req in prereqs:
                if req not in completed_so_far:
                    violations.append(
                        f"{step} requires {req}, but {req} not yet completed"
                    )
            completed_so_far.append(step)
        prereqs_ok = len(violations) == 0
        holonomy = self._compute_plan_holonomy(plan)
        return PlanValidation(
            holonomy=holonomy,
            prerequisites_satisfied=prereqs_ok,
            violations=violations,
        )

    def _cached_score_triple(
        self, head: str, relation: str, tail: str,
    ) -> float | None:
        """Memoized ``score_triple`` --- a cohort scan reuses many pairs."""
        key = (head, relation, tail)
        if key in self._score_cache:
            return self._score_cache[key]
        s = self.store.score_triple(head, relation, tail)
        val = float(s) if s is not None else None
        self._score_cache[key] = val
        return val

    def _best_precedes_relation(self) -> str | None:
        """Return whichever of ``precedes`` or ``has_precedes`` the store knows.

        The extractor often prefixes relations with ``has_``, so we try
        both.  If neither exists the feature degrades gracefully.
        """
        for name in ("precedes", "has_precedes"):
            if self.store.relation_exists(name):
                return name
        return None

    # -------------------------------------------------------------- #
    # Calibration
    # -------------------------------------------------------------- #

    def calibrate(
        self,
        cohort: list,
        seed: int = 42,
        test_fraction: float = 0.3,
        review_band: float = 0.15,
    ) -> dict[str, Any]:
        """Fit a logistic classifier on labeled plans and derive thresholds.

        Mirrors the pattern used by :class:`knowlytix.harness.testing.judge.GMSJudge`:
        fit a classifier on plan features, derive scalar probability
        cut-points (``tau_reject``, ``tau_accept``) that partition the
        outcome space into accept / review / reject, and report held-out
        FPR, FNR and bootstrap 95 % CIs.

        Parameters
        ----------
        cohort : list of PlanRecord-like
            Each record must have ``plan`` and ``label`` attributes.
            Label ``"valid"`` is treated as the positive class.
        seed : int
            RNG seed for the train/test split and bootstrap.
        test_fraction : float
            Hold-out fraction for reporting FPR/FNR.
        review_band : float
            Width of the ``review`` band around the 0.5 decision
            boundary.  Plans with probability in
            ``[0.5 - review_band, 0.5 + review_band]`` are routed to
            human review rather than auto-accept/reject.

        Returns
        -------
        dict
            Calibration summary with keys ``fpr``, ``fnr``, ``fpr_ci``,
            ``fnr_ci``, ``tau_accept``, ``tau_reject``, ``n_train``,
            ``n_test`` and ``feature_weights``.
        """
        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "scikit-learn and numpy are required for calibrate()"
            ) from e

        if not cohort:
            raise ValueError("Empty cohort passed to calibrate()")

        # Build feature matrix + label vector.
        feats = [self.extract_features(rec.plan) for rec in cohort]
        X = np.array(
            [[f[name] for name in self.FEATURE_NAMES] for f in feats],
            dtype=float,
        )
        y = np.array(
            [1 if rec.label == "valid" else 0 for rec in cohort],
            dtype=int,
        )

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_fraction, random_state=seed, stratify=y,
        )

        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed,
        )
        clf.fit(X_tr, y_tr)

        # Held-out metrics.
        p_te = clf.predict_proba(X_te)[:, 1]
        pred_accept = p_te >= (0.5 + review_band)
        pred_reject = p_te <= (0.5 - review_band)
        # False positive = invalid plan accepted.
        fp = int(((y_te == 0) & pred_accept).sum())
        tn = int(((y_te == 0) & ~pred_accept).sum())
        fpr = fp / max(fp + tn, 1)
        # False negative = valid plan rejected.
        fn = int(((y_te == 1) & pred_reject).sum())
        tp = int(((y_te == 1) & ~pred_reject).sum())
        fnr = fn / max(fn + tp, 1)

        # Bootstrap 95% CIs (percentile method).
        rng = np.random.default_rng(seed)
        fpr_bs, fnr_bs = [], []
        for _ in range(1000):
            idx = rng.integers(0, len(y_te), len(y_te))
            y_b = y_te[idx]
            p_b = p_te[idx]
            ac = p_b >= (0.5 + review_band)
            rj = p_b <= (0.5 - review_band)
            fp_b = ((y_b == 0) & ac).sum()
            tn_b = ((y_b == 0) & ~ac).sum()
            fpr_bs.append(fp_b / max(fp_b + tn_b, 1))
            fn_b = ((y_b == 1) & rj).sum()
            tp_b = ((y_b == 1) & ~rj).sum()
            fnr_bs.append(fn_b / max(fn_b + tp_b, 1))
        fpr_ci = (float(np.percentile(fpr_bs, 2.5)), float(np.percentile(fpr_bs, 97.5)))
        fnr_ci = (float(np.percentile(fnr_bs, 2.5)), float(np.percentile(fnr_bs, 97.5)))

        # Store state.
        self._classifier = clf
        self._feature_names = list(self.FEATURE_NAMES)
        self._thresholds = {
            "tau_accept": 0.5 + review_band,
            "tau_reject": 0.5 - review_band,
            "review_band": review_band,
        }
        self._calibration_meta = {
            "seed": seed,
            "test_fraction": test_fraction,
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "fpr": float(fpr),
            "fnr": float(fnr),
            "fpr_ci": fpr_ci,
            "fnr_ci": fnr_ci,
            "feature_weights": {
                name: float(w)
                for name, w in zip(self.FEATURE_NAMES, clf.coef_[0])
            },
            "intercept": float(clf.intercept_[0]),
            "cohort_size": len(cohort),
        }

        return {
            "tau_accept": self._thresholds["tau_accept"],
            "tau_reject": self._thresholds["tau_reject"],
            "fpr": float(fpr),
            "fnr": float(fnr),
            "fpr_ci": fpr_ci,
            "fnr_ci": fnr_ci,
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "feature_weights": self._calibration_meta["feature_weights"],
        }

    def is_calibrated(self) -> bool:
        """Return True if a classifier has been fit or loaded."""
        return self._classifier is not None

    def save_calibration(self, path: str | Path) -> str:
        """Persist classifier + thresholds to ``path``.

        Writes ``calibration.json`` (thresholds + metadata) and
        ``calibration.pkl`` (pickled classifier) mirroring the
        :class:`GMSJudge` persistence format.
        """
        if not self.is_calibrated():
            raise RuntimeError(
                "Planner has not been calibrated yet; call calibrate() first."
            )
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        meta = {
            "feature_names": self._feature_names,
            "thresholds": self._thresholds,
            "meta": self._calibration_meta,
        }
        (p / "calibration.json").write_text(json.dumps(meta, indent=2))
        with (p / "calibration.pkl").open("wb") as f:
            pickle.dump(self._classifier, f)
        return str(p)

    def load_calibration(self, path: str | Path) -> bool:
        """Restore classifier + thresholds from a previous ``save_calibration``.

        Returns True on success, False if files are missing.
        """
        p = Path(path)
        json_path = p / "calibration.json"
        pkl_path = p / "calibration.pkl"
        if not json_path.exists() or not pkl_path.exists():
            return False
        meta = json.loads(json_path.read_text())
        self._feature_names = list(meta["feature_names"])
        self._thresholds = dict(meta["thresholds"])
        self._calibration_meta = dict(meta["meta"])
        with pkl_path.open("rb") as f:
            self._classifier = pickle.load(f)
        return True

    # -------------------------------------------------------------- #
    # Second attestor (policy co-attestor)
    # -------------------------------------------------------------- #

    def attach_policy(self, policy: Any) -> None:
        """Bind a plan-level policy engine as an independent co-attestor.

        The policy object must expose ``check_plan(plan) -> PolicyVerdict``
        where ``PolicyVerdict`` has ``allowed: bool`` and
        ``reasons: list[str]`` attributes.
        """
        self._policy = policy

    def __repr__(self) -> str:
        s = self.store.stats()
        return (
            f"WorkflowPlanner(entities={s['entities']}, "
            f"relations={s['relations']}, "
            f"triples={s['triples']})"
        )
